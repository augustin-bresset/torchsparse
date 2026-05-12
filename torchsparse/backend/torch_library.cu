// TorchScript registration for TorchSparse.
//
// Registers:
//   - torchsparse::SparseTensorImpl   (torch custom class via m.class_<T>)
//   - torchsparse::scatter_conv_forward (TORCH_LIBRARY op, CUDA impl)
//
// The forward pass uses GatherScatter (inference-only, no autograd).
// It mirrors the Python path in:
//   torchsparse/nn/functional/conv/kmap/func/hashmap.py
//   torchsparse/nn/functional/conv/func/gather_scatter.py
// without the TensorCache.

#include <torch/library.h>
#include <torch/torch.h>

#include "sparse_tensor_class.h"
#include "convolution/convolution_gather_scatter_cuda.h"
#include "hashmap/hashmap_cuda.cuh"
#include "others/sparsemapping_cuda.h"

namespace {

// ────────────────────────────────────────────────────────────────────────────
// Output coordinate computation (Minkowski floor-division mode)
// coords: [N, 4] int32, layout [batch, x, y, z]
// ────────────────────────────────────────────────────────────────────────────
static at::Tensor compute_out_coords(const at::Tensor& coords,
                                     const std::vector<int64_t>& stride_vec) {
    auto coords_f = coords.to(at::kFloat);
    // xyz columns: narrow(dim=1, start=1, length=3)
    auto xyz_f = coords_f.narrow(1, 1, 3);

    auto stride_f = at::tensor(
        std::vector<float>{(float)stride_vec[0],
                           (float)stride_vec[1],
                           (float)stride_vec[2]},
        at::TensorOptions().dtype(at::kFloat).device(coords.device())
    ).unsqueeze(0);  // [1, 3]

    auto xyz_down = at::floor(xyz_f / stride_f).to(at::kInt);  // [N, 3]
    auto batch    = coords.narrow(1, 0, 1);                     // [N, 1]
    auto all      = at::cat({batch, xyz_down}, 1).contiguous(); // [N, 4]

    return std::get<0>(
        at::unique_dim(all, 0, /*sorted=*/true,
                       /*return_inverse=*/false,
                       /*return_counts=*/false)
    );
}

// ────────────────────────────────────────────────────────────────────────────
// High-level GatherScatter forward (inference only, no gradient)
//
// Args:
//   feats       [N, C_in]             float32 or float16
//   coords      [N, 4]                int32, [batch, x, y, z]
//   weight      [K_vol, C_in, C_out]
//   kernel_size [3], stride [3], padding [3]
//   subm        true → submanifold (output coords = input coords, stride=1)
//
// Returns: (out_feats [N_out, C_out], out_coords [N_out, 4])
// ────────────────────────────────────────────────────────────────────────────
std::tuple<at::Tensor, at::Tensor> scatter_conv_forward(
    const at::Tensor& feats,
    const at::Tensor& coords,
    const at::Tensor& weight,
    std::vector<int64_t> kernel_size_vec,
    std::vector<int64_t> stride_vec,
    std::vector<int64_t> padding_vec,
    bool subm)
{
    TORCH_CHECK(feats.device().is_cuda(),
                "torchsparse::scatter_conv_forward requires CUDA tensors");
    TORCH_CHECK(coords.scalar_type() == at::kInt, "coords must be int32");
    TORCH_CHECK(weight.dim() == 3, "weight must be [kernel_vol, C_in, C_out]");

    const int N_in      = (int)coords.size(0);
    const int kernel_vol = (int)weight.size(0);

    // 1. Output coordinates
    at::Tensor out_coords = subm ? coords : compute_out_coords(coords, stride_vec);
    const int N_out = (int)out_coords.size(0);

    // 2. Build hashtable and insert input coords in [x,y,z,batch] order
    //    hashmap expects layout [x, y, z, batch]
    auto in_xyz   = coords.narrow(1, 1, 3);               // [N, 3]
    auto in_batch = coords.narrow(1, 0, 1);               // [N, 1]
    auto in_xyzb  = at::cat({in_xyz, in_batch}, 1).contiguous(); // [N, 4]

    // Use 64-bit hashtable: hashtable32 truncates hash_func_64b to int32,
    // which causes key collisions for large point clouds and produces
    // nondeterministic wrong in_idx values in the gather step.
    // hashtable (GPUHashTable<int64_t, int>) keeps the full 64-bit hash key.
    //
    // Allocate via ATen so the zeros fill runs on the PyTorch CUDA stream,
    // preventing a race with stream-0 cudaMemset in the default ctor.
    auto key_opts = at::TensorOptions().dtype(at::kLong).device(coords.device());
    auto val_opts = at::TensorOptions().dtype(at::kInt).device(coords.device());
    auto tbl_keys = at::zeros({2 * N_in}, key_opts);
    auto tbl_vals = at::zeros({2 * N_in}, val_opts);
    hashtable table(tbl_keys, tbl_vals);
    table.insert_coords(in_xyzb);

    // 3. Lookup: for each output coord × kernel offset → input index
    auto out_xyz   = out_coords.narrow(1, 1, 3);
    auto out_batch = out_coords.narrow(1, 0, 1);
    auto out_xyzb  = at::cat({out_xyz, out_batch}, 1).contiguous(); // [N_out, 4]

    auto opts_i = at::TensorOptions().dtype(at::kInt).device(coords.device());
    auto ks_t   = at::tensor(std::vector<int>{(int)kernel_size_vec[0],
                                               (int)kernel_size_vec[1],
                                               (int)kernel_size_vec[2]}, opts_i);
    auto str_t  = at::tensor(std::vector<int>{(int)stride_vec[0],
                                               (int)stride_vec[1],
                                               (int)stride_vec[2]}, opts_i);

    // out_in_map: [N_out_padded, kernel_vol], 1-indexed (0 = no neighbor)
    auto out_in_map_raw = table.lookup_coords(out_xyzb, ks_t, str_t, kernel_vol);
    // Truncate padding and go to 0-indexed (-1 = invalid)
    auto out_in_map = out_in_map_raw.narrow(0, 0, N_out) - 1; // [N_out, kernel_vol]

    // 4. Build nbmaps / nbsizes (GatherScatter format)
    //    results: [kernel_vol, N_out]
    auto results = out_in_map.t().contiguous();
    auto valid   = (results != -1);
    auto nbsizes = valid.to(at::kInt).sum(1).to(at::kInt); // [kernel_vol] — sum widens to int64; cast back
    auto nbmaps  = valid.nonzero().contiguous();   // [M, 2]: (k_pos, out_idx)

    if (nbmaps.size(0) > 0) {
        // Replace kernel position (col 0) with actual in_idx from results
        // flat index = k_pos * N_out + out_idx
        auto flat     = results.reshape(-1);
        auto col0     = nbmaps.narrow(1, 0, 1).squeeze(1);  // [M] k_pos
        auto col1     = nbmaps.narrow(1, 1, 1).squeeze(1);  // [M] out_idx
        auto flat_idx = (col0 * (int)results.size(1) + col1).to(at::kLong);
        auto in_idx   = flat.index_select(0, flat_idx);
        nbmaps.narrow(1, 0, 1).squeeze(1).copy_(in_idx);
    }
    nbmaps = nbmaps.to(at::kInt).contiguous();

    // 5. Build input/output masks
    // nbsizes has shape [kernel_vol] — pass it whole (matches Python behaviour:
    // nbsizes[:N_out] in Python is a no-op when kernel_vol <= N_out)
    auto masks = build_mask_from_kmap(
        N_in, N_out,
        nbmaps,
        nbsizes.contiguous()
    );
    auto input_mask  = masks[0];
    auto output_mask = masks[1];

    // 6. GatherScatter conv (conv_mode=0 → basic path, no benchmark buffer)
    at::Tensor buffer = at::zeros({0}, feats.options());
    auto out_feats = conv_forward_gather_scatter_cuda(
        feats.contiguous(),
        weight.contiguous(),
        nbmaps,
        nbsizes.cpu().contiguous(),
        input_mask,
        output_mask,
        N_out,
        0.0f,  // epsilon
        0,     // mm_thresh
        0,     // conv_mode=0: basic (no benchmark buffer)
        false, // transposed
        buffer
    );

    return {out_feats, out_coords};
}

} // anonymous namespace

// ────────────────────────────────────────────────────────────────────────────
// TORCH_LIBRARY registration
// ────────────────────────────────────────────────────────────────────────────
TORCH_LIBRARY(torchsparse, m) {
    // Custom class — use m.class_<T> inside TORCH_LIBRARY
    m.class_<SparseTensorImpl>("SparseTensorImpl")
        .def(torch::init<at::Tensor, at::Tensor,
                         std::vector<int64_t>,
                         c10::optional<std::vector<int64_t>>>())
        .def("feats",         &SparseTensorImpl::get_feats)
        .def("coords",        &SparseTensorImpl::get_coords)
        .def("stride",        &SparseTensorImpl::get_stride)
        .def("spatial_range", &SparseTensorImpl::get_spatial_range)
        .def("set_feats",     &SparseTensorImpl::set_feats)
        .def("set_coords",    &SparseTensorImpl::set_coords)
        // Pickle support required for torch.jit.save / torch.jit.load
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<SparseTensorImpl>& self)
                -> std::tuple<at::Tensor, at::Tensor,
                              std::vector<int64_t>,
                              c10::optional<std::vector<int64_t>>>
            {
                return {self->feats, self->coords,
                        self->stride, self->spatial_range};
            },
            // __setstate__
            [](std::tuple<at::Tensor, at::Tensor,
                          std::vector<int64_t>,
                          c10::optional<std::vector<int64_t>>> state)
                -> c10::intrusive_ptr<SparseTensorImpl>
            {
                return c10::make_intrusive<SparseTensorImpl>(
                    std::get<0>(state), std::get<1>(state),
                    std::get<2>(state), std::get<3>(state));
            }
        );

    // Op schema
    m.def("scatter_conv_forward("
          "Tensor feats, Tensor coords, Tensor weight, "
          "int[] kernel_size, int[] stride, int[] padding, "
          "bool subm) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torchsparse, CUDA, m) {
    m.impl("scatter_conv_forward", &scatter_conv_forward);
}
