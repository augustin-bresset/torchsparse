#pragma once

#include <torch/custom_class.h>
#include <torch/torch.h>
#include <vector>

// C++ mirror of the Python SparseTensor, registered as a TorchScript custom
// class.  The TensorCache is intentionally omitted: the scriptable forward
// path recomputes the kernel map from scratch each call.
struct SparseTensorImpl : torch::CustomClassHolder {
    at::Tensor feats;
    at::Tensor coords;
    std::vector<int64_t> stride;
    c10::optional<std::vector<int64_t>> spatial_range;

    SparseTensorImpl() = default;

    SparseTensorImpl(at::Tensor feats,
                     at::Tensor coords,
                     std::vector<int64_t> stride,
                     c10::optional<std::vector<int64_t>> spatial_range)
        : feats(std::move(feats)),
          coords(std::move(coords)),
          stride(std::move(stride)),
          spatial_range(std::move(spatial_range)) {}

    at::Tensor get_feats() const { return feats; }
    at::Tensor get_coords() const { return coords; }
    std::vector<int64_t> get_stride() const { return stride; }
    c10::optional<std::vector<int64_t>> get_spatial_range() const { return spatial_range; }

    void set_feats(at::Tensor f) { feats = std::move(f); }
    void set_coords(at::Tensor c) { coords = std::move(c); }
};
