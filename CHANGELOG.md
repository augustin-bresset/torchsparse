# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-25

### Fixed

* **Empty kernel-map crash (sparse clouds, bs=1 inference)**: Encoder-decoder bottlenecks (MinkUNet, SparseTravNet decoders) at batch size 1 on real sparse frames can downsample until a frame has 0 output voxels, giving a kernel map with `out_in_map.shape == (0, K)`. Every backend launcher computed `grid = (work + tile - 1) / tile` and dispatched a 0-block CUDA grid → `cudaErrorInvalidValue`, which poisons the CUDA context for the whole process (no catch-and-continue). Guarded all six affected launchers (`convert_transposed_out_in_map`, `derive_bitmask_from_out_in_map`, `reorder_out_in_map_cuda`, `reduce_bitmask_cuda`, `conv_forward_implicit_gemm_cuda`, `conv_forward_implicit_gemm_sorted_cuda`) to return the already-allocated empty/identity tensor without launching. Non-empty convolutions are unchanged. Present verbatim in upstream `mit-han-lab/torchsparse`.
* **Hashmap capacity overflow (on-the-fly downsample)**: The `hashmap_on_the_fly` kernel-map builder inserts the *output* coords into the GPU hashtable, but capacity was sized on the *input* count (`hash_rsv_ratio * input_node_num`). With `downsample_mode="spconv"` a strided conv on a scattered sparse cloud expands the point count (up to `input * kernel_volume`; ~3x at stride 2), overflowing the table at the default `hash_rsv_ratio=2` — a silent illegal memory access upstream, a clean error here. Capacity is now sized on `input * kernel_volume` for the downsample path; the subm path is unchanged.

### Testing

* Added `tests/python/test_empty_kmap.py`: standalone regression tests asserting no crash, correct empty shapes, and a surviving CUDA context for each guarded launcher and an end-to-end empty `Conv3d`.
* Added `tests/python/test_hashmap_capacity.py`: a strided on-the-fly conv on a scattered cloud must not overflow the hashtable at the default `hash_rsv_ratio`.

## [1.0.0] - 2026-05-12

### Added

### Added

* **TorchScript Support**: Added full support for model exporting, saving, and loading (`torch.jit.save` / `load`) for sparse convolution layers.
* Registered `SparseTensorImpl` as a TorchScript custom class.
* Exposed `scatter_conv_forward` as a CUDA operator via `TORCH_LIBRARY`.
* Added pickle support (`getstate`/`setstate`) for `SparseTensorImpl` to enable seamless round-trips.


* **ScriptableConv3d**: Updated bias handling to use `Optional[torch.Tensor]` instead of `Optional[nn.Parameter]` for TorchScript type compatibility.

### Fixed

* **Hashmap Collision Stability**: Upgraded `GPUHashTable` from 32-bit to 64-bit keys (`hashtable32` → `hashtable`) to eliminate hash-truncation collisions on large-scale point clouds.
* **CUDA Stream Synchronization**:
* Migrated hashmap storage allocation from raw `cudaMalloc` to ATen (`at::zeros`) to ensure zeroing occurs on the PyTorch CUDA stream.
* Routed all hashmap kernel launches (`insert`/`lookup`) to the current PyTorch stream via `at::cuda::getCurrentCUDAStream()`.
* Added stream synchronization in `check_overflow()` to resolve race conditions between kernel execution and subsequent ATen reads/writes.


* **Shape & Type Corrections**:
* Fixed `nbsizes` shape: now passing the full `[kernel_vol]` tensor to `build_mask_from_kmap` instead of incorrectly narrowing it.
* Added explicit cast for `nbsizes` back to `int32` after ATen summation (which previously widened it to `int64`).



### Testing

* Verified stability with 12 passing tests, covering `SparseConv`, `ToDense`, TorchScript serialization, and Point IO.
