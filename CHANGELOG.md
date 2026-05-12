# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
