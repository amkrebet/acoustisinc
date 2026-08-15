# Project Rules & Guidelines

- **Double Precision Requirement**: Always retain strict 64-bit double precision (`float64` / `complex128` / OpenCL `double2`) across all CPU and GPU DSP pipelines, FFTs, filters, and kernels. Never drop to single precision (`float32` / `complex64`).
- **FLAC Compression Level**: Always retain FLAC compression level 5 (equivalent to `compression_level=5/8` or `0.625` in `soundfile` / `libsndfile`).
