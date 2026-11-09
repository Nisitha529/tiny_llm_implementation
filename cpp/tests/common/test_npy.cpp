// Planned checks for common/npy.hpp:
// 1. write_npy then read_npy gives back the same dtype, shape and bytes,
//    for "<f4", "<u2" and "<u4", 1-D and 2-D.
// 2. Reads a file written by numpy (from reference/data/) once it exists.
// 3. A missing or truncated file throws std::runtime_error.
