

#include "types.h"

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

/* CUDA device functions (NVIDIA) */
int init_cuda_devices(DEVICE_CTX *ctx, int len);

/* OpenCL device functions (AMD, Intel, and cross-platform) */
int init_opencl_devices(DEVICE_CTX *ctx, int len);
int peach_init_opencl_device(DEVICE_CTX *ctx);
int peach_solve_opencl(DEVICE_CTX *ctx, BTRAILER *bt, word8 diff, BTRAILER *btout);
int peach_checkhash_opencl(int count, BTRAILER bt[], void *out);
void peach_free_opencl_device(DEVICE_CTX *ctx);

#ifdef __cplusplus
}  /* end extern "C" */
#endif
