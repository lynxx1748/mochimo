/**
 * @file device_opencl.h
 * @brief OpenCL device support header
 * @copyright Adequate Systems LLC, 2025. All Rights Reserved.
 * For license information, please refer to ../LICENSE.md
 */

#ifndef MOCHIMO_DEVICE_OPENCL_H
#define MOCHIMO_DEVICE_OPENCL_H

/* Define OpenCL version before including headers */
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/**
 * OpenCL context structure stored in DEVICE_CTX.peach for OpenCL devices
 */
typedef struct OPENCL_CTX_STRUCT {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue[2];       /* dual command queues for async ops */
    cl_program program;
    
    /* Kernels */
    cl_kernel k_srand64;
    cl_kernel k_peach_build;
    cl_kernel k_peach_solve;
    
    /* Device memory buffers */
    cl_mem d_map;                    /* Peach map (1GB) */
    cl_mem d_phash;                  /* Previous hash */
    cl_mem d_bt[2];                  /* Block trailers */
    cl_mem d_state[2];               /* PRNG state */
    cl_mem d_solve[2];               /* Solve output */
    
    /* Host memory (pinned if available) */
    void *h_bt[2];
    void *h_solve[2];
    
    /* Work dimensions */
    size_t global_work_size;
    size_t local_work_size;
} OPENCL_CTX;

#endif /* MOCHIMO_DEVICE_OPENCL_H */

