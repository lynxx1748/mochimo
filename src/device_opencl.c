/**
 * @file device_opencl.c
 * @brief OpenCL device detection and initialization for AMD GPU support
 * @copyright Adequate Systems LLC, 2025. All Rights Reserved.
 * For license information, please refer to ../LICENSE.md
 */

#ifndef MOCHIMO_DEVICE_OPENCL_C
#define MOCHIMO_DEVICE_OPENCL_C

#include "device.h"
#include "device_opencl.h"
#include "error.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Maximum number of platforms/devices to enumerate */
#define MAX_PLATFORMS 8
#define MAX_DEVICES_PER_PLATFORM 16

/* Forward declaration of kernel source loading */
extern const char *peach_opencl_kernel_source;
extern size_t peach_opencl_kernel_source_len;

/**
 * Log OpenCL errors
 */
static void opencl_log_error(cl_int err, const char *context)
{
    const char *errstr;
    switch (err) {
        case CL_SUCCESS: return;
        case CL_DEVICE_NOT_FOUND: errstr = "CL_DEVICE_NOT_FOUND"; break;
        case CL_DEVICE_NOT_AVAILABLE: errstr = "CL_DEVICE_NOT_AVAILABLE"; break;
        case CL_COMPILER_NOT_AVAILABLE: errstr = "CL_COMPILER_NOT_AVAILABLE"; break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: errstr = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
        case CL_OUT_OF_RESOURCES: errstr = "CL_OUT_OF_RESOURCES"; break;
        case CL_OUT_OF_HOST_MEMORY: errstr = "CL_OUT_OF_HOST_MEMORY"; break;
        case CL_BUILD_PROGRAM_FAILURE: errstr = "CL_BUILD_PROGRAM_FAILURE"; break;
        case CL_INVALID_VALUE: errstr = "CL_INVALID_VALUE"; break;
        case CL_INVALID_DEVICE: errstr = "CL_INVALID_DEVICE"; break;
        case CL_INVALID_CONTEXT: errstr = "CL_INVALID_CONTEXT"; break;
        case CL_INVALID_COMMAND_QUEUE: errstr = "CL_INVALID_COMMAND_QUEUE"; break;
        case CL_INVALID_MEM_OBJECT: errstr = "CL_INVALID_MEM_OBJECT"; break;
        case CL_INVALID_PROGRAM: errstr = "CL_INVALID_PROGRAM"; break;
        case CL_INVALID_KERNEL: errstr = "CL_INVALID_KERNEL"; break;
        case CL_INVALID_KERNEL_ARGS: errstr = "CL_INVALID_KERNEL_ARGS"; break;
        case CL_INVALID_WORK_DIMENSION: errstr = "CL_INVALID_WORK_DIMENSION"; break;
        case CL_INVALID_WORK_GROUP_SIZE: errstr = "CL_INVALID_WORK_GROUP_SIZE"; break;
        case CL_INVALID_WORK_ITEM_SIZE: errstr = "CL_INVALID_WORK_ITEM_SIZE"; break;
        default: errstr = "Unknown error"; break;
    }
    perr("OpenCL ERROR (%d) %s: %s", (int)err, errstr, context);
    set_errno(EMCM_OPENCL);
}


/**
 * Initialize OpenCL devices and populate DEVICE_CTX array.
 * @param ctx Pointer to DEVICE_CTX array
 * @param len Maximum number of devices to initialize
 * @returns Number of OpenCL devices found, or -1 on error
 */
int init_opencl_devices(DEVICE_CTX *ctx, int len)
{
    cl_platform_id platforms[MAX_PLATFORMS];
    cl_device_id devices[MAX_DEVICES_PER_PLATFORM];
    cl_uint num_platforms, num_devices;
    cl_int err;
    int total_devices = 0;
    
    /* Get all OpenCL platforms */
    err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        if (num_platforms == 0 || err == CL_INVALID_VALUE) {
            pdebug("No OpenCL platforms found");
            return 0;
        }
        opencl_log_error(err, "clGetPlatformIDs");
        return -1;
    }
    
    pdebug("Found %u OpenCL platform(s)", num_platforms);
    
    /* Iterate through platforms */
    for (cl_uint p = 0; p < num_platforms && total_devices < len; p++) {
        char platform_name[256] = {0};
        char platform_vendor[256] = {0};
        
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 
                         sizeof(platform_name), platform_name, NULL);
        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR,
                         sizeof(platform_vendor), platform_vendor, NULL);
        
        pdebug("Platform %u: %s (%s)", p, platform_name, platform_vendor);
        
        /* Get GPU devices for this platform */
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU,
                            MAX_DEVICES_PER_PLATFORM, devices, &num_devices);
        if (err != CL_SUCCESS) {
            if (err == CL_DEVICE_NOT_FOUND) {
                pdebug("  No GPU devices on this platform");
                continue;
            }
            opencl_log_error(err, "clGetDeviceIDs");
            continue;
        }
        
        pdebug("  Found %u GPU device(s)", num_devices);
        
        /* Initialize each device */
        for (cl_uint d = 0; d < num_devices && total_devices < len; d++) {
            char device_name[256] = {0};
            char device_vendor[256] = {0};
            cl_ulong global_mem = 0;
            cl_uint compute_units = 0;
            size_t max_work_group = 0;
            
            /* Get device properties */
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME,
                           sizeof(device_name), device_name, NULL);
            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR,
                           sizeof(device_vendor), device_vendor, NULL);
            clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE,
                           sizeof(global_mem), &global_mem, NULL);
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS,
                           sizeof(compute_units), &compute_units, NULL);
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                           sizeof(max_work_group), &max_work_group, NULL);
            
            /* Check minimum memory requirement (1GB for Peach map + overhead) */
            if (global_mem < 1200000000ULL) {
                pdebug("  Device %s has insufficient memory (%lu MB), skipping",
                       device_name, (unsigned long)(global_mem / 1024 / 1024));
                continue;
            }
            
            /* Allocate OpenCL context structure */
            OPENCL_CTX *ocl = (OPENCL_CTX *)calloc(1, sizeof(OPENCL_CTX));
            if (ocl == NULL) {
                perr("Failed to allocate OpenCL context");
                continue;
            }
            
            /* Store platform and device */
            ocl->platform = platforms[p];
            ocl->device = devices[d];
            
            /* Initialize DEVICE_CTX */
            ctx[total_devices].id = total_devices;
            ctx[total_devices].type = OPENCL_DEVICE;
            ctx[total_devices].status = DEV_NULL;
            ctx[total_devices].work = 0;
            ctx[total_devices].hps = 0;
            ctx[total_devices].last = time(NULL);
            ctx[total_devices].peach = ocl;
            
            /* Calculate work dimensions optimized for modern GPUs */
            ctx[total_devices].block = (int)max_work_group;
            if (ctx[total_devices].block > 256) {
                ctx[total_devices].block = 256;
            }
            /* RDNA3/RDNA2 work best with very high occupancy - use 256x multiplier */
            /* For 48 CU GPU: 48 * 256 * 256 = 3,145,728 threads */
            ctx[total_devices].grid = compute_units * 256;
            ctx[total_devices].threads = ctx[total_devices].grid * 
                                         ctx[total_devices].block;
            
            pdebug("Work dimensions: %d threads (%d groups x %d)",
                   ctx[total_devices].threads, ctx[total_devices].grid,
                   ctx[total_devices].block);
            
            /* Store work dimensions in OpenCL context */
            ocl->local_work_size = ctx[total_devices].block;
            ocl->global_work_size = ctx[total_devices].threads;
            
            /* Build device info string (truncate device name if too long) */
            device_name[200] = '\0';  /* Ensure room for other info */
            snprintf(ctx[total_devices].info, sizeof(ctx[total_devices].info),
                    "[OpenCL] %.200s (%u CU, %lu MB)",
                    device_name, compute_units,
                    (unsigned long)(global_mem / 1024 / 1024));
            
            pdebug("  Added device: %s", ctx[total_devices].info);
            total_devices++;
        }
    }
    
    return total_devices;
}

/**
 * Load OpenCL kernel source from file or embedded source
 */
static char *load_kernel_source(size_t *len)
{
    FILE *fp;
    char *source = NULL;
    size_t file_size;
    char exe_path[4000];
    char kernel_path[4096];
    
    /* Try to load from file first - check current directory */
    fp = fopen("peach.cl", "rb");
    if (fp == NULL) {
        fp = fopen("src/peach.cl", "rb");
    }
    
    /* Try to find kernel relative to executable location */
#ifndef _WIN32
    if (fp == NULL) {
        ssize_t exe_len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (exe_len > 0) {
            exe_path[exe_len] = '\0';
            /* Find last slash to get directory */
            char *last_slash = strrchr(exe_path, '/');
            if (last_slash != NULL) {
                *last_slash = '\0';
                snprintf(kernel_path, sizeof(kernel_path), "%s/peach.cl", exe_path);
                fp = fopen(kernel_path, "rb");
                if (fp == NULL) {
                    /* Also try ../src/peach.cl relative to bin directory */
                    snprintf(kernel_path, sizeof(kernel_path), "%s/../src/peach.cl", exe_path);
                    fp = fopen(kernel_path, "rb");
                }
            }
        }
    }
#else
    if (fp == NULL) {
        DWORD exe_len = GetModuleFileNameA(NULL, exe_path, sizeof(exe_path));
        if (exe_len > 0) {
            char *last_slash = strrchr(exe_path, '\\');
            if (last_slash != NULL) {
                *last_slash = '\0';
                snprintf(kernel_path, sizeof(kernel_path), "%s\\peach.cl", exe_path);
                fp = fopen(kernel_path, "rb");
                if (fp == NULL) {
                    snprintf(kernel_path, sizeof(kernel_path), "%s\\..\\src\\peach.cl", exe_path);
                    fp = fopen(kernel_path, "rb");
                }
            }
        }
    }
#endif
    
    if (fp == NULL) {
        fp = fopen("/opt/mochimo/peach.cl", "rb");
    }
    
    if (fp != NULL) {
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        
        source = (char *)malloc(file_size + 1);
        if (source != NULL) {
            if (fread(source, 1, file_size, fp) == file_size) {
                source[file_size] = '\0';
                *len = file_size;
            } else {
                free(source);
                source = NULL;
            }
        }
        fclose(fp);
    }
    
    if (source == NULL) {
        perr("Failed to load OpenCL kernel source (peach.cl)");
        return NULL;
    }
    
    return source;
}

/**
 * Initialize OpenCL device context for Peach algorithm
 */
int peach_init_opencl_device(DEVICE_CTX *ctx)
{
    OPENCL_CTX *ocl;
    cl_int err;
    char *kernel_source = NULL;
    size_t kernel_source_len = 0;
    char build_log[16384];
    size_t log_size;
    
    if (ctx == NULL || ctx->peach == NULL) {
        set_errno(EINVAL);
        return VERROR;
    }
    
    ocl = (OPENCL_CTX *)ctx->peach;
    
    /* Create OpenCL context */
    ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        opencl_log_error(err, "clCreateContext");
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    /* Create command queues (dual queues for async operations) */
    #ifdef CL_VERSION_2_0
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 
                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 };
    ocl->queue[0] = clCreateCommandQueueWithProperties(ocl->context, ocl->device, 
                                                        props, &err);
    if (err != CL_SUCCESS) {
        /* Fall back to in-order queue */
        ocl->queue[0] = clCreateCommandQueueWithProperties(ocl->context, 
                                                            ocl->device, NULL, &err);
    }
    ocl->queue[1] = clCreateCommandQueueWithProperties(ocl->context, ocl->device,
                                                        NULL, &err);
    #else
    ocl->queue[0] = clCreateCommandQueue(ocl->context, ocl->device,
                                         CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    if (err != CL_SUCCESS) {
        ocl->queue[0] = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
    }
    ocl->queue[1] = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
    #endif
    
    if (err != CL_SUCCESS) {
        opencl_log_error(err, "clCreateCommandQueue");
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    /* Load and compile kernel source */
    kernel_source = load_kernel_source(&kernel_source_len);
    if (kernel_source == NULL) {
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    ocl->program = clCreateProgramWithSource(ocl->context, 1,
                                             (const char **)&kernel_source,
                                             &kernel_source_len, &err);
    free(kernel_source);
    
    if (err != CL_SUCCESS) {
        opencl_log_error(err, "clCreateProgramWithSource");
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    /* Build program with optimizations */
    err = clBuildProgram(ocl->program, 1, &ocl->device,
                        "-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math",
                        NULL, NULL);
    if (err != CL_SUCCESS) {
        opencl_log_error(err, "clBuildProgram");
        
        /* Get build log */
        clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG,
                             sizeof(build_log), build_log, &log_size);
        if (log_size > 0) {
            perr("OpenCL build log:\n%s", build_log);
        }
        
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    /* Create kernels */
    ocl->k_srand64 = clCreateKernel(ocl->program, "kcl_srand64", &err);
    if (err != CL_SUCCESS) {
        opencl_log_error(err, "clCreateKernel(kcl_srand64)");
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    ocl->k_peach_build = clCreateKernel(ocl->program, "kcl_peach_build", &err);
    if (err != CL_SUCCESS) {
        opencl_log_error(err, "clCreateKernel(kcl_peach_build)");
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    ocl->k_peach_solve = clCreateKernel(ocl->program, "kcl_peach_solve", &err);
    if (err != CL_SUCCESS) {
        opencl_log_error(err, "clCreateKernel(kcl_peach_solve)");
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    /* Allocate device memory */
    /* Peach map: 1GB */
    ocl->d_map = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                               1073741824ULL, NULL, &err);
    if (err != CL_SUCCESS) {
        opencl_log_error(err, "clCreateBuffer(d_map)");
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    /* Previous hash: 32 bytes */
    ocl->d_phash = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                 32, NULL, &err);
    if (err != CL_SUCCESS) {
        opencl_log_error(err, "clCreateBuffer(d_phash)");
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    /* Block trailers and state for dual command queues */
    size_t bt_size = 160; /* sizeof(BTRAILER) */
    size_t state_size = sizeof(cl_ulong) * ocl->global_work_size;
    
    for (int i = 0; i < 2; i++) {
        ocl->d_bt[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                     bt_size, NULL, &err);
        if (err != CL_SUCCESS) {
            opencl_log_error(err, "clCreateBuffer(d_bt)");
            ctx->status = DEV_FAIL;
            return VERROR;
        }
        
        ocl->d_state[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                        state_size, NULL, &err);
        if (err != CL_SUCCESS) {
            opencl_log_error(err, "clCreateBuffer(d_state)");
            ctx->status = DEV_FAIL;
            return VERROR;
        }
        
        ocl->d_solve[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                        32, NULL, &err);
        if (err != CL_SUCCESS) {
            opencl_log_error(err, "clCreateBuffer(d_solve)");
            ctx->status = DEV_FAIL;
            return VERROR;
        }
        
        /* Allocate host memory */
        ocl->h_bt[i] = calloc(1, bt_size);
        ocl->h_solve[i] = calloc(1, 32);
        if (ocl->h_bt[i] == NULL || ocl->h_solve[i] == NULL) {
            perr("Failed to allocate host memory");
            ctx->status = DEV_FAIL;
            return VERROR;
        }
        
        /* Clear device memory */
        cl_uchar zero = 0;
        clEnqueueFillBuffer(ocl->queue[i], ocl->d_bt[i], &zero, 1,
                           0, bt_size, 0, NULL, NULL);
        clEnqueueFillBuffer(ocl->queue[i], ocl->d_solve[i], &zero, 1,
                           0, 32, 0, NULL, NULL);
        
        /* Initialize PRNG state */
        cl_ulong seed = (cl_ulong)time(NULL) ^ ((cl_ulong)ctx->id << 32) ^ i;
        err = clSetKernelArg(ocl->k_srand64, 0, sizeof(cl_mem), &ocl->d_state[i]);
        err |= clSetKernelArg(ocl->k_srand64, 1, sizeof(cl_ulong), &seed);
        if (err != CL_SUCCESS) {
            opencl_log_error(err, "clSetKernelArg(k_srand64)");
            ctx->status = DEV_FAIL;
            return VERROR;
        }
        
        err = clEnqueueNDRangeKernel(ocl->queue[i], ocl->k_srand64, 1, NULL,
                                    &ocl->global_work_size, &ocl->local_work_size,
                                    0, NULL, NULL);
        if (err != CL_SUCCESS) {
            opencl_log_error(err, "clEnqueueNDRangeKernel(k_srand64)");
            ctx->status = DEV_FAIL;
            return VERROR;
        }
    }
    
    /* Wait for initialization to complete */
    clFinish(ocl->queue[0]);
    clFinish(ocl->queue[1]);
    
    ctx->status = DEV_INIT;
    pdebug("OpenCL device %d initialized successfully", ctx->id);
    
    return VEOK;
}

/**
 * Free OpenCL device resources
 */
void peach_free_opencl_device(DEVICE_CTX *ctx)
{
    OPENCL_CTX *ocl;
    
    if (ctx == NULL || ctx->peach == NULL) return;
    
    ocl = (OPENCL_CTX *)ctx->peach;
    
    /* Release kernels */
    if (ocl->k_srand64) clReleaseKernel(ocl->k_srand64);
    if (ocl->k_peach_build) clReleaseKernel(ocl->k_peach_build);
    if (ocl->k_peach_solve) clReleaseKernel(ocl->k_peach_solve);
    
    /* Release memory objects */
    if (ocl->d_map) clReleaseMemObject(ocl->d_map);
    if (ocl->d_phash) clReleaseMemObject(ocl->d_phash);
    
    for (int i = 0; i < 2; i++) {
        if (ocl->d_bt[i]) clReleaseMemObject(ocl->d_bt[i]);
        if (ocl->d_state[i]) clReleaseMemObject(ocl->d_state[i]);
        if (ocl->d_solve[i]) clReleaseMemObject(ocl->d_solve[i]);
        if (ocl->h_bt[i]) free(ocl->h_bt[i]);
        if (ocl->h_solve[i]) free(ocl->h_solve[i]);
    }
    
    /* Release program and queues */
    if (ocl->program) clReleaseProgram(ocl->program);
    if (ocl->queue[0]) clReleaseCommandQueue(ocl->queue[0]);
    if (ocl->queue[1]) clReleaseCommandQueue(ocl->queue[1]);
    if (ocl->context) clReleaseContext(ocl->context);
    
    free(ocl);
    ctx->peach = NULL;
    ctx->status = DEV_NULL;
}

#endif /* MOCHIMO_DEVICE_OPENCL_C */

