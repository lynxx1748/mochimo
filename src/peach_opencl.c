/**
 * @file peach_opencl.c
 * @brief OpenCL implementation of Peach Proof-of-Work solver
 * @copyright Adequate Systems LLC, 2025. All Rights Reserved.
 * For license information, please refer to ../LICENSE.md
 *
 * This file provides the OpenCL implementation for AMD GPU mining support.
 * It mirrors the functionality of peach.cu for CUDA devices.
 */

#ifndef MOCHIMO_PEACH_OPENCL_C
#define MOCHIMO_PEACH_OPENCL_C

#include "device.h"
#include "device_opencl.h"
#include "types.h"
#include "peach.h"
#include "trigg.h"
#include "error.h"

/* external support */
#include "extint.h"
#include "extmath.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Log OpenCL errors
 */
static void opencl_log_error_peach(cl_int err, const char *context)
{
    if (err == CL_SUCCESS) return;
    perr("OpenCL ERROR (%d): %s", (int)err, context);
    set_errno(EMCM_OPENCL);
}

/**
 * Check if an OpenCL command queue has completed
 * @returns 1 if ready, 0 if not ready
 */
static int opencl_queue_ready(cl_command_queue queue)
{
    cl_int err;
    cl_event event;
    
    /* Create a marker to check queue status */
    #ifdef CL_VERSION_1_2
    err = clEnqueueMarkerWithWaitList(queue, 0, NULL, &event);
    #else
    err = clEnqueueMarker(queue, &event);
    #endif
    
    if (err != CL_SUCCESS) return 0;
    
    cl_int status;
    err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(status), &status, NULL);
    clReleaseEvent(event);
    
    if (err != CL_SUCCESS) return 0;
    return (status == CL_COMPLETE);
}

/**
 * Try solve for a tokenized haiku as nonce output for Peach proof of work
 * on OpenCL devices.
 * @param ctx Pointer to DEVICE_CTX to perform work with
 * @param bt Pointer to block trailer to solve for
 * @param diff Difficulty to test against entropy of final hash
 * @param btout Pointer to location to place solved block trailer
 * @returns VEOK on solve, VERROR on no solve, or VETIMEOUT if GPU is
 * either stopped or unrecoverable.
 */
int peach_solve_opencl(DEVICE_CTX *ctx, BTRAILER *bt, word8 diff, BTRAILER *btout)
{
    OPENCL_CTX *ocl;
    cl_int err;
    int id, build;
    size_t build_global, build_local;
    
    if (ctx == NULL || ctx->peach == NULL) {
        set_errno(EINVAL);
        return VERROR;
    }
    
    ocl = (OPENCL_CTX *)ctx->peach;
    
    /* Report unusable GPUs */
    if (ctx->status < DEV_NULL) return VETIMEOUT;
    
    /* Check for previous execution errors */
    err = clGetCommandQueueInfo(ocl->queue[0], CL_QUEUE_CONTEXT,
                               0, NULL, NULL);
    if (err != CL_SUCCESS) {
        opencl_log_error_peach(err, "Queue check failed");
        ctx->status = DEV_FAIL;
        return VERROR;
    }
    
    /* Build Peach map */
    if (ctx->status == DEV_INIT) {
        for (build = id = 0; id < 2; id++) {
            /* Check if queue is ready */
            if (!opencl_queue_ready(ocl->queue[id])) continue;
            
            /* Check pre-build state */
            if (ctx->work == 0 && build == 0) {
                /* Ensure secondary queue is ready */
                if (!opencl_queue_ready(ocl->queue[id ^ 1])) break;
                
                /* Clear late solves */
                cl_uchar zero = 0;
                clEnqueueFillBuffer(ocl->queue[0], ocl->d_solve[0], &zero, 1,
                                   0, 32, 0, NULL, NULL);
                clEnqueueFillBuffer(ocl->queue[1], ocl->d_solve[1], &zero, 1,
                                   0, 32, 0, NULL, NULL);
                memset(ocl->h_solve[0], 0, 32);
                memset(ocl->h_solve[1], 0, 32);
                
                /* Update block trailer */
                memcpy(ocl->h_bt[0], bt, sizeof(BTRAILER));
                memcpy(ocl->h_bt[1], bt, sizeof(BTRAILER));
                
                /* Update device phash */
                err = clEnqueueWriteBuffer(ocl->queue[0], ocl->d_phash, CL_FALSE,
                                          0, 32, ((BTRAILER *)ocl->h_bt[0])->phash,
                                          0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    opencl_log_error_peach(err, "Write phash failed");
                    ctx->status = DEV_FAIL;
                    return VERROR;
                }
                
                /* Synchronize before building */
                clFinish(ocl->queue[0]);
                clFinish(ocl->queue[1]);
                
                build = 1;
            }
            
            /* Check build state */
            if (ctx->work > 0 || build) {
                if (ctx->work < PEACHCACHELEN) {
                    /* Calculate work dimensions for build kernel */
                    build_local = ocl->local_work_size;
                    size_t remaining = PEACHCACHELEN - ctx->work;
                    build_global = (remaining < ocl->global_work_size) ? 
                                   remaining : ocl->global_work_size;
                    /* Round up to multiple of local work size */
                    build_global = ((build_global + build_local - 1) / build_local) 
                                   * build_local;
                    
                    /* Set kernel arguments */
                    cl_uint offset = (cl_uint)ctx->work;
                    err = clSetKernelArg(ocl->k_peach_build, 0, sizeof(cl_uint), &offset);
                    err |= clSetKernelArg(ocl->k_peach_build, 1, sizeof(cl_mem), &ocl->d_map);
                    err |= clSetKernelArg(ocl->k_peach_build, 2, sizeof(cl_mem), &ocl->d_phash);
                    
                    if (err != CL_SUCCESS) {
                        opencl_log_error_peach(err, "Set build kernel args failed");
                        ctx->status = DEV_FAIL;
                        return VERROR;
                    }
                    
                    /* Launch build kernel */
                    err = clEnqueueNDRangeKernel(ocl->queue[id], ocl->k_peach_build,
                                                1, NULL, &build_global, &build_local,
                                                0, NULL, NULL);
                    if (err != CL_SUCCESS) {
                        opencl_log_error_peach(err, "Enqueue build kernel failed");
                        ctx->status = DEV_FAIL;
                        return VERROR;
                    }
                    
                    ctx->work += build_global;
                } else {
                    /* Ensure secondary queue is finished */
                    if (!opencl_queue_ready(ocl->queue[id ^ 1])) break;
                    
                    /* Build is complete */
                    ctx->last = time(NULL);
                    ctx->status = DEV_IDLE;
                    ctx->work = 0;
                    break;
                }
            }
        }
    }
    
    /* Switch to WORK mode when conditions are met */
    while (ctx->status == DEV_IDLE) {
        if (get32(bt->tcount) == 0) break;
        if (cmp64(bt->bnum, btout->bnum) == 0) break;
        if (difftime(time(NULL), get32(bt->time0)) >= BRIDGEv3) break;
        ctx->last = time(NULL);
        ctx->status = DEV_WORK;
        ctx->work = 0;
        break;
    }
    
    /* Solve work in block trailer */
    if (ctx->status == DEV_WORK) {
        for (id = 0; id < 2; id++) {
            if (!opencl_queue_ready(ocl->queue[id])) continue;
            
            /* Check trailer for block update */
            if (memcmp(((BTRAILER *)ocl->h_bt[id])->phash, bt->phash, HASHLEN)) {
                ctx->status = DEV_INIT;
                ctx->work = 0;
                break;
            }
            
            /* Switch to IDLE mode when reasonable */
            if (get32(bt->tcount) == 0 || cmp64(bt->bnum, btout->bnum) == 0 ||
                difftime(time(NULL), get32(bt->time0)) >= BRIDGEv3) {
                ctx->status = DEV_IDLE;
                ctx->work = 0;
                break;
            }
            
            /* Check for solves */
            word64 *h_solve = (word64 *)ocl->h_solve[id];
            if (*h_solve) {
                /* Combine solve with nonce and copy to output */
                memcpy(((BTRAILER *)ocl->h_bt[id])->nonce, ocl->h_solve[id], 32);
                memcpy(btout, ocl->h_bt[id], sizeof(BTRAILER));
                
                /* Clear solve */
                cl_uchar zero = 0;
                clEnqueueFillBuffer(ocl->queue[id], ocl->d_solve[id], &zero, 1,
                                   0, 32, 0, NULL, NULL);
                memset(ocl->h_solve[id], 0, 32);
                
                return VEOK;
            }
            
            /* Update block trailer (including half nonce) */
            memcpy(ocl->h_bt[id], bt, 92);
            trigg_generate(((BTRAILER *)ocl->h_bt[id])->nonce);
            
            /* Write trailer to device */
            err = clEnqueueWriteBuffer(ocl->queue[id], ocl->d_bt[id], CL_FALSE,
                                      0, 92 + 16, ocl->h_bt[id],
                                      0, NULL, NULL);
            if (err != CL_SUCCESS) {
                opencl_log_error_peach(err, "Write trailer failed");
                ctx->status = DEV_FAIL;
                return VERROR;
            }
            
            /* Set kernel arguments for solve */
            cl_uchar solve_diff = diff && diff < bt->difficulty[0] ? 
                                  diff : bt->difficulty[0];
            
            err = clSetKernelArg(ocl->k_peach_solve, 0, sizeof(cl_mem), &ocl->d_map);
            err |= clSetKernelArg(ocl->k_peach_solve, 1, sizeof(cl_mem), &ocl->d_bt[id]);
            err |= clSetKernelArg(ocl->k_peach_solve, 2, sizeof(cl_mem), &ocl->d_state[id]);
            err |= clSetKernelArg(ocl->k_peach_solve, 3, sizeof(cl_uchar), &solve_diff);
            err |= clSetKernelArg(ocl->k_peach_solve, 4, sizeof(cl_mem), &ocl->d_solve[id]);
            
            if (err != CL_SUCCESS) {
                opencl_log_error_peach(err, "Set solve kernel args failed");
                ctx->status = DEV_FAIL;
                return VERROR;
            }
            
            /* Launch solve kernel */
            err = clEnqueueNDRangeKernel(ocl->queue[id], ocl->k_peach_solve,
                                        1, NULL, &ocl->global_work_size,
                                        &ocl->local_work_size, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                opencl_log_error_peach(err, "Enqueue solve kernel failed");
                ctx->status = DEV_FAIL;
                return VERROR;
            }
            
            /* Read solve result */
            err = clEnqueueReadBuffer(ocl->queue[id], ocl->d_solve[id], CL_FALSE,
                                     0, 32, ocl->h_solve[id],
                                     0, NULL, NULL);
            if (err != CL_SUCCESS) {
                opencl_log_error_peach(err, "Read solve failed");
                ctx->status = DEV_FAIL;
                return VERROR;
            }
            
            /* Update progress counters */
            ctx->work += ctx->threads;
            double delta = difftime(time(NULL), ctx->last);
            ctx->hps = ctx->work / (delta ? delta : 1);
        }
    }
    
    return VERROR;
}

/**
 * Check Peach proof of work with OpenCL device.
 * @param count Number of block trailers to check
 * @param bt Pointer to block trailer array
 * @param out Pointer to final hash array, if non-null
 * @returns (int) value representing the result of the operation
 * @retval (-1) Error occurred during operation
 * @retval 0 Evaluation successful
 * @retval 1 Evaluation failed
 */
int peach_checkhash_opencl(int count, BTRAILER bt[], void *out)
{
    /* For now, fall back to CPU implementation for hash checking */
    /* OpenCL hash checking can be added later if needed for performance */
    (void)count;
    (void)bt;
    (void)out;
    
    /* Return error to indicate not implemented - caller should use CPU */
    return -1;
}

#endif /* MOCHIMO_PEACH_OPENCL_C */

