/**
 * @file stratum.h
 * @brief Stratum protocol client for pool mining
 * @copyright Adequate Systems LLC, 2025. All Rights Reserved.
 */

#ifndef MOCHIMO_STRATUM_H
#define MOCHIMO_STRATUM_H

#include "types.h"
#include <stdint.h>

/* Stratum connection states */
#define STRATUM_DISCONNECTED  0
#define STRATUM_CONNECTING    1
#define STRATUM_SUBSCRIBING   2
#define STRATUM_AUTHORIZING   3
#define STRATUM_CONNECTED     4

/* Stratum buffer sizes */
#define STRATUM_BUF_SIZE      4096
#define STRATUM_JOB_ID_LEN    64

/* Stratum job structure */
typedef struct {
    char job_id[STRATUM_JOB_ID_LEN];
    word8 phash[32];           /* previous block hash */
    word8 bnum[8];             /* block number */
    word8 difficulty[4];       /* difficulty */
    word8 time0[4];            /* block time */
    word8 mroot[32];           /* merkle root */
    word8 maddr[ADDR_TAG_LEN]; /* mining address from pool */
    int valid;                 /* job validity flag */
    uint64_t job_seq;          /* job sequence number */
} STRATUM_JOB;

/* Stratum context structure */
typedef struct {
    int sd;                    /* socket descriptor */
    int state;                 /* connection state */
    char host[256];            /* pool hostname */
    int port;                  /* pool port */
    char wallet[64];           /* wallet address */
    char worker[64];           /* worker name */
    char session_id[64];       /* session ID from pool */
    int msg_id;                /* message ID counter */
    char recv_buf[STRATUM_BUF_SIZE];
    int recv_len;
    STRATUM_JOB current_job;
    STRATUM_JOB pending_job;
    uint64_t accepted_shares;
    uint64_t rejected_shares;
    int difficulty;            /* pool difficulty */
} STRATUM_CTX;

/* Initialize stratum context */
int stratum_init(STRATUM_CTX *ctx, const char *host, int port,
                 const char *wallet, const char *worker);

/* Connect to stratum pool */
int stratum_connect(STRATUM_CTX *ctx);

/* Disconnect from stratum pool */
void stratum_disconnect(STRATUM_CTX *ctx);

/* Process incoming stratum messages */
int stratum_process(STRATUM_CTX *ctx);

/* Submit share to pool */
int stratum_submit(STRATUM_CTX *ctx, const char *job_id,
                   const word8 *nonce, const word8 *hash);

/* Check if new job is available */
int stratum_has_job(STRATUM_CTX *ctx);

/* Get current job (copies to provided BTRAILER) */
int stratum_get_job(STRATUM_CTX *ctx, BTRAILER *bt);

/* Check connection status */
int stratum_is_connected(STRATUM_CTX *ctx);

#endif /* MOCHIMO_STRATUM_H */

