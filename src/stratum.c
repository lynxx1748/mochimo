/**
 * @file stratum.c
 * @brief Stratum protocol client for pool mining
 * @copyright Adequate Systems LLC, 2025. All Rights Reserved.
 */

#ifndef MOCHIMO_STRATUM_C
#define MOCHIMO_STRATUM_C

#include "stratum.h"
#include "error.h"
#include "extinet.h"
#include "extio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <poll.h>

/* Simple JSON value extraction (no external dependencies) */
static char *json_get_string(const char *json, const char *key, char *out, size_t outlen)
{
    char search[128];
    const char *start, *end;
    size_t len;

    snprintf(search, sizeof(search), "\"%s\"", key);
    start = strstr(json, search);
    if (!start) return NULL;

    start = strchr(start + strlen(search), ':');
    if (!start) return NULL;
    start++;

    /* Skip whitespace */
    while (*start == ' ' || *start == '\t') start++;

    if (*start == '"') {
        start++;
        end = strchr(start, '"');
        if (!end) return NULL;
        len = end - start;
        if (len >= outlen) len = outlen - 1;
        memcpy(out, start, len);
        out[len] = '\0';
        return out;
    }

    return NULL;
}

static int json_get_int(const char *json, const char *key, int *out)
{
    char search[128];
    const char *start;

    snprintf(search, sizeof(search), "\"%s\"", key);
    start = strstr(json, search);
    if (!start) return -1;

    start = strchr(start + strlen(search), ':');
    if (!start) return -1;
    start++;

    /* Skip whitespace */
    while (*start == ' ' || *start == '\t') start++;

    *out = atoi(start);
    return 0;
}

static int json_get_bool(const char *json, const char *key, int *out)
{
    char search[128];
    const char *start;

    snprintf(search, sizeof(search), "\"%s\"", key);
    start = strstr(json, search);
    if (!start) return -1;

    start = strchr(start + strlen(search), ':');
    if (!start) return -1;
    start++;

    /* Skip whitespace */
    while (*start == ' ' || *start == '\t') start++;

    if (strncmp(start, "true", 4) == 0) {
        *out = 1;
        return 0;
    } else if (strncmp(start, "false", 5) == 0) {
        *out = 0;
        return 0;
    }

    return -1;
}

/* Convert hex string to bytes */
static int hex_to_bytes(const char *hex, word8 *out, size_t outlen)
{
    size_t i, len = strlen(hex);
    if (len > outlen * 2) len = outlen * 2;

    for (i = 0; i < len / 2; i++) {
        unsigned int byte;
        if (sscanf(hex + i * 2, "%2x", &byte) != 1) return -1;
        out[i] = (word8)byte;
    }

    return (int)(len / 2);
}

/* Convert bytes to hex string */
static void bytes_to_hex(const word8 *bytes, size_t len, char *out)
{
    size_t i;
    for (i = 0; i < len; i++) {
        sprintf(out + i * 2, "%02x", bytes[i]);
    }
    out[len * 2] = '\0';
}

/**
 * Initialize stratum context
 */
int stratum_init(STRATUM_CTX *ctx, const char *host, int port,
                 const char *wallet, const char *worker)
{
    memset(ctx, 0, sizeof(STRATUM_CTX));
    ctx->sd = -1;
    ctx->state = STRATUM_DISCONNECTED;
    ctx->msg_id = 1;
    ctx->difficulty = 28;  /* default pool difficulty */

    strncpy(ctx->host, host, sizeof(ctx->host) - 1);
    ctx->port = port;
    strncpy(ctx->wallet, wallet, sizeof(ctx->wallet) - 1);
    strncpy(ctx->worker, worker, sizeof(ctx->worker) - 1);

    return 0;
}

/**
 * Connect to stratum pool
 */
int stratum_connect(STRATUM_CTX *ctx)
{
    struct hostent *he;
    struct sockaddr_in addr;
    int flags;
    char msg[512];

    if (ctx->sd >= 0) {
        close(ctx->sd);
        ctx->sd = -1;
    }

    ctx->state = STRATUM_CONNECTING;
    pdebug("Stratum: Connecting to %s:%d", ctx->host, ctx->port);

    /* Resolve hostname */
    he = gethostbyname(ctx->host);
    if (!he) {
        perr("Stratum: Failed to resolve hostname %s", ctx->host);
        ctx->state = STRATUM_DISCONNECTED;
        return -1;
    }

    /* Create socket */
    ctx->sd = socket(AF_INET, SOCK_STREAM, 0);
    if (ctx->sd < 0) {
        perr("Stratum: Failed to create socket");
        ctx->state = STRATUM_DISCONNECTED;
        return -1;
    }

    /* Connect */
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(ctx->port);
    memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);

    if (connect(ctx->sd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perr("Stratum: Failed to connect to %s:%d", ctx->host, ctx->port);
        close(ctx->sd);
        ctx->sd = -1;
        ctx->state = STRATUM_DISCONNECTED;
        return -1;
    }

    /* Set non-blocking */
    flags = fcntl(ctx->sd, F_GETFL, 0);
    fcntl(ctx->sd, F_SETFL, flags | O_NONBLOCK);

    plog("Stratum: Connected to %s:%d", ctx->host, ctx->port);
    
    /* Lucky Pool may use a custom protocol - try authorize directly */
    ctx->state = STRATUM_AUTHORIZING;

    /* Send mining.authorize with wallet.worker format */
    snprintf(msg, sizeof(msg),
        "{\"id\":%d,\"method\":\"mining.authorize\","
        "\"params\":[\"%s.%s\",\"x\"]}\n",
        ctx->msg_id++, ctx->wallet, ctx->worker);

    plog("Stratum: Sending: %s", msg);
    
    if (send(ctx->sd, msg, strlen(msg), 0) < 0) {
        perr("Stratum: Failed to send authorize");
        stratum_disconnect(ctx);
        return -1;
    }

    plog("Stratum: Sent authorize request, waiting for response...");
    return 0;
}

/**
 * Disconnect from stratum pool
 */
void stratum_disconnect(STRATUM_CTX *ctx)
{
    if (ctx->sd >= 0) {
        close(ctx->sd);
        ctx->sd = -1;
    }
    ctx->state = STRATUM_DISCONNECTED;
    ctx->recv_len = 0;
    pdebug("Stratum: Disconnected");
}

/**
 * Send authorize message
 */
static int stratum_authorize(STRATUM_CTX *ctx)
{
    char msg[512];

    snprintf(msg, sizeof(msg),
        "{\"id\":%d,\"method\":\"mining.authorize\","
        "\"params\":[\"%s.%s\",\"x\"]}\n",
        ctx->msg_id++, ctx->wallet, ctx->worker);

    if (send(ctx->sd, msg, strlen(msg), 0) < 0) {
        perr("Stratum: Failed to send authorize");
        return -1;
    }

    ctx->state = STRATUM_AUTHORIZING;
    pdebug("Stratum: Sent authorize request");
    return 0;
}

/**
 * Parse stratum job notification
 */
static int stratum_parse_job(STRATUM_CTX *ctx, const char *params)
{
    char job_id[64] = {0};
    char phash[128] = {0};
    char bnum[32] = {0};
    char diff[16] = {0};
    char time0[16] = {0};
    char mroot[128] = {0};
    const char *p;
    int i;

    /* Parse params array: ["job_id", "phash", "bnum", "diff", "time0", "mroot", clean] */
    p = strchr(params, '[');
    if (!p) return -1;
    p++;

    /* Job ID */
    if (*p == '"') {
        p++;
        for (i = 0; *p && *p != '"' && i < 63; i++, p++) job_id[i] = *p;
        job_id[i] = '\0';
        if (*p == '"') p++;
    }

    /* Skip to next field */
    p = strchr(p, ',');
    if (!p) return -1;
    p++;
    while (*p == ' ') p++;

    /* Previous hash */
    if (*p == '"') {
        p++;
        for (i = 0; *p && *p != '"' && i < 127; i++, p++) phash[i] = *p;
        phash[i] = '\0';
        if (*p == '"') p++;
    }

    /* Skip to next field */
    p = strchr(p, ',');
    if (!p) return -1;
    p++;
    while (*p == ' ') p++;

    /* Block number */
    if (*p == '"') {
        p++;
        for (i = 0; *p && *p != '"' && i < 31; i++, p++) bnum[i] = *p;
        bnum[i] = '\0';
        if (*p == '"') p++;
    }

    /* Skip to next field */
    p = strchr(p, ',');
    if (!p) return -1;
    p++;
    while (*p == ' ') p++;

    /* Difficulty */
    if (*p == '"') {
        p++;
        for (i = 0; *p && *p != '"' && i < 15; i++, p++) diff[i] = *p;
        diff[i] = '\0';
        if (*p == '"') p++;
    } else {
        for (i = 0; *p && *p != ',' && *p != ']' && i < 15; i++, p++) diff[i] = *p;
        diff[i] = '\0';
    }

    /* Skip to next field */
    p = strchr(p, ',');
    if (!p) return -1;
    p++;
    while (*p == ' ') p++;

    /* Time0 */
    if (*p == '"') {
        p++;
        for (i = 0; *p && *p != '"' && i < 15; i++, p++) time0[i] = *p;
        time0[i] = '\0';
        if (*p == '"') p++;
    } else {
        for (i = 0; *p && *p != ',' && *p != ']' && i < 15; i++, p++) time0[i] = *p;
        time0[i] = '\0';
    }

    /* Skip to next field */
    p = strchr(p, ',');
    if (!p) return -1;
    p++;
    while (*p == ' ') p++;

    /* Merkle root */
    if (*p == '"') {
        p++;
        for (i = 0; *p && *p != '"' && i < 127; i++, p++) mroot[i] = *p;
        mroot[i] = '\0';
    }

    /* Store in pending job */
    strncpy(ctx->pending_job.job_id, job_id, sizeof(ctx->pending_job.job_id) - 1);
    hex_to_bytes(phash, ctx->pending_job.phash, 32);
    hex_to_bytes(bnum, ctx->pending_job.bnum, 8);

    /* Difficulty can be hex or decimal */
    if (strncmp(diff, "0x", 2) == 0) {
        ctx->pending_job.difficulty[0] = (word8)strtol(diff + 2, NULL, 16);
    } else {
        ctx->pending_job.difficulty[0] = (word8)atoi(diff);
    }

    /* Time0 can be hex or decimal */
    uint32_t t0;
    if (strncmp(time0, "0x", 2) == 0) {
        t0 = (uint32_t)strtoul(time0 + 2, NULL, 16);
    } else {
        t0 = (uint32_t)strtoul(time0, NULL, 10);
    }
    ctx->pending_job.time0[0] = t0 & 0xFF;
    ctx->pending_job.time0[1] = (t0 >> 8) & 0xFF;
    ctx->pending_job.time0[2] = (t0 >> 16) & 0xFF;
    ctx->pending_job.time0[3] = (t0 >> 24) & 0xFF;

    hex_to_bytes(mroot, ctx->pending_job.mroot, 32);
    ctx->pending_job.valid = 1;
    ctx->pending_job.job_seq++;

    plog("Stratum: New job %s (diff=%d)", job_id, ctx->pending_job.difficulty[0]);

    return 0;
}

/**
 * Handle stratum message
 */
static int stratum_handle_message(STRATUM_CTX *ctx, const char *msg)
{
    char method[64] = {0};
    int id = 0;
    int result_bool = 0;

    /* Always log received messages for debugging */
    plog("Stratum recv: %.200s%s", msg, strlen(msg) > 200 ? "..." : "");

    /* Check for method (notification) */
    if (json_get_string(msg, "method", method, sizeof(method))) {
        if (strcmp(method, "mining.notify") == 0) {
            /* New job notification */
            const char *params = strstr(msg, "\"params\"");
            if (params) {
                stratum_parse_job(ctx, params);
            }
        } else if (strcmp(method, "mining.set_difficulty") == 0) {
            /* Difficulty update */
            int diff;
            const char *params = strstr(msg, "\"params\"");
            if (params) {
                params = strchr(params, '[');
                if (params) {
                    params++;
                    diff = atoi(params);
                    if (diff > 0) {
                        ctx->difficulty = diff;
                        plog("Stratum: Pool difficulty set to %d", diff);
                    }
                }
            }
        }
        return 0;
    }

    /* Check for response */
    if (json_get_int(msg, "id", &id) == 0) {
        /* This is a response to our request */
        plog("Stratum: Response id=%d, state=%d", id, ctx->state);
        
        if (ctx->state == STRATUM_SUBSCRIBING) {
            /* Subscribe response - check for result (can be array or object) */
            const char *result = strstr(msg, "\"result\"");
            const char *error = strstr(msg, "\"error\":null");
            if (result && (error || strstr(msg, "\"error\": null"))) {
                plog("Stratum: Subscribed successfully");
                ctx->state = STRATUM_AUTHORIZING;
                stratum_authorize(ctx);
            } else if (result) {
                /* Result exists, assume success even without explicit null error */
                plog("Stratum: Subscribed (result found)");
                ctx->state = STRATUM_AUTHORIZING;
                stratum_authorize(ctx);
            } else {
                perr("Stratum: Subscribe failed - no result");
                return -1;
            }
        } else if (ctx->state == STRATUM_AUTHORIZING) {
            /* Authorize response */
            if (json_get_bool(msg, "result", &result_bool) == 0 && result_bool) {
                plog("Stratum: Authorized as %s.%s", ctx->wallet, ctx->worker);
                ctx->state = STRATUM_CONNECTED;
            } else if (strstr(msg, "\"result\":true") || strstr(msg, "\"result\": true")) {
                plog("Stratum: Authorized as %s.%s", ctx->wallet, ctx->worker);
                ctx->state = STRATUM_CONNECTED;
            } else {
                perr("Stratum: Authorization failed");
                return -1;
            }
        } else {
            /* Share submission response */
            if (json_get_bool(msg, "result", &result_bool) == 0) {
                if (result_bool) {
                    ctx->accepted_shares++;
                    plog("Stratum: Share accepted (%lu/%lu)",
                        ctx->accepted_shares, ctx->accepted_shares + ctx->rejected_shares);
                } else {
                    ctx->rejected_shares++;
                    pwarn("Stratum: Share rejected (%lu/%lu)",
                        ctx->rejected_shares, ctx->accepted_shares + ctx->rejected_shares);
                }
            } else if (strstr(msg, "\"result\":true") || strstr(msg, "\"result\": true")) {
                ctx->accepted_shares++;
                plog("Stratum: Share accepted (%lu/%lu)",
                    ctx->accepted_shares, ctx->accepted_shares + ctx->rejected_shares);
            }
        }
    }

    return 0;
}

/**
 * Process incoming stratum messages
 */
int stratum_process(STRATUM_CTX *ctx)
{
    struct pollfd pfd;
    char *newline;
    ssize_t n;

    if (ctx->sd < 0 || ctx->state == STRATUM_DISCONNECTED) {
        return -1;
    }

    /* Poll for incoming data */
    pfd.fd = ctx->sd;
    pfd.events = POLLIN;

    if (poll(&pfd, 1, 100) <= 0) {
        return 0;  /* No data or timeout */
    }

    /* Check for data first (server may send data then close) */
    if (pfd.revents & POLLIN) {
        /* Read available data */
        n = recv(ctx->sd, ctx->recv_buf + ctx->recv_len,
                 sizeof(ctx->recv_buf) - ctx->recv_len - 1, 0);

        if (n <= 0) {
            if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
                perr("Stratum: Connection closed");
                stratum_disconnect(ctx);
                return -1;
            }
            return 0;
        }

        ctx->recv_len += n;
        ctx->recv_buf[ctx->recv_len] = '\0';

        /* Process complete lines */
        while ((newline = strchr(ctx->recv_buf, '\n')) != NULL) {
            *newline = '\0';

            /* Handle the message */
            if (stratum_handle_message(ctx, ctx->recv_buf) < 0) {
                stratum_disconnect(ctx);
                return -1;
            }

            /* Shift buffer */
            memmove(ctx->recv_buf, newline + 1,
                    ctx->recv_len - (newline - ctx->recv_buf) - 1);
            ctx->recv_len -= (newline - ctx->recv_buf) + 1;
            ctx->recv_buf[ctx->recv_len] = '\0';
        }

        /* Prevent buffer overflow */
        if ((size_t)ctx->recv_len >= sizeof(ctx->recv_buf) - 100) {
            pwarn("Stratum: Buffer overflow, clearing");
            ctx->recv_len = 0;
        }
    }

    /* Check for connection errors after reading any available data */
    if (pfd.revents & (POLLERR | POLLHUP)) {
        if (ctx->state != STRATUM_CONNECTED) {
            perr("Stratum: Connection closed by server (revents=0x%x)", pfd.revents);
        }
        stratum_disconnect(ctx);
        return -1;
    }

    return 0;
}

/**
 * Submit share to pool
 */
int stratum_submit(STRATUM_CTX *ctx, const char *job_id,
                   const word8 *nonce, const word8 *hash)
{
    char msg[1024];
    char nonce_hex[65];
    char hash_hex[65];

    if (ctx->sd < 0 || ctx->state != STRATUM_CONNECTED) {
        return -1;
    }

    bytes_to_hex(nonce, 32, nonce_hex);
    bytes_to_hex(hash, 32, hash_hex);

    snprintf(msg, sizeof(msg),
        "{\"id\":%d,\"method\":\"mining.submit\","
        "\"params\":[\"%s.%s\",\"%s\",\"%s\",\"%s\"]}\n",
        ctx->msg_id++, ctx->wallet, ctx->worker, job_id, nonce_hex, hash_hex);

    if (send(ctx->sd, msg, strlen(msg), 0) < 0) {
        perr("Stratum: Failed to send share");
        return -1;
    }

    pdebug("Stratum: Submitted share for job %s", job_id);
    return 0;
}

/**
 * Check if new job is available
 */
int stratum_has_job(STRATUM_CTX *ctx)
{
    return ctx->pending_job.valid &&
           ctx->pending_job.job_seq != ctx->current_job.job_seq;
}

/**
 * Get current job (copies to provided BTRAILER)
 */
int stratum_get_job(STRATUM_CTX *ctx, BTRAILER *bt)
{
    if (!ctx->pending_job.valid) {
        return -1;
    }

    /* Copy pending job to current */
    memcpy(&ctx->current_job, &ctx->pending_job, sizeof(STRATUM_JOB));

    /* Convert to BTRAILER format */
    memset(bt, 0, sizeof(BTRAILER));
    memcpy(bt->phash, ctx->current_job.phash, 32);
    memcpy(bt->bnum, ctx->current_job.bnum, 8);
    memcpy(bt->mroot, ctx->current_job.mroot, 32);
    memcpy(bt->difficulty, ctx->current_job.difficulty, 1);
    memcpy(bt->time0, ctx->current_job.time0, 4);

    return 0;
}

/**
 * Check connection status (any active state)
 */
int stratum_is_connected(STRATUM_CTX *ctx)
{
    return ctx->sd >= 0 && ctx->state >= STRATUM_SUBSCRIBING;
}

#endif /* MOCHIMO_STRATUM_C */

