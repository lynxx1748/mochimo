/**
 * @file peach.cl
 * @brief OpenCL kernels for Peach Proof-of-Work algorithm
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * For license information, please refer to ../LICENSE.md
 *
 * Ported from CUDA implementation (peach.cu) for AMD GPU support.
 */

/* -------------------- Type Definitions -------------------- */

typedef unsigned char word8;
typedef unsigned short word16;
typedef unsigned int word32;
typedef unsigned long word64;
typedef int int32;

/* -------------------- Constants -------------------- */

#define HASHLEN          32
#define SHA256LEN        32
#define PEACHROUNDS      8
#define PEACHGENLEN      36
#define PEACHJUMPLEN     1060
#define PEACHTILELEN     1024
#define PEACHTILELEN32   256
#define PEACHTILELEN64   128
#define PEACHCACHELEN    1048576
#define PEACHCACHELEN_M1 1048575

#define WORD64_C(x) (x##UL)
#define WORD32_C(x) (x##U)

/* -------------------- Utility Functions -------------------- */

inline word32 bswap32(word32 x) {
    return ((x >> 24) & 0xff) | ((x >> 8) & 0xff00) |
           ((x << 8) & 0xff0000) | ((x << 24) & 0xff000000);
}

inline word64 rotr64(word64 x, int n) {
    return (x >> n) | (x << (64 - n));
}

inline word32 rotr32(word32 x, int n) {
    return (x >> n) | (x << (32 - n));
}

/* Count leading zeros - OpenCL equivalent of CUDA's __clz */
inline int clz32(word32 x) {
    return clz(x);
}

/* Byte permutation - OpenCL equivalent of CUDA's __byte_perm */
inline word32 byte_perm(word32 x, word32 y, word32 s) {
    word8 bytes[8];
    bytes[0] = x & 0xff;
    bytes[1] = (x >> 8) & 0xff;
    bytes[2] = (x >> 16) & 0xff;
    bytes[3] = (x >> 24) & 0xff;
    bytes[4] = y & 0xff;
    bytes[5] = (y >> 8) & 0xff;
    bytes[6] = (y >> 16) & 0xff;
    bytes[7] = (y >> 24) & 0xff;
    
    word32 result = 0;
    result |= bytes[s & 0x7];
    result |= bytes[(s >> 4) & 0x7] << 8;
    result |= bytes[(s >> 8) & 0x7] << 16;
    result |= bytes[(s >> 12) & 0x7] << 24;
    return result;
}

/* -------------------- Haiku Word Tables -------------------- */

__constant word64 Z_ING[32] = {
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 23, 24, 31, 32, 33, 34
};

__constant word64 Z_NS[64] = {
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 145, 149, 154,
    155, 156, 157, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187,
    188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201,
    202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 241,
    244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};

__constant word64 Z_MASS[32] = {
    214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235,
    236, 237, 238, 239, 240, 242, 214, 215, 216, 219
};

__constant word64 Z_PREP[8] = {
    12, 13, 14, 15, 16, 17, 12, 13
};

__constant word64 Z_ADJ[64] = {
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
    105, 107, 108, 109, 110, 112, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128
};

/* -------------------- Blake2b Constants -------------------- */

__constant word8 blake2b_sigma[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
};

__constant word64 blake2b_iv[8] = {
    WORD64_C(0x6a09e667f3bcc908), WORD64_C(0xbb67ae8584caa73b),
    WORD64_C(0x3c6ef372fe94f82b), WORD64_C(0xa54ff53a5f1d36f1),
    WORD64_C(0x510e527fade682d1), WORD64_C(0x9b05688c2b3e6c1f),
    WORD64_C(0x1f83d9abfb41bd6b), WORD64_C(0x5be0cd19137e2179)
};

/* -------------------- SHA256 Constants -------------------- */

__constant word32 sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* -------------------- SHA1 Constants -------------------- */

__constant word32 sha1_k[4] = {
    0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6
};

/* -------------------- MD2 S-box -------------------- */

__constant word8 md2_s[256] = {
    41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6,
    19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188,
    76, 130, 202, 30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24,
    138, 23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251,
    245, 142, 187, 47, 238, 122, 169, 104, 121, 145, 21, 178, 7, 63,
    148, 194, 16, 137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50,
    39, 53, 62, 204, 231, 191, 247, 151, 3, 255, 25, 48, 179, 72, 165,
    181, 209, 215, 94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210,
    150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241, 69, 157,
    112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2, 27,
    96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15,
    85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197,
    234, 38, 44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65,
    129, 77, 82, 106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123,
    8, 12, 189, 177, 74, 120, 136, 149, 139, 227, 99, 232, 109, 233,
    203, 213, 254, 59, 0, 29, 57, 242, 239, 183, 14, 102, 88, 208, 228,
    166, 119, 114, 248, 235, 117, 75, 10, 49, 68, 80, 180, 143, 237,
    31, 26, 219, 153, 141, 51, 159, 17, 131, 20
};

/* -------------------- Keccak Constants -------------------- */

__constant word64 keccakf_rndc[24] = {
    WORD64_C(0x0000000000000001), WORD64_C(0x0000000000008082),
    WORD64_C(0x800000000000808a), WORD64_C(0x8000000080008000),
    WORD64_C(0x000000000000808b), WORD64_C(0x0000000080000001),
    WORD64_C(0x8000000080008081), WORD64_C(0x8000000000008009),
    WORD64_C(0x000000000000008a), WORD64_C(0x0000000000000088),
    WORD64_C(0x0000000080008009), WORD64_C(0x000000008000000a),
    WORD64_C(0x000000008000808b), WORD64_C(0x800000000000008b),
    WORD64_C(0x8000000000008089), WORD64_C(0x8000000000008003),
    WORD64_C(0x8000000000008002), WORD64_C(0x8000000000000080),
    WORD64_C(0x000000000000800a), WORD64_C(0x800000008000000a),
    WORD64_C(0x8000000080008081), WORD64_C(0x8000000000008080),
    WORD64_C(0x0000000080000001), WORD64_C(0x8000000080008008)
};

/* -------------------- Blake2b Implementation -------------------- */

#define BLAKE2B_G(r, i, a, b, c, d, m) do { \
    a = a + b + m[blake2b_sigma[r][2*i+0]]; \
    d = rotr64(d ^ a, 32); \
    c = c + d; \
    b = rotr64(b ^ c, 24); \
    a = a + b + m[blake2b_sigma[r][2*i+1]]; \
    d = rotr64(d ^ a, 16); \
    c = c + d; \
    b = rotr64(b ^ c, 63); \
} while(0)

#define BLAKE2B_ROUND(r, v, m) do { \
    BLAKE2B_G(r, 0, v[0], v[4], v[8],  v[12], m); \
    BLAKE2B_G(r, 1, v[1], v[5], v[9],  v[13], m); \
    BLAKE2B_G(r, 2, v[2], v[6], v[10], v[14], m); \
    BLAKE2B_G(r, 3, v[3], v[7], v[11], v[15], m); \
    BLAKE2B_G(r, 4, v[0], v[5], v[10], v[15], m); \
    BLAKE2B_G(r, 5, v[1], v[6], v[11], v[12], m); \
    BLAKE2B_G(r, 6, v[2], v[7], v[8],  v[13], m); \
    BLAKE2B_G(r, 7, v[3], v[4], v[9],  v[14], m); \
} while(0)

void cl_peach_blake2b(__private const word64 *in, size_t inlen, int keylen,
    __private word64 *out)
{
    word64 v[16];
    word64 state[8];
    word64 final_block[16];
    word64 t[2] = { 128, 0 };

    /* Fast-forward state to known keylen states */
    if (keylen == 64) {
        state[0] = WORD64_C(0x00B8AA23C261EF69);
        state[1] = WORD64_C(0xD38AE6ABCA237B9E);
        state[2] = WORD64_C(0x67FB881E5EE89069);
        state[3] = WORD64_C(0x3E5B8BD06B58D002);
        state[4] = WORD64_C(0x252D3F68395AAE91);
        state[5] = WORD64_C(0xD25465E23C6C1B27);
        state[6] = WORD64_C(0x852B4CC2E13303B5);
        state[7] = WORD64_C(0x3F38B9FF245BE7C1);
    } else {
        state[0] = WORD64_C(0x63320ACE264383EB);
        state[1] = WORD64_C(0x012AF5FD045A2737);
        state[2] = WORD64_C(0xF4F49C55E6BE39DF);
        state[3] = WORD64_C(0x791C5BC8AFFB11A7);
        state[4] = WORD64_C(0xC9BCACC002C0EA21);
        state[5] = WORD64_C(0x8295B8ABE2FDEDD6);
        state[6] = WORD64_C(0xB711490E5F9F41C8);
        state[7] = WORD64_C(0x3F8E4D1D9EBEAF1A);
    }

    /* blake2b_update */
    for (; inlen > 128; inlen -= 128, in = &in[16]) {
        t[0] += 128;
        /* compress_init */
        for (int i = 0; i < 8; i++) v[i] = state[i];
        v[8] = blake2b_iv[0];
        v[9] = blake2b_iv[1];
        v[10] = blake2b_iv[2];
        v[11] = blake2b_iv[3];
        v[12] = blake2b_iv[4] ^ t[0];
        v[13] = blake2b_iv[5] ^ t[1];
        v[14] = blake2b_iv[6];
        v[15] = blake2b_iv[7];
        /* rounds */
        BLAKE2B_ROUND(0, v, in);
        BLAKE2B_ROUND(1, v, in);
        BLAKE2B_ROUND(2, v, in);
        BLAKE2B_ROUND(3, v, in);
        BLAKE2B_ROUND(4, v, in);
        BLAKE2B_ROUND(5, v, in);
        BLAKE2B_ROUND(6, v, in);
        BLAKE2B_ROUND(7, v, in);
        BLAKE2B_ROUND(8, v, in);
        BLAKE2B_ROUND(9, v, in);
        BLAKE2B_ROUND(10, v, in);
        BLAKE2B_ROUND(11, v, in);
        /* set state */
        for (int i = 0; i < 8; i++) state[i] ^= v[i] ^ v[i + 8];
    }

    /* blake2b_final - remaining datalen will always be 36 */
    final_block[0] = in[0];
    final_block[1] = in[1];
    final_block[2] = in[2];
    final_block[3] = in[3];
    final_block[4] = (word64)((__private word32 *)in)[8];
    for (int i = 5; i < 16; i++) final_block[i] = 0;

    t[0] += 36;
    /* compress_init with final flag */
    for (int i = 0; i < 8; i++) v[i] = state[i];
    v[8] = blake2b_iv[0];
    v[9] = blake2b_iv[1];
    v[10] = blake2b_iv[2];
    v[11] = blake2b_iv[3];
    v[12] = blake2b_iv[4] ^ t[0];
    v[13] = blake2b_iv[5] ^ t[1];
    v[14] = ~blake2b_iv[6];  /* final flag */
    v[15] = blake2b_iv[7];
    /* rounds */
    BLAKE2B_ROUND(0, v, final_block);
    BLAKE2B_ROUND(1, v, final_block);
    BLAKE2B_ROUND(2, v, final_block);
    BLAKE2B_ROUND(3, v, final_block);
    BLAKE2B_ROUND(4, v, final_block);
    BLAKE2B_ROUND(5, v, final_block);
    BLAKE2B_ROUND(6, v, final_block);
    BLAKE2B_ROUND(7, v, final_block);
    BLAKE2B_ROUND(8, v, final_block);
    BLAKE2B_ROUND(9, v, final_block);
    BLAKE2B_ROUND(10, v, final_block);
    BLAKE2B_ROUND(11, v, final_block);

    /* blake2b_output - 256 bits */
    out[0] = state[0] ^ v[0] ^ v[8];
    out[1] = state[1] ^ v[1] ^ v[9];
    out[2] = state[2] ^ v[2] ^ v[10];
    out[3] = state[3] ^ v[3] ^ v[11];
}

/* -------------------- MD2 Implementation -------------------- */

void cl_peach_md2(__private const word64 *in, size_t inlen, __private word64 *out)
{
    word8 state[48] = { 0 };
    word8 checksum[16] = { 0 };
    word8 pad;
    int i, j, t;

    /* prepare padding */
    pad = 16 - (inlen & 0xf);

    /* md2_update */
    for (; inlen >= 16; inlen -= 16, in = &in[2]) {
        __private word8 *inp = (__private word8 *)in;
        /* transform init */
        for (i = 0; i < 16; i++) {
            state[16 + i] = inp[i];
            state[32 + i] = state[16 + i] ^ state[i];
        }
        /* transform checksum */
        t = checksum[15];
        for (i = 0; i < 16; i++) {
            checksum[i] ^= md2_s[inp[i] ^ t];
            t = checksum[i];
        }
        /* transform state */
        t = 0;
        for (i = 0; i < 18; i++) {
            for (j = 0; j < 48; j++) {
                state[j] ^= md2_s[t];
                t = state[j];
            }
            t = (t + i) & 0xff;
        }
    }

    /* md2_final - only 4 bytes left, so 12 remaining bytes are pad */
    __private word8 *inp = (__private word8 *)in;
    for (i = 0; i < 4; i++) state[16 + i] = inp[i];
    for (i = 4; i < 16; i++) state[16 + i] = pad;
    for (i = 0; i < 16; i++) state[32 + i] = state[16 + i] ^ state[i];
    
    /* final transform part1 - checksum */
    t = checksum[15];
    for (i = 0; i < 16; i++) {
        checksum[i] ^= md2_s[state[16 + i] ^ t];
        t = checksum[i];
    }
    /* final transform part1 - state */
    t = 0;
    for (i = 0; i < 18; i++) {
        for (j = 0; j < 48; j++) {
            state[j] ^= md2_s[t];
            t = state[j];
        }
        t = (t + i) & 0xff;
    }
    
    /* final transform part2 */
    for (i = 0; i < 16; i++) {
        state[16 + i] = checksum[i];
        state[32 + i] = checksum[i] ^ state[i];
    }
    t = 0;
    for (i = 0; i < 18; i++) {
        for (j = 0; j < 48; j++) {
            state[j] ^= md2_s[t];
            t = state[j];
        }
        t = (t + i) & 0xff;
    }

    /* MD2 hash = 128 bits, zero fill remaining */
    __private word64 *st64 = (__private word64 *)state;
    out[0] = st64[0];
    out[1] = st64[1];
    out[2] = 0;
    out[3] = 0;
}

/* -------------------- MD5 Implementation -------------------- */

#define MD5_F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5_G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5_H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5_I(x, y, z) ((y) ^ ((x) | (~z)))

#define MD5_FF(a, b, c, d, x, s, ac) { \
    (a) += MD5_F((b), (c), (d)) + (x) + (word32)(ac); \
    (a) = rotate((a), (word32)(s)); \
    (a) += (b); \
}

#define MD5_GG(a, b, c, d, x, s, ac) { \
    (a) += MD5_G((b), (c), (d)) + (x) + (word32)(ac); \
    (a) = rotate((a), (word32)(s)); \
    (a) += (b); \
}

#define MD5_HH(a, b, c, d, x, s, ac) { \
    (a) += MD5_H((b), (c), (d)) + (x) + (word32)(ac); \
    (a) = rotate((a), (word32)(s)); \
    (a) += (b); \
}

#define MD5_II(a, b, c, d, x, s, ac) { \
    (a) += MD5_I((b), (c), (d)) + (x) + (word32)(ac); \
    (a) = rotate((a), (word32)(s)); \
    (a) += (b); \
}

void md5_transform(__private word32 *state, __private const word32 *x)
{
    word32 a = state[0], b = state[1], c = state[2], d = state[3];

    /* Round 1 */
    MD5_FF(a, b, c, d, x[ 0],  7, 0xd76aa478);
    MD5_FF(d, a, b, c, x[ 1], 12, 0xe8c7b756);
    MD5_FF(c, d, a, b, x[ 2], 17, 0x242070db);
    MD5_FF(b, c, d, a, x[ 3], 22, 0xc1bdceee);
    MD5_FF(a, b, c, d, x[ 4],  7, 0xf57c0faf);
    MD5_FF(d, a, b, c, x[ 5], 12, 0x4787c62a);
    MD5_FF(c, d, a, b, x[ 6], 17, 0xa8304613);
    MD5_FF(b, c, d, a, x[ 7], 22, 0xfd469501);
    MD5_FF(a, b, c, d, x[ 8],  7, 0x698098d8);
    MD5_FF(d, a, b, c, x[ 9], 12, 0x8b44f7af);
    MD5_FF(c, d, a, b, x[10], 17, 0xffff5bb1);
    MD5_FF(b, c, d, a, x[11], 22, 0x895cd7be);
    MD5_FF(a, b, c, d, x[12],  7, 0x6b901122);
    MD5_FF(d, a, b, c, x[13], 12, 0xfd987193);
    MD5_FF(c, d, a, b, x[14], 17, 0xa679438e);
    MD5_FF(b, c, d, a, x[15], 22, 0x49b40821);

    /* Round 2 */
    MD5_GG(a, b, c, d, x[ 1],  5, 0xf61e2562);
    MD5_GG(d, a, b, c, x[ 6],  9, 0xc040b340);
    MD5_GG(c, d, a, b, x[11], 14, 0x265e5a51);
    MD5_GG(b, c, d, a, x[ 0], 20, 0xe9b6c7aa);
    MD5_GG(a, b, c, d, x[ 5],  5, 0xd62f105d);
    MD5_GG(d, a, b, c, x[10],  9, 0x02441453);
    MD5_GG(c, d, a, b, x[15], 14, 0xd8a1e681);
    MD5_GG(b, c, d, a, x[ 4], 20, 0xe7d3fbc8);
    MD5_GG(a, b, c, d, x[ 9],  5, 0x21e1cde6);
    MD5_GG(d, a, b, c, x[14],  9, 0xc33707d6);
    MD5_GG(c, d, a, b, x[ 3], 14, 0xf4d50d87);
    MD5_GG(b, c, d, a, x[ 8], 20, 0x455a14ed);
    MD5_GG(a, b, c, d, x[13],  5, 0xa9e3e905);
    MD5_GG(d, a, b, c, x[ 2],  9, 0xfcefa3f8);
    MD5_GG(c, d, a, b, x[ 7], 14, 0x676f02d9);
    MD5_GG(b, c, d, a, x[12], 20, 0x8d2a4c8a);

    /* Round 3 */
    MD5_HH(a, b, c, d, x[ 5],  4, 0xfffa3942);
    MD5_HH(d, a, b, c, x[ 8], 11, 0x8771f681);
    MD5_HH(c, d, a, b, x[11], 16, 0x6d9d6122);
    MD5_HH(b, c, d, a, x[14], 23, 0xfde5380c);
    MD5_HH(a, b, c, d, x[ 1],  4, 0xa4beea44);
    MD5_HH(d, a, b, c, x[ 4], 11, 0x4bdecfa9);
    MD5_HH(c, d, a, b, x[ 7], 16, 0xf6bb4b60);
    MD5_HH(b, c, d, a, x[10], 23, 0xbebfbc70);
    MD5_HH(a, b, c, d, x[13],  4, 0x289b7ec6);
    MD5_HH(d, a, b, c, x[ 0], 11, 0xeaa127fa);
    MD5_HH(c, d, a, b, x[ 3], 16, 0xd4ef3085);
    MD5_HH(b, c, d, a, x[ 6], 23, 0x04881d05);
    MD5_HH(a, b, c, d, x[ 9],  4, 0xd9d4d039);
    MD5_HH(d, a, b, c, x[12], 11, 0xe6db99e5);
    MD5_HH(c, d, a, b, x[15], 16, 0x1fa27cf8);
    MD5_HH(b, c, d, a, x[ 2], 23, 0xc4ac5665);

    /* Round 4 */
    MD5_II(a, b, c, d, x[ 0],  6, 0xf4292244);
    MD5_II(d, a, b, c, x[ 7], 10, 0x432aff97);
    MD5_II(c, d, a, b, x[14], 15, 0xab9423a7);
    MD5_II(b, c, d, a, x[ 5], 21, 0xfc93a039);
    MD5_II(a, b, c, d, x[12],  6, 0x655b59c3);
    MD5_II(d, a, b, c, x[ 3], 10, 0x8f0ccc92);
    MD5_II(c, d, a, b, x[10], 15, 0xffeff47d);
    MD5_II(b, c, d, a, x[ 1], 21, 0x85845dd1);
    MD5_II(a, b, c, d, x[ 8],  6, 0x6fa87e4f);
    MD5_II(d, a, b, c, x[15], 10, 0xfe2ce6e0);
    MD5_II(c, d, a, b, x[ 6], 15, 0xa3014314);
    MD5_II(b, c, d, a, x[13], 21, 0x4e0811a1);
    MD5_II(a, b, c, d, x[ 4],  6, 0xf7537e82);
    MD5_II(d, a, b, c, x[11], 10, 0xbd3af235);
    MD5_II(c, d, a, b, x[ 2], 15, 0x2ad7d2bb);
    MD5_II(b, c, d, a, x[ 9], 21, 0xeb86d391);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

void cl_peach_md5(__private const word32 *in, size_t inlen, __private word64 *out)
{
    word32 final_block[16];
    word32 state[4] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476 };

    /* prepare bitlen in final data */
    final_block[14] = inlen << 3;
    final_block[15] = 0;

    /* md5_update */
    for (; inlen >= 64; inlen -= 64, in = &in[16]) {
        md5_transform(state, in);
    }

    /* md5_final - remaining datalen will always be 36 */
    for (int i = 0; i < 9; i++) final_block[i] = in[i];
    final_block[9] = 0x80;
    for (int i = 10; i < 14; i++) final_block[i] = 0;

    md5_transform(state, final_block);

    /* MD5 hash = 128 bits, zero fill remaining */
    out[0] = ((__private word64 *)state)[0];
    out[1] = ((__private word64 *)state)[1];
    out[2] = 0;
    out[3] = 0;
}

/* -------------------- SHA1 Implementation -------------------- */

#define SHA1_ROTL(x, n) rotate((x), (word32)(n))

void sha1_transform(__private word32 *state, __private const word32 *block)
{
    word32 w[80];
    word32 a, b, c, d, e, temp;
    int i;

    /* Initialize working variables */
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];

    /* Prepare message schedule */
    for (i = 0; i < 16; i++) {
        w[i] = bswap32(block[i]);
    }
    for (i = 16; i < 80; i++) {
        w[i] = SHA1_ROTL(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    }

    /* Main loop */
    for (i = 0; i < 20; i++) {
        temp = SHA1_ROTL(a, 5) + ((b & c) | ((~b) & d)) + e + sha1_k[0] + w[i];
        e = d; d = c; c = SHA1_ROTL(b, 30); b = a; a = temp;
    }
    for (i = 20; i < 40; i++) {
        temp = SHA1_ROTL(a, 5) + (b ^ c ^ d) + e + sha1_k[1] + w[i];
        e = d; d = c; c = SHA1_ROTL(b, 30); b = a; a = temp;
    }
    for (i = 40; i < 60; i++) {
        temp = SHA1_ROTL(a, 5) + ((b & c) | (b & d) | (c & d)) + e + sha1_k[2] + w[i];
        e = d; d = c; c = SHA1_ROTL(b, 30); b = a; a = temp;
    }
    for (i = 60; i < 80; i++) {
        temp = SHA1_ROTL(a, 5) + (b ^ c ^ d) + e + sha1_k[3] + w[i];
        e = d; d = c; c = SHA1_ROTL(b, 30); b = a; a = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}

void cl_peach_sha1(__private const word32 *in, size_t inlen, __private word32 *out)
{
    word32 state[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    word32 final_block[16];

    final_block[15] = bswap32(inlen << 3);

    /* sha1_update */
    for (; inlen >= 64; inlen -= 64, in = &in[16]) {
        sha1_transform(state, in);
    }

    /* sha1_final - remaining datalen will always be 36 */
    for (int i = 0; i < 9; i++) final_block[i] = in[i];
    final_block[9] = bswap32(0x80000000);
    for (int i = 10; i < 15; i++) final_block[i] = 0;

    sha1_transform(state, final_block);

    /* SHA1 hash = 160 bits, zero fill remaining */
    out[0] = bswap32(state[0]);
    out[1] = bswap32(state[1]);
    out[2] = bswap32(state[2]);
    out[3] = bswap32(state[3]);
    out[4] = bswap32(state[4]);
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
}

/* -------------------- SHA256 Implementation -------------------- */

#define SHA256_SIG0(x) (rotr32(x, 7) ^ rotr32(x, 18) ^ ((x) >> 3))
#define SHA256_SIG1(x) (rotr32(x, 17) ^ rotr32(x, 19) ^ ((x) >> 10))
#define SHA256_EP0(x)  (rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22))
#define SHA256_EP1(x)  (rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25))
#define SHA256_CH(x,y,z) (((x) & (y)) ^ ((~(x)) & (z)))
#define SHA256_MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

void sha256_transform(__private word32 *state, __private const word32 *data)
{
    word32 w[64];
    word32 a, b, c, d, e, f, g, h;
    word32 t1, t2;
    int i;

    /* Prepare message schedule */
    for (i = 0; i < 16; i++) {
        w[i] = bswap32(data[i]);
    }
    for (i = 16; i < 64; i++) {
        w[i] = SHA256_SIG1(w[i-2]) + w[i-7] + SHA256_SIG0(w[i-15]) + w[i-16];
    }

    /* Initialize working variables */
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    /* Main loop */
    for (i = 0; i < 64; i++) {
        t1 = h + SHA256_EP1(e) + SHA256_CH(e, f, g) + sha256_k[i] + w[i];
        t2 = SHA256_EP0(a) + SHA256_MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void cl_peach_sha256(__private const word32 *in, size_t inlen, __private word32 *out)
{
    word32 final_block[16];
    word32 state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    /* prepare bitlen in final */
    final_block[15] = bswap32(inlen << 3);

    /* sha256_update */
    for (; inlen >= 64; inlen -= 64, in = &in[16]) {
        sha256_transform(state, in);
    }

    /* sha256_final - remaining datalen will always be 36 */
    for (int i = 0; i < 9; i++) final_block[i] = in[i];
    final_block[9] = bswap32(0x80000000);
    for (int i = 10; i < 15; i++) final_block[i] = 0;

    sha256_transform(state, final_block);

    /* Output with byte swap */
    for (int i = 0; i < 8; i++) {
        out[i] = bswap32(state[i]);
    }
}

/* Full SHA256 for block trailer hashing */
void cl_sha256(__private const void *data, size_t len, __private word8 *out)
{
    word32 state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    word32 block[16];
    __private const word8 *ptr = (__private const word8 *)data;
    size_t remaining = len;
    int i;

    /* Process full blocks */
    while (remaining >= 64) {
        for (i = 0; i < 16; i++) {
            block[i] = ((__private const word32 *)ptr)[i];
        }
        sha256_transform(state, block);
        ptr += 64;
        remaining -= 64;
    }

    /* Final block */
    for (i = 0; i < 16; i++) block[i] = 0;
    for (i = 0; i < (int)remaining; i++) {
        ((__private word8 *)block)[i] = ptr[i];
    }
    ((__private word8 *)block)[remaining] = 0x80;
    
    if (remaining >= 56) {
        sha256_transform(state, block);
        for (i = 0; i < 16; i++) block[i] = 0;
    }
    
    block[15] = bswap32(len << 3);
    sha256_transform(state, block);

    /* Output */
    for (i = 0; i < 8; i++) {
        ((__private word32 *)out)[i] = bswap32(state[i]);
    }
}

/* -------------------- SHA3/Keccak Implementation -------------------- */

#define KECCAK_ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

void keccakf(__private word64 *st)
{
    word64 t, bc[5];
    int i, j, r;

    for (r = 0; r < 24; r++) {
        /* Theta */
        for (i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ KECCAK_ROTL64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        /* Rho Pi */
        t = st[1];
        st[1]  = KECCAK_ROTL64(st[6], 44);
        st[6]  = KECCAK_ROTL64(st[9], 20);
        st[9]  = KECCAK_ROTL64(st[22], 61);
        st[22] = KECCAK_ROTL64(st[14], 39);
        st[14] = KECCAK_ROTL64(st[20], 18);
        st[20] = KECCAK_ROTL64(st[2], 62);
        st[2]  = KECCAK_ROTL64(st[12], 43);
        st[12] = KECCAK_ROTL64(st[13], 25);
        st[13] = KECCAK_ROTL64(st[19], 8);
        st[19] = KECCAK_ROTL64(st[23], 56);
        st[23] = KECCAK_ROTL64(st[15], 41);
        st[15] = KECCAK_ROTL64(st[4], 27);
        st[4]  = KECCAK_ROTL64(st[24], 14);
        st[24] = KECCAK_ROTL64(st[21], 2);
        st[21] = KECCAK_ROTL64(st[8], 55);
        st[8]  = KECCAK_ROTL64(st[16], 45);
        st[16] = KECCAK_ROTL64(st[5], 36);
        st[5]  = KECCAK_ROTL64(st[3], 28);
        st[3]  = KECCAK_ROTL64(st[18], 21);
        st[18] = KECCAK_ROTL64(st[17], 15);
        st[17] = KECCAK_ROTL64(st[11], 10);
        st[11] = KECCAK_ROTL64(st[7], 6);
        st[7]  = KECCAK_ROTL64(st[10], 3);
        st[10] = KECCAK_ROTL64(t, 1);

        /* Chi */
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        /* Iota */
        st[0] ^= keccakf_rndc[r];
    }
}

void cl_peach_sha3(__private const word64 *in, size_t inlen, int keccak_final,
    __private word64 *out)
{
    word64 state[25] = { 0 };
    int i;

    /* sha3_update - 136 is ctx->rsiz, fill only 17x 64-bit words in state */
    for (; inlen >= 136; inlen -= 136, in = &in[17]) {
        for (i = 0; i < 17; i++) state[i] ^= in[i];
        keccakf(state);
    }

    /* sha3_final */
    state[0] ^= in[0];
    state[1] ^= in[1];
    state[2] ^= in[2];
    state[3] ^= in[3];
    if (inlen > PEACHGENLEN) {
        state[4] ^= in[4];
        state[5] ^= in[5];
        state[6] ^= in[6];
        state[7] ^= in[7];
        state[8] ^= in[8];
        state[9] ^= in[9];
        state[10] ^= in[10];
        state[11] ^= in[11];
        state[12] ^= in[12];
        ((__private word32 *)state)[26] ^= ((__private const word32 *)in)[26];
        ((__private word8 *)state)[108] ^= keccak_final ? 0x01 : 0x06;
    } else {
        ((__private word32 *)state)[8] ^= ((__private const word32 *)in)[8];
        ((__private word8 *)state)[36] ^= keccak_final ? 0x01 : 0x06;
    }
    ((__private word8 *)state)[135] ^= 0x80;
    keccakf(state);

    /* sha3_output */
    out[0] = state[0];
    out[1] = state[1];
    out[2] = state[2];
    out[3] = state[3];
}

/* -------------------- Deterministic Float Operations -------------------- */

__constant word32 c_float[4] = {
    WORD32_C(0x26C34), WORD32_C(0x14198),
    WORD32_C(0x3D6EC), WORD32_C(0x80000000)
};

word32 cl_peach_dflops(__private void *data, size_t len, word32 index, int txf)
{
    __private word8 *bp;
    float temp, flv;
    int32 operand;
    word32 op;
    unsigned i;
    word8 shift;

    for (op = i = 0; i < len; i += 4) {
        bp = &((__private word8 *)data)[i];
        float *flp;
        if (txf) {
            flp = (__private float *)bp;
        } else {
            temp = *((__private float *)bp);
            flp = &temp;
        }
        /* first byte allocated to determine shift amount */
        shift = ((*bp & 7) + 1) << 1;
        /* determine operation type, operand value, and sign */
        op += bp[((c_float[0] >> shift) & 3)];
        operand = bp[((c_float[1] >> shift) & 3)];
        if (bp[((c_float[2] >> shift) & 3)] & 1) operand ^= c_float[3];
        
        /* interpret operand as SIGNED integer and cast to float */
        flv = (float)operand;
        
        /* Replace pre-operation NaN with index */
        if (isnan(*flp)) *flp = (float)index;
        
        /* Perform predetermined floating point operation */
        switch (op & 3) {
            case 3: *flp = *flp / flv; break;
            case 2: *flp = *flp * flv; break;
            case 1: *flp = *flp - flv; break;
            case 0: *flp = *flp + flv; break;
        }
        
        /* Replace post-operation NaN with index */
        if (isnan(*flp)) *flp = (float)index;
        
        /* Add result of the operation to `op` as an array of bytes */
        bp = (__private word8 *)flp;
        op += bp[0];
        op += bp[1];
        op += bp[2];
        op += bp[3];
    }

    return op;
}

/* -------------------- Deterministic Memory Transformations -------------------- */

word32 cl_peach_dmemtx(__private void *data, size_t len, word32 op)
{
    __constant word64 c_flip64 = WORD64_C(0x8181818181818181);
    __constant word32 c_flip32 = WORD32_C(0x81818181);
    __private word64 *qp = (__private word64 *)data;
    __private word32 *dp = (__private word32 *)data;
    __private word8 *bp = (__private word8 *)data;
    size_t len16, len32, len64, y;
    unsigned i, z;
    word8 temp;

    /* prepare memory pointers and lengths */
    len64 = (len32 = (len16 = len >> 1) >> 1) >> 1;
    
    /* perform memory transformations multiple times */
    for (i = 0; i < PEACHROUNDS; i++) {
        op += bp[i];
        switch (op & 7) {
            case 0:  /* flip the first and last bit in every byte */
                for (z = 0; z < len64; z++) qp[z] ^= c_flip64;
                for (z <<= 1; z < len32; z++) dp[z] ^= c_flip32;
                break;
            case 1:  /* Swap bytes */
                for (y = len16, z = 0; z < len16; y++, z++) {
                    temp = bp[z]; bp[z] = bp[y]; bp[y] = temp;
                }
                break;
            case 2:  /* 1's complement, all bytes */
                for (z = 0; z < len64; z++) qp[z] = ~qp[z];
                for (z <<= 1; z < len32; z++) dp[z] = ~dp[z];
                break;
            case 3:  /* Alternate +1 and -1 on all bytes */
                for (z = 0; z < len; z++) bp[z] += (z & 1) ? -1 : 1;
                break;
            case 4:  /* Alternate -i and +i on all bytes */
                for (z = 0; z < len; z++) bp[z] += (word8)((z & 1) ? i : -i);
                break;
            case 5:  /* Replace every occurrence of 104 with 72 */
                for (z = 0; z < len; z++) if (bp[z] == 104) bp[z] = 72;
                break;
            case 6:  /* If byte a is > byte b, swap them. */
                for (y = len16, z = 0; z < len16; y++, z++) {
                    if (bp[z] > bp[y]) {
                        temp = bp[z]; bp[z] = bp[y]; bp[y] = temp;
                    }
                }
                break;
            case 7:  /* XOR all bytes */
                for (y = 0, z = 1; z < len; y++, z++) bp[z] ^= bp[y];
                break;
        }
    }

    return op;
}

/* -------------------- Nighthash -------------------- */

void cl_peach_nighthash(__private word64 *in, size_t inlen,
    word32 index, size_t txlen, __private word64 *out)
{
    if (txlen) {
        index = cl_peach_dflops(in, txlen, index, 1);
        index = cl_peach_dmemtx(in, txlen, index);
    } else {
        index = cl_peach_dflops(in, inlen, index, 0);
    }

    switch (index & 7) {
        case 0: cl_peach_blake2b(in, inlen, 32, out); break;
        case 1: cl_peach_blake2b(in, inlen, 64, out); break;
        case 2: cl_peach_sha1((__private word32 *)in, inlen, (__private word32 *)out); break;
        case 3: cl_peach_sha256((__private word32 *)in, inlen, (__private word32 *)out); break;
        case 4: cl_peach_sha3(in, inlen, 0, out); break;
        case 5: cl_peach_sha3(in, inlen, 1, out); break;
        case 6: cl_peach_md2(in, inlen, out); break;
        case 7: cl_peach_md5((__private word32 *)in, inlen, out); break;
    }
}

/* -------------------- Peach Tile Generation -------------------- */

void cl_peach_generate(word32 index, __private word64 *tilep, __global word32 *phash)
{
    int i;

    /* place initial data into seed */
    ((__private word32 *)tilep)[0] = index;
    ((__private word32 *)tilep)[1] = phash[0];
    ((__private word32 *)tilep)[2] = phash[1];
    ((__private word32 *)tilep)[3] = phash[2];
    ((__private word32 *)tilep)[4] = phash[3];
    ((__private word32 *)tilep)[5] = phash[4];
    ((__private word32 *)tilep)[6] = phash[5];
    ((__private word32 *)tilep)[7] = phash[6];
    ((__private word32 *)tilep)[8] = phash[7];
    
    /* perform initial nighthash into first row of tile */
    cl_peach_nighthash(tilep, PEACHGENLEN, index, PEACHGENLEN, tilep);
    
    /* fill the rest of the tile with the preceding Nighthash result */
    for (i = 0; i < (PEACHTILELEN64 - 4); i += 4) {
        tilep[i + 4] = index;
        cl_peach_nighthash(&tilep[i], PEACHGENLEN, index, SHA256LEN, &tilep[i + 4]);
    }
}

/* -------------------- Peach Jump -------------------- */

void cl_peach_jump(__private word32 *index, __private word64 *nonce, 
    __global word64 *tilep)
{
    word64 seed[(PEACHJUMPLEN / 8) + 1];
    __private word32 *dp = (__private word32 *)seed;
    int i;

    /* construct seed for use as Nighthash input */
    seed[0] = nonce[0];
    seed[1] = nonce[1];
    seed[2] = nonce[2];
    seed[3] = nonce[3];
    dp[8] = *index;
    for (i = 0; i < PEACHTILELEN32; i++) {
        dp[i + 9] = ((__global word32 *)tilep)[i];
    }

    /* perform nighthash on PEACHJUMPLEN bytes of seed */
    cl_peach_nighthash(seed, PEACHJUMPLEN, *index, 0, seed);
    
    /* sum hash as 8x 32-bit unsigned integers */
    *index = dp[0] + dp[1] + dp[2] + dp[3] + dp[4] + dp[5] + dp[6] + dp[7];
    *index &= PEACHCACHELEN_M1;
}

/* -------------------- PRNG (SplitMix64 based) -------------------- */

__kernel void kcl_srand64(__global word64 *d_state, word64 seed)
{
    word64 index = get_global_id(0);
    d_state[index] = (seed ^ (index * WORD64_C(0x9e3779b97f4a7c15))) * 
                     WORD64_C(0xc6bc279692b5c323);
}

word64 cl_rand64(__global word64 *d_state)
{
    word64 index = get_global_id(0);
    word64 z = (d_state[index] += WORD64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * WORD64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * WORD64_C(0x94d049bb133111eb);
    return (d_state[index] = z ^ (z >> 31));
}

/* -------------------- Peach Map Building Kernel -------------------- */

__kernel void kcl_peach_build(
    word32 offset,
    __global word64 *d_map,
    __global word32 *d_phash)
{
    const word32 index = get_global_id(0) + offset;
    if (index < PEACHCACHELEN) {
        word64 tile[PEACHTILELEN64];
        cl_peach_generate(index, tile, d_phash);
        /* Copy tile to global memory */
        for (int i = 0; i < PEACHTILELEN64; i++) {
            d_map[index * PEACHTILELEN64 + i] = tile[i];
        }
    }
}

/* -------------------- Peach Solving Kernel -------------------- */

/* Block trailer structure for kernel */
typedef struct __attribute__((packed)) {
    word8 phash[32];
    word8 bnum[8];
    word8 mfee[8];
    word8 tcount[4];
    word8 time0[4];
    word8 difficulty[4];
    word8 mroot[32];
    word8 nonce[32];
    word8 stime[4];
    word8 bhash[32];
} BTRAILER_CL;

__kernel void kcl_peach_solve(
    __global word64 *d_map,
    __global BTRAILER_CL *d_bt,
    __global word64 *d_state,
    word8 diff,
    __global word64 *d_solve)
{
    word64 nonce[4], seed;
    word8 hash[SHA256LEN];
    word32 mario, i;

    /* extract nonce from trailer */
    for (i = 0; i < 4; i++) {
        ((__private word32 *)nonce)[i] = ((__global word32 *)d_bt->nonce)[i];
    }

    /* generate last half of nonce from seed */
    seed = cl_rand64(d_state);
    nonce[2] = WORD64_C(0x10000050000) |
        Z_ING[(seed     )  & 31]       |
       Z_PREP[(seed >> 5)  &  7] <<  8 |
        Z_ADJ[(seed >> 8)  & 63] << 24 |
         Z_NS[(seed >> 14) & 63] << 32 |
       Z_MASS[(seed >> 20) & 31] << 48 |
        Z_ING[(seed >> 25) & 31] << 56;
    nonce[3] = WORD64_C(0x50103) |
        Z_ADJ[(seed >> 30) & 63] << 24 |
         Z_NS[(seed >> 36) & 63] << 32;

    /* sha256 hash trailer and nonce */
    {
        word8 buffer[124];
        for (i = 0; i < 92; i++) buffer[i] = ((__global word8 *)d_bt)[i];
        for (i = 0; i < 32; i++) buffer[92 + i] = ((__private word8 *)nonce)[i];
        cl_sha256(buffer, 124, hash);
    }

    /* initialize mario's starting index on the map */
    for (mario = hash[0], i = 1; i < SHA256LEN; i++) {
        mario *= hash[i];
    }
    mario &= PEACHCACHELEN_M1;

    /* perform tile jumps to find the final tile */
    for (i = 0; i < PEACHROUNDS; i++) {
        cl_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN64]);
    }

    /* hash block trailer with final tile */
    {
        word8 buffer[SHA256LEN + PEACHTILELEN];
        for (i = 0; i < SHA256LEN; i++) buffer[i] = hash[i];
        for (i = 0; i < PEACHTILELEN; i++) {
            buffer[SHA256LEN + i] = ((__global word8 *)&d_map[mario * PEACHTILELEN64])[i];
        }
        cl_sha256(buffer, SHA256LEN + PEACHTILELEN, hash);
    }

    /* Coarse/Fine evaluation checks */
    __private word32 *x = (__private word32 *)hash;
    for (i = diff >> 5; i; i--) if (*(x++) != 0) return;
    if (clz32(byte_perm(*x, 0, 0x0123)) < (diff & 31)) return;

    /* check first to solve with atomic solve handling */
    if (atomic_cmpxchg((__global int *)d_solve, 0, *((__private int *)nonce)) == 0) {
        d_solve[0] = nonce[0];
        d_solve[1] = nonce[1];
        d_solve[2] = nonce[2];
        d_solve[3] = nonce[3];
    }
}

