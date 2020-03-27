/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define SECUREC_INLINE_DO_MEMSET   1

#include "securecutil.h"

#ifndef SECUREC_MEMSET_WITH_PERFORMANCE
#define SECUREC_MEMSET_WITH_PERFORMANCE 0
#endif

#define SECUREC_MEMSET_PARAM_OK(dest, destMax, count) (SECUREC_LIKELY((count) <= (destMax) && \
    (dest) != NULL && (destMax) <= SECUREC_MEM_MAX_LEN))


#if SECUREC_WITH_PERFORMANCE_ADDONS || SECUREC_MEMSET_WITH_PERFORMANCE
/*
 * Determine whether the address is 8-byte aligned, use static to increase performance
 * return 0 is aligned
 */
static int SecIsAddrAligned8(const void *addr, const void *zeroAddr)
{
    return (int)(((size_t)((const char*)addr - (const char*)zeroAddr)) & 7); /* use 7 to check aligned 8 */
}

/* use union to clear strict-aliasing warning */
typedef union {
    SecStrBuf32 buf32;
    SecStrBuf31 buf31;
    SecStrBuf30 buf30;
    SecStrBuf29 buf29;
    SecStrBuf28 buf28;
    SecStrBuf27 buf27;
    SecStrBuf26 buf26;
    SecStrBuf25 buf25;
    SecStrBuf24 buf24;
    SecStrBuf23 buf23;
    SecStrBuf22 buf22;
    SecStrBuf21 buf21;
    SecStrBuf20 buf20;
    SecStrBuf19 buf19;
    SecStrBuf18 buf18;
    SecStrBuf17 buf17;
    SecStrBuf16 buf16;
    SecStrBuf15 buf15;
    SecStrBuf14 buf14;
    SecStrBuf13 buf13;
    SecStrBuf12 buf12;
    SecStrBuf11 buf11;
    SecStrBuf10 buf10;
    SecStrBuf9 buf9;
    SecStrBuf8 buf8;
    SecStrBuf7 buf7;
    SecStrBuf6 buf6;
    SecStrBuf5 buf5;
    SecStrBuf4 buf4;
    SecStrBuf3 buf3;
    SecStrBuf2 buf2;
    SecStrBuf1 buf1;
} SecStrBuf32Union;
/* C standard initializes the first member of the consortium. */
static const SecStrBuf32 g_allZero = {{
    '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
    '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
    '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
    '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
}};
static const SecStrBuf32 g_allFF = {{
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
}};

static const SecStrBuf32Union *SecStrictAliasingCast(const SecStrBuf32 *buf)
{
    return (const SecStrBuf32Union *)buf;
}

#ifndef SECUREC_MEMSET_THRESHOLD_SIZE
#define SECUREC_MEMSET_THRESHOLD_SIZE 32UL
#endif

#define SECUREC_UNALIGNED_SET do { \
    char *pcDest = (char *)dest; \
    switch (count) { \
        case 32: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 31: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 30: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 29: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 28: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 27: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 26: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 25: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 24: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 23: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 22: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 21: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 20: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 19: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 18: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 17: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 16: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 15: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 14: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 13: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 12: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 11: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 10: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 9: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 8: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 7: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 6: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 5: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 4: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 3: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 2: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        case 1: \
            *(pcDest++) = (char)c; \
            /* fall-through */ /* FALLTHRU */ \
        default: \
            break; \
    } \
} SECUREC_WHILE_ZERO

#define SECUREC_ALIGNED_SET_OPT_ZERO_FF do { \
    switch (c) { \
        case 0: \
            switch (count) { \
                case 1: \
                    *(SecStrBuf1 *)dest = *(const SecStrBuf1 *)(&((SecStrictAliasingCast(&g_allZero))->buf1)); \
                    break; \
                case 2: \
                    *(SecStrBuf2 *)dest = *(const SecStrBuf2 *)(&((SecStrictAliasingCast(&g_allZero))->buf2)); \
                    break; \
                case 3: \
                    *(SecStrBuf3 *)dest = *(const SecStrBuf3 *)(&((SecStrictAliasingCast(&g_allZero))->buf3)); \
                    break; \
                case 4: \
                    *(SecStrBuf4 *)dest = *(const SecStrBuf4 *)(&((SecStrictAliasingCast(&g_allZero))->buf4)); \
                    break; \
                case 5: \
                    *(SecStrBuf5 *)dest = *(const SecStrBuf5 *)(&((SecStrictAliasingCast(&g_allZero))->buf5)); \
                    break; \
                case 6: \
                    *(SecStrBuf6 *)dest = *(const SecStrBuf6 *)(&((SecStrictAliasingCast(&g_allZero))->buf6)); \
                    break; \
                case 7: \
                    *(SecStrBuf7 *)dest = *(const SecStrBuf7 *)(&((SecStrictAliasingCast(&g_allZero))->buf7)); \
                    break; \
                case 8: \
                    *(SecStrBuf8 *)dest = *(const SecStrBuf8 *)(&((SecStrictAliasingCast(&g_allZero))->buf8)); \
                    break; \
                case 9: \
                    *(SecStrBuf9 *)dest = *(const SecStrBuf9 *)(&((SecStrictAliasingCast(&g_allZero))->buf9)); \
                    break; \
                case 10: \
                    *(SecStrBuf10 *)dest = *(const SecStrBuf10 *)(&((SecStrictAliasingCast(&g_allZero))->buf10)); \
                    break; \
                case 11: \
                    *(SecStrBuf11 *)dest = *(const SecStrBuf11 *)(&((SecStrictAliasingCast(&g_allZero))->buf11)); \
                    break; \
                case 12: \
                    *(SecStrBuf12 *)dest = *(const SecStrBuf12 *)(&((SecStrictAliasingCast(&g_allZero))->buf12)); \
                    break; \
                case 13: \
                    *(SecStrBuf13 *)dest = *(const SecStrBuf13 *)(&((SecStrictAliasingCast(&g_allZero))->buf13)); \
                    break; \
                case 14: \
                    *(SecStrBuf14 *)dest = *(const SecStrBuf14 *)(&((SecStrictAliasingCast(&g_allZero))->buf14)); \
                    break; \
                case 15: \
                    *(SecStrBuf15 *)dest = *(const SecStrBuf15 *)(&((SecStrictAliasingCast(&g_allZero))->buf15)); \
                    break; \
                case 16: \
                    *(SecStrBuf16 *)dest = *(const SecStrBuf16 *)(&((SecStrictAliasingCast(&g_allZero))->buf16)); \
                    break; \
                case 17: \
                    *(SecStrBuf17 *)dest = *(const SecStrBuf17 *)(&((SecStrictAliasingCast(&g_allZero))->buf17)); \
                    break; \
                case 18: \
                    *(SecStrBuf18 *)dest = *(const SecStrBuf18 *)(&((SecStrictAliasingCast(&g_allZero))->buf18)); \
                    break; \
                case 19: \
                    *(SecStrBuf19 *)dest = *(const SecStrBuf19 *)(&((SecStrictAliasingCast(&g_allZero))->buf19)); \
                    break; \
                case 20: \
                    *(SecStrBuf20 *)dest = *(const SecStrBuf20 *)(&((SecStrictAliasingCast(&g_allZero))->buf20)); \
                    break; \
                case 21: \
                    *(SecStrBuf21 *)dest = *(const SecStrBuf21 *)(&((SecStrictAliasingCast(&g_allZero))->buf21)); \
                    break; \
                case 22: \
                    *(SecStrBuf22 *)dest = *(const SecStrBuf22 *)(&((SecStrictAliasingCast(&g_allZero))->buf22)); \
                    break; \
                case 23: \
                    *(SecStrBuf23 *)dest = *(const SecStrBuf23 *)(&((SecStrictAliasingCast(&g_allZero))->buf23)); \
                    break; \
                case 24: \
                    *(SecStrBuf24 *)dest = *(const SecStrBuf24 *)(&((SecStrictAliasingCast(&g_allZero))->buf24)); \
                    break; \
                case 25: \
                    *(SecStrBuf25 *)dest = *(const SecStrBuf25 *)(&((SecStrictAliasingCast(&g_allZero))->buf25)); \
                    break; \
                case 26: \
                    *(SecStrBuf26 *)dest = *(const SecStrBuf26 *)(&((SecStrictAliasingCast(&g_allZero))->buf26)); \
                    break; \
                case 27: \
                    *(SecStrBuf27 *)dest = *(const SecStrBuf27 *)(&((SecStrictAliasingCast(&g_allZero))->buf27)); \
                    break; \
                case 28: \
                    *(SecStrBuf28 *)dest = *(const SecStrBuf28 *)(&((SecStrictAliasingCast(&g_allZero))->buf28)); \
                    break; \
                case 29: \
                    *(SecStrBuf29 *)dest = *(const SecStrBuf29 *)(&((SecStrictAliasingCast(&g_allZero))->buf29)); \
                    break; \
                case 30: \
                    *(SecStrBuf30 *)dest = *(const SecStrBuf30 *)(&((SecStrictAliasingCast(&g_allZero))->buf30)); \
                    break; \
                case 31: \
                    *(SecStrBuf31 *)dest = *(const SecStrBuf31 *)(&((SecStrictAliasingCast(&g_allZero))->buf31)); \
                    break; \
                case 32: \
                    *(SecStrBuf32 *)dest = *(const SecStrBuf32 *)(&((SecStrictAliasingCast(&g_allZero))->buf32)); \
                    break; \
                default: \
                    break; \
            } \
            break; \
        case 0xFF: \
            switch (count) { \
                case 1: \
                    *(SecStrBuf1 *)dest = *(const SecStrBuf1 *)(&((SecStrictAliasingCast(&g_allFF))->buf1)); \
                    break; \
                case 2: \
                    *(SecStrBuf2 *)dest = *(const SecStrBuf2 *)(&((SecStrictAliasingCast(&g_allFF))->buf2)); \
                    break; \
                case 3: \
                    *(SecStrBuf3 *)dest = *(const SecStrBuf3 *)(&((SecStrictAliasingCast(&g_allFF))->buf3)); \
                    break; \
                case 4: \
                    *(SecStrBuf4 *)dest = *(const SecStrBuf4 *)(&((SecStrictAliasingCast(&g_allFF))->buf4)); \
                    break; \
                case 5: \
                    *(SecStrBuf5 *)dest = *(const SecStrBuf5 *)(&((SecStrictAliasingCast(&g_allFF))->buf5)); \
                    break; \
                case 6: \
                    *(SecStrBuf6 *)dest = *(const SecStrBuf6 *)(&((SecStrictAliasingCast(&g_allFF))->buf6)); \
                    break; \
                case 7: \
                    *(SecStrBuf7 *)dest = *(const SecStrBuf7 *)(&((SecStrictAliasingCast(&g_allFF))->buf7)); \
                    break; \
                case 8: \
                    *(SecStrBuf8 *)dest = *(const SecStrBuf8 *)(&((SecStrictAliasingCast(&g_allFF))->buf8)); \
                    break; \
                case 9: \
                    *(SecStrBuf9 *)dest = *(const SecStrBuf9 *)(&((SecStrictAliasingCast(&g_allFF))->buf9)); \
                    break; \
                case 10: \
                    *(SecStrBuf10 *)dest = *(const SecStrBuf10 *)(&((SecStrictAliasingCast(&g_allFF))->buf10)); \
                    break; \
                case 11: \
                    *(SecStrBuf11 *)dest = *(const SecStrBuf11 *)(&((SecStrictAliasingCast(&g_allFF))->buf11)); \
                    break; \
                case 12: \
                    *(SecStrBuf12 *)dest = *(const SecStrBuf12 *)(&((SecStrictAliasingCast(&g_allFF))->buf12)); \
                    break; \
                case 13: \
                    *(SecStrBuf13 *)dest = *(const SecStrBuf13 *)(&((SecStrictAliasingCast(&g_allFF))->buf13)); \
                    break; \
                case 14: \
                    *(SecStrBuf14 *)dest = *(const SecStrBuf14 *)(&((SecStrictAliasingCast(&g_allFF))->buf14)); \
                    break; \
                case 15: \
                    *(SecStrBuf15 *)dest = *(const SecStrBuf15 *)(&((SecStrictAliasingCast(&g_allFF))->buf15)); \
                    break; \
                case 16: \
                    *(SecStrBuf16 *)dest = *(const SecStrBuf16 *)(&((SecStrictAliasingCast(&g_allFF))->buf16)); \
                    break; \
                case 17: \
                    *(SecStrBuf17 *)dest = *(const SecStrBuf17 *)(&((SecStrictAliasingCast(&g_allFF))->buf17)); \
                    break; \
                case 18: \
                    *(SecStrBuf18 *)dest = *(const SecStrBuf18 *)(&((SecStrictAliasingCast(&g_allFF))->buf18)); \
                    break; \
                case 19: \
                    *(SecStrBuf19 *)dest = *(const SecStrBuf19 *)(&((SecStrictAliasingCast(&g_allFF))->buf19)); \
                    break; \
                case 20: \
                    *(SecStrBuf20 *)dest = *(const SecStrBuf20 *)(&((SecStrictAliasingCast(&g_allFF))->buf20)); \
                    break; \
                case 21: \
                    *(SecStrBuf21 *)dest = *(const SecStrBuf21 *)(&((SecStrictAliasingCast(&g_allFF))->buf21)); \
                    break; \
                case 22: \
                    *(SecStrBuf22 *)dest = *(const SecStrBuf22 *)(&((SecStrictAliasingCast(&g_allFF))->buf22)); \
                    break; \
                case 23: \
                    *(SecStrBuf23 *)dest = *(const SecStrBuf23 *)(&((SecStrictAliasingCast(&g_allFF))->buf23)); \
                    break; \
                case 24: \
                    *(SecStrBuf24 *)dest = *(const SecStrBuf24 *)(&((SecStrictAliasingCast(&g_allFF))->buf24)); \
                    break; \
                case 25: \
                    *(SecStrBuf25 *)dest = *(const SecStrBuf25 *)(&((SecStrictAliasingCast(&g_allFF))->buf25)); \
                    break; \
                case 26: \
                    *(SecStrBuf26 *)dest = *(const SecStrBuf26 *)(&((SecStrictAliasingCast(&g_allFF))->buf26)); \
                    break; \
                case 27: \
                    *(SecStrBuf27 *)dest = *(const SecStrBuf27 *)(&((SecStrictAliasingCast(&g_allFF))->buf27)); \
                    break; \
                case 28: \
                    *(SecStrBuf28 *)dest = *(const SecStrBuf28 *)(&((SecStrictAliasingCast(&g_allFF))->buf28)); \
                    break; \
                case 29: \
                    *(SecStrBuf29 *)dest = *(const SecStrBuf29 *)(&((SecStrictAliasingCast(&g_allFF))->buf29)); \
                    break; \
                case 30: \
                    *(SecStrBuf30 *)dest = *(const SecStrBuf30 *)(&((SecStrictAliasingCast(&g_allFF))->buf30)); \
                    break; \
                case 31: \
                    *(SecStrBuf31 *)dest = *(const SecStrBuf31 *)(&((SecStrictAliasingCast(&g_allFF))->buf31)); \
                    break; \
                case 32: \
                    *(SecStrBuf32 *)dest = *(const SecStrBuf32 *)(&((SecStrictAliasingCast(&g_allFF))->buf32)); \
                    break; \
                default: \
                    break; \
            } \
            break; \
        default: \
            SECUREC_UNALIGNED_SET; \
    } /* END switch */ \
} SECUREC_WHILE_ZERO
#endif

/*
 * Handling errors
 */
static errno_t SecMemsetError(void *dest, size_t destMax, int c, size_t count)
{
    if (destMax == 0 || destMax > SECUREC_MEM_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("memset_s");
        return ERANGE;
    }
    if (dest == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("memset_s");
        return EINVAL;
    }
    if (count > destMax) {
        (void)memset(dest, c, destMax); /* set entire buffer to value c */
        SECUREC_ERROR_INVALID_RANGE("memset_s");
        return ERANGE_AND_RESET;
    }
    return EOK;
}

#if SECUREC_WITH_PERFORMANCE_ADDONS || SECUREC_MEMSET_WITH_PERFORMANCE
/*
 * Performance optimization
 */
static void SecDoMemsetOpt(void *dest, int c, size_t count)
{
    if (count > SECUREC_MEMSET_THRESHOLD_SIZE) {
        SecDoMemset(dest, c, count);
    } else {
        if (SECUREC_ADDR_ALIGNED_8(dest)) {
            /* use struct assignment */
            SECUREC_ALIGNED_SET_OPT_ZERO_FF;
        } else {
            SECUREC_UNALIGNED_SET;
        }
    }
    return;
}
#endif

/*
 * <FUNCTION DESCRIPTION>
 *    The memset_s function copies the value of c (converted to an unsigned char)
 *     into each of the first count characters of the object pointed to by dest.
 *
 * <INPUT PARAMETERS>
 *    dest                           Pointer to destination.
 *    destMax                     The size of the buffer.
 *    c                               Character to set.
 *    count                          Number of characters.
 *
 * <OUTPUT PARAMETERS>
 *    dest buffer                   is uptdated.
 *
 * <RETURN VALUE>
 *    EOK                            Success
 *    EINVAL                        dest == NULL and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN
 *    ERANGE                       destMax is  0 or destMax > SECUREC_MEM_MAX_LEN
 *    ERANGE_AND_RESET    count > destMax and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN and dest != NULL
 *
 *    if return ERANGE_AND_RESET then fill dest to c ,fill length is destMax
 */
errno_t memset_s(void *dest, size_t destMax, int c, size_t count)
{
    if (SECUREC_MEMSET_PARAM_OK(dest, destMax, count)) {
#if SECUREC_MEMSET_WITH_PERFORMANCE
        SecDoMemsetOpt(dest, c, count);
#else
        SecDoMemset(dest, c, count);
#endif
        return EOK;
    } else {
        /* meet some runtime violation, return error code */
        return SecMemsetError(dest, destMax, c, count);
    }
}

#if SECUREC_IN_KERNEL
EXPORT_SYMBOL(memset_s);
#endif

#if SECUREC_WITH_PERFORMANCE_ADDONS
/*
 * Performance optimization
 */
errno_t memset_sOptAsm(void *dest, size_t destMax, int c, size_t count)
{
    if (SECUREC_MEMSET_PARAM_OK(dest, destMax, count)) {
        SecDoMemsetOpt(dest, c, count);
        return EOK;
    }
    /* meet some runtime violation, return error code */
    return SecMemsetError(dest, destMax, c, count);
}

/*
 * Performance optimization
 */
errno_t memset_sOptTc(void *dest, size_t destMax, int c, size_t count)
{
    if (SECUREC_LIKELY(count <= destMax && dest != NULL)) {
        SecDoMemsetOpt(dest, c, count);
        return EOK;
    }
    /* meet some runtime violation, return error code */
    return SecMemsetError(dest, destMax, c, count);
}
#endif

