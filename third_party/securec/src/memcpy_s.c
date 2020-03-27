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

#define SECUREC_INLINE_DO_MEMCPY   1
#include "securecutil.h"

#ifndef SECUREC_MEMCOPY_WITH_PERFORMANCE
#define SECUREC_MEMCOPY_WITH_PERFORMANCE 0
#endif

#if SECUREC_WITH_PERFORMANCE_ADDONS || SECUREC_MEMCOPY_WITH_PERFORMANCE
#ifndef SECUREC_MEMCOPY_THRESHOLD_SIZE
#define SECUREC_MEMCOPY_THRESHOLD_SIZE 64UL
#endif
/*
 * Determine whether the address is 8-byte aligned, use static to increase performance
 * return 0 is aligned
 */
static int SecIsAddrAligned8(const void *addr, const void *zeroAddr)
{
    return (int)(((size_t)((const char*)addr - (const char*)zeroAddr)) & 7); /* use 7 to check aligned 8 */
}

#define SECUREC_SMALL_MEM_COPY do { \
    if (SECUREC_ADDR_ALIGNED_8(dest) && SECUREC_ADDR_ALIGNED_8(src)) { \
        /* use struct assignment */ \
        switch (count) { \
            case 1: \
                *(SecStrBuf1 *)dest = *(const SecStrBuf1 *)src; \
                break; \
            case 2: \
                *(SecStrBuf2 *)dest = *(const SecStrBuf2 *)src; \
                break; \
            case 3: \
                *(SecStrBuf3 *)dest = *(const SecStrBuf3 *)src; \
                break; \
            case 4: \
                *(SecStrBuf4 *)dest = *(const SecStrBuf4 *)src; \
                break; \
            case 5: \
                *(SecStrBuf5 *)dest = *(const SecStrBuf5 *)src; \
                break; \
            case 6: \
                *(SecStrBuf6 *)dest = *(const SecStrBuf6 *)src; \
                break; \
            case 7: \
                *(SecStrBuf7 *)dest = *(const SecStrBuf7 *)src; \
                break; \
            case 8: \
                *(SecStrBuf8 *)dest = *(const SecStrBuf8 *)src; \
                break; \
            case 9: \
                *(SecStrBuf9 *)dest = *(const SecStrBuf9 *)src; \
                break; \
            case 10: \
                *(SecStrBuf10 *)dest = *(const SecStrBuf10 *)src; \
                break; \
            case 11: \
                *(SecStrBuf11 *)dest = *(const SecStrBuf11 *)src; \
                break; \
            case 12: \
                *(SecStrBuf12 *)dest = *(const SecStrBuf12 *)src; \
                break; \
            case 13: \
                *(SecStrBuf13 *)dest = *(const SecStrBuf13 *)src; \
                break; \
            case 14: \
                *(SecStrBuf14 *)dest = *(const SecStrBuf14 *)src; \
                break; \
            case 15: \
                *(SecStrBuf15 *)dest = *(const SecStrBuf15 *)src; \
                break; \
            case 16: \
                *(SecStrBuf16 *)dest = *(const SecStrBuf16 *)src; \
                break; \
            case 17: \
                *(SecStrBuf17 *)dest = *(const SecStrBuf17 *)src; \
                break; \
            case 18: \
                *(SecStrBuf18 *)dest = *(const SecStrBuf18 *)src; \
                break; \
            case 19: \
                *(SecStrBuf19 *)dest = *(const SecStrBuf19 *)src; \
                break; \
            case 20: \
                *(SecStrBuf20 *)dest = *(const SecStrBuf20 *)src; \
                break; \
            case 21: \
                *(SecStrBuf21 *)dest = *(const SecStrBuf21 *)src; \
                break; \
            case 22: \
                *(SecStrBuf22 *)dest = *(const SecStrBuf22 *)src; \
                break; \
            case 23: \
                *(SecStrBuf23 *)dest = *(const SecStrBuf23 *)src; \
                break; \
            case 24: \
                *(SecStrBuf24 *)dest = *(const SecStrBuf24 *)src; \
                break; \
            case 25: \
                *(SecStrBuf25 *)dest = *(const SecStrBuf25 *)src; \
                break; \
            case 26: \
                *(SecStrBuf26 *)dest = *(const SecStrBuf26 *)src; \
                break; \
            case 27: \
                *(SecStrBuf27 *)dest = *(const SecStrBuf27 *)src; \
                break; \
            case 28: \
                *(SecStrBuf28 *)dest = *(const SecStrBuf28 *)src; \
                break; \
            case 29: \
                *(SecStrBuf29 *)dest = *(const SecStrBuf29 *)src; \
                break; \
            case 30: \
                *(SecStrBuf30 *)dest = *(const SecStrBuf30 *)src; \
                break; \
            case 31: \
                *(SecStrBuf31 *)dest = *(const SecStrBuf31 *)src; \
                break; \
            case 32: \
                *(SecStrBuf32 *)dest = *(const SecStrBuf32 *)src; \
                break; \
            case 33: \
                *(SecStrBuf33 *)dest = *(const SecStrBuf33 *)src; \
                break; \
            case 34: \
                *(SecStrBuf34 *)dest = *(const SecStrBuf34 *)src; \
                break; \
            case 35: \
                *(SecStrBuf35 *)dest = *(const SecStrBuf35 *)src; \
                break; \
            case 36: \
                *(SecStrBuf36 *)dest = *(const SecStrBuf36 *)src; \
                break; \
            case 37: \
                *(SecStrBuf37 *)dest = *(const SecStrBuf37 *)src; \
                break; \
            case 38: \
                *(SecStrBuf38 *)dest = *(const SecStrBuf38 *)src; \
                break; \
            case 39: \
                *(SecStrBuf39 *)dest = *(const SecStrBuf39 *)src; \
                break; \
            case 40: \
                *(SecStrBuf40 *)dest = *(const SecStrBuf40 *)src; \
                break; \
            case 41: \
                *(SecStrBuf41 *)dest = *(const SecStrBuf41 *)src; \
                break; \
            case 42: \
                *(SecStrBuf42 *)dest = *(const SecStrBuf42 *)src; \
                break; \
            case 43: \
                *(SecStrBuf43 *)dest = *(const SecStrBuf43 *)src; \
                break; \
            case 44: \
                *(SecStrBuf44 *)dest = *(const SecStrBuf44 *)src; \
                break; \
            case 45: \
                *(SecStrBuf45 *)dest = *(const SecStrBuf45 *)src; \
                break; \
            case 46: \
                *(SecStrBuf46 *)dest = *(const SecStrBuf46 *)src; \
                break; \
            case 47: \
                *(SecStrBuf47 *)dest = *(const SecStrBuf47 *)src; \
                break; \
            case 48: \
                *(SecStrBuf48 *)dest = *(const SecStrBuf48 *)src; \
                break; \
            case 49: \
                *(SecStrBuf49 *)dest = *(const SecStrBuf49 *)src; \
                break; \
            case 50: \
                *(SecStrBuf50 *)dest = *(const SecStrBuf50 *)src; \
                break; \
            case 51: \
                *(SecStrBuf51 *)dest = *(const SecStrBuf51 *)src; \
                break; \
            case 52: \
                *(SecStrBuf52 *)dest = *(const SecStrBuf52 *)src; \
                break; \
            case 53: \
                *(SecStrBuf53 *)dest = *(const SecStrBuf53 *)src; \
                break; \
            case 54: \
                *(SecStrBuf54 *)dest = *(const SecStrBuf54 *)src; \
                break; \
            case 55: \
                *(SecStrBuf55 *)dest = *(const SecStrBuf55 *)src; \
                break; \
            case 56: \
                *(SecStrBuf56 *)dest = *(const SecStrBuf56 *)src; \
                break; \
            case 57: \
                *(SecStrBuf57 *)dest = *(const SecStrBuf57 *)src; \
                break; \
            case 58: \
                *(SecStrBuf58 *)dest = *(const SecStrBuf58 *)src; \
                break; \
            case 59: \
                *(SecStrBuf59 *)dest = *(const SecStrBuf59 *)src; \
                break; \
            case 60: \
                *(SecStrBuf60 *)dest = *(const SecStrBuf60 *)src; \
                break; \
            case 61: \
                *(SecStrBuf61 *)dest = *(const SecStrBuf61 *)src; \
                break; \
            case 62: \
                *(SecStrBuf62 *)dest = *(const SecStrBuf62 *)src; \
                break; \
            case 63: \
                *(SecStrBuf63 *)dest = *(const SecStrBuf63 *)src; \
                break; \
            case 64: \
                *(SecStrBuf64 *)dest = *(const SecStrBuf64 *)src; \
                break; \
            default: \
                break; \
        } /* END switch */ \
    } else { \
        char *tmpDest = (char *)dest; \
        const char *tmpSrc = (const char *)src; \
        switch (count) { \
            case 64: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 63: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 62: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 61: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 60: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 59: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 58: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 57: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 56: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 55: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 54: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 53: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 52: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 51: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 50: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 49: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 48: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 47: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 46: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 45: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 44: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 43: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 42: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 41: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 40: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 39: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 38: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 37: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 36: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 35: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 34: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 33: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 32: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 31: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 30: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 29: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 28: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 27: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 26: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 25: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 24: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 23: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 22: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 21: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 20: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 19: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 18: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 17: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 16: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 15: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 14: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 13: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 12: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 11: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 10: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 9: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 8: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 7: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 6: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 5: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 4: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 3: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 2: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 1: \
                *(tmpDest++) = *(tmpSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            default: \
                break; \
        } \
    } \
} SECUREC_WHILE_ZERO
#endif

/*
 * Handling errors
 */
static errno_t SecMemcpyError(void *dest, size_t destMax, const void *src, size_t count)
{
    if (destMax == 0 || destMax > SECUREC_MEM_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("memcpy_s");
        return ERANGE;
    }
    if (dest == NULL || src == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("memcpy_s");
        if (dest != NULL) {
            (void)memset(dest, 0, destMax);
            return EINVAL_AND_RESET;
        }
        return EINVAL;
    }
    if (count > destMax) {
        (void)memset(dest, 0, destMax);
        SECUREC_ERROR_INVALID_RANGE("memcpy_s");
        return ERANGE_AND_RESET;
    }
    if (dest == src) {
        return EOK;
    }
    if ((dest > src && dest < (const void *)((const unsigned char *)src + count)) || \
        (src > dest && src < (void *)((unsigned char *)dest + count))) {
        (void)memset(dest, 0, destMax);
        SECUREC_ERROR_BUFFER_OVERLAP("memcpy_s");
        return EOVERLAP_AND_RESET;
    }
    /* count == 0 also return EOK */
    return EOK;
}

#if SECUREC_WITH_PERFORMANCE_ADDONS || SECUREC_MEMCOPY_WITH_PERFORMANCE
/*
 * Performance optimization
 */
static void SecDoMemcpyOpt(void *dest, const void *src, size_t count)
{
    if (count > SECUREC_MEMCOPY_THRESHOLD_SIZE) {
        SecDoMemcpy(dest, src, count);
    } else {
        SECUREC_SMALL_MEM_COPY;
    }
    return;
}
#endif

#if defined(SECUREC_COMPATIBLE_WIN_FORMAT)
    /* fread API in windows will call memcpy_s and pass 0xffffffff to destMax.
     * To avoid the failure of fread, we don't check desMax limit.
     */
#define SECUREC_MEMCPY_PARAM_OK(dest, destMax, src, count) (SECUREC_LIKELY((count) <= (destMax) && \
    (dest) != NULL && (src) != NULL && \
    (count) > 0 && SECUREC_MEMORY_NO_OVERLAP((dest), (src), (count))))
#else
#define SECUREC_MEMCPY_PARAM_OK(dest, destMax, src, count) (SECUREC_LIKELY((count) <= (destMax) && \
    (dest) != NULL && (src) != NULL && \
    (destMax) <= SECUREC_MEM_MAX_LEN && \
    (count) > 0 && SECUREC_MEMORY_NO_OVERLAP((dest), (src), (count))))
#endif

/*
 * <FUNCTION DESCRIPTION>
 *    The memcpy_s function copies n characters from the object pointed to by src into the object pointed to by dest
 *
 * <INPUT PARAMETERS>
 *    dest                      Destination buffer.
 *    destMax                   Size of the destination buffer.
 *    src                       Buffer to copy from.
 *    count                     Number of characters to copy
 *
 * <OUTPUT PARAMETERS>
 *    dest buffer               is updated.
 *
 * <RETURN VALUE>
 *    EOK                      Success
 *    EINVAL                   dest is  NULL and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN
 *    EINVAL_AND_RESET         dest != NULL and src is NULLL and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN
 *    ERANGE                   destMax > SECUREC_MEM_MAX_LEN or destMax is 0
 *    ERANGE_AND_RESET         count > destMax and destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN
 *                             and dest  !=  NULL  and src != NULL
 *    EOVERLAP_AND_RESET       dest buffer and source buffer are overlapped and
 *                             count <= destMax destMax != 0 and destMax <= SECUREC_MEM_MAX_LEN and dest  !=  NULL
 *                             and src != NULL  and dest != src
 *
 *    if an error occured, dest will be filled with 0.
 *    If the source and destination overlap, the behavior of memcpy_s is undefined.
 *    Use memmove_s to handle overlapping regions.
 */
errno_t memcpy_s(void *dest, size_t destMax, const void *src, size_t count)
{
    if (SECUREC_MEMCPY_PARAM_OK(dest, destMax, src, count)) {
#if SECUREC_MEMCOPY_WITH_PERFORMANCE
        SecDoMemcpyOpt(dest, src, count);
#else
        SecDoMemcpy(dest, src, count);
#endif
        return EOK;
    }
    /* meet some runtime violation, return error code */
    return SecMemcpyError(dest, destMax, src, count);
}

#if SECUREC_IN_KERNEL
EXPORT_SYMBOL(memcpy_s);
#endif

#if SECUREC_WITH_PERFORMANCE_ADDONS
/*
 * Performance optimization
 */
errno_t memcpy_sOptAsm(void *dest, size_t destMax, const void *src, size_t count)
{
    if (SECUREC_MEMCPY_PARAM_OK(dest, destMax, src, count)) {
        SecDoMemcpyOpt(dest, src, count);
        return EOK;
    }
    /* meet some runtime violation, return error code */
    return SecMemcpyError(dest, destMax, src, count);
}

/* trim judgement on "destMax <= SECUREC_MEM_MAX_LEN" */
errno_t memcpy_sOptTc(void *dest, size_t destMax, const void *src, size_t count)
{
    if (SECUREC_LIKELY(count <= destMax && dest != NULL && src != NULL && \
                       count > 0 && \
                       ((dest > src && (const void *)((const unsigned char *)src + count) <= dest) || \
                       (src > dest && (void *)((unsigned char *)dest + count) <= src)))) {
        SecDoMemcpyOpt(dest, src, count);
        return EOK;
    }
    /* meet some runtime violation, return error code */
    return SecMemcpyError(dest, destMax, src, count);
}
#endif

