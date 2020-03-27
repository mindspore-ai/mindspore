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

#define SECUREC_INLINE_STR_LEN   1
#define SECUREC_INLINE_DO_MEMCPY 1

#include "securecutil.h"

#if SECUREC_IN_KERNEL== 0
#ifndef SECUREC_STRCOPY_THRESHOLD_SIZE
#define SECUREC_STRCOPY_THRESHOLD_SIZE   32UL
#endif

/*
 * Determine whether the address is 8-byte aligned, use static to increase performance
 * return 0 is aligned
 */
static int SecIsAddrAligned8(const void *addr, const void *zeroAddr)
{
    return (int)(((size_t)((const char*)addr - (const char*)zeroAddr)) & 7); /* use 7 to check aligned 8 */
}

/* The purpose of converting to void is to clean up the alarm */
#define SECUREC_SMALL_STR_COPY do { \
    if (SECUREC_ADDR_ALIGNED_8(strDest) && SECUREC_ADDR_ALIGNED_8(strSrc)) { \
        /* use struct assignment */ \
        switch (srcStrLen) { \
            case 1: \
                *(SecStrBuf1 *)(void *)strDest = *(const SecStrBuf1 *)(const void *)strSrc; \
                break; \
            case 2: \
                *(SecStrBuf2 *)(void *)strDest = *(const SecStrBuf2 *)(const void *)strSrc; \
                break; \
            case 3: \
                *(SecStrBuf3 *)(void *)strDest = *(const SecStrBuf3 *)(const void *)strSrc; \
                break; \
            case 4: \
                *(SecStrBuf4 *)(void *)strDest = *(const SecStrBuf4 *)(const void *)strSrc; \
                break; \
            case 5: \
                *(SecStrBuf5 *)(void *)strDest = *(const SecStrBuf5 *)(const void *)strSrc; \
                break; \
            case 6: \
                *(SecStrBuf6 *)(void *)strDest = *(const SecStrBuf6 *)(const void *)strSrc; \
                break; \
            case 7: \
                *(SecStrBuf7 *)(void *)strDest = *(const SecStrBuf7 *)(const void *)strSrc; \
                break; \
            case 8: \
                *(SecStrBuf8 *)(void *)strDest = *(const SecStrBuf8 *)(const void *)strSrc; \
                break; \
            case 9: \
                *(SecStrBuf9 *)(void *)strDest = *(const SecStrBuf9 *)(const void *)strSrc; \
                break; \
            case 10: \
                *(SecStrBuf10 *)(void *)strDest = *(const SecStrBuf10 *)(const void *)strSrc; \
                break; \
            case 11: \
                *(SecStrBuf11 *)(void *)strDest = *(const SecStrBuf11 *)(const void *)strSrc; \
                break; \
            case 12: \
                *(SecStrBuf12 *)(void *)strDest = *(const SecStrBuf12 *)(const void *)strSrc; \
                break; \
            case 13: \
                *(SecStrBuf13 *)(void *)strDest = *(const SecStrBuf13 *)(const void *)strSrc; \
                break; \
            case 14: \
                *(SecStrBuf14 *)(void *)strDest = *(const SecStrBuf14 *)(const void *)strSrc; \
                break; \
            case 15: \
                *(SecStrBuf15 *)(void *)strDest = *(const SecStrBuf15 *)(const void *)strSrc; \
                break; \
            case 16: \
                *(SecStrBuf16 *)(void *)strDest = *(const SecStrBuf16 *)(const void *)strSrc; \
                break; \
            case 17: \
                *(SecStrBuf17 *)(void *)strDest = *(const SecStrBuf17 *)(const void *)strSrc; \
                break; \
            case 18: \
                *(SecStrBuf18 *)(void *)strDest = *(const SecStrBuf18 *)(const void *)strSrc; \
                break; \
            case 19: \
                *(SecStrBuf19 *)(void *)strDest = *(const SecStrBuf19 *)(const void *)strSrc; \
                break; \
            case 20: \
                *(SecStrBuf20 *)(void *)strDest = *(const SecStrBuf20 *)(const void *)strSrc; \
                break; \
            case 21: \
                *(SecStrBuf21 *)(void *)strDest = *(const SecStrBuf21 *)(const void *)strSrc; \
                break; \
            case 22: \
                *(SecStrBuf22 *)(void *)strDest = *(const SecStrBuf22 *)(const void *)strSrc; \
                break; \
            case 23: \
                *(SecStrBuf23 *)(void *)strDest = *(const SecStrBuf23 *)(const void *)strSrc; \
                break; \
            case 24: \
                *(SecStrBuf24 *)(void *)strDest = *(const SecStrBuf24 *)(const void *)strSrc; \
                break; \
            case 25: \
                *(SecStrBuf25 *)(void *)strDest = *(const SecStrBuf25 *)(const void *)strSrc; \
                break; \
            case 26: \
                *(SecStrBuf26 *)(void *)strDest = *(const SecStrBuf26 *)(const void *)strSrc; \
                break; \
            case 27: \
                *(SecStrBuf27 *)(void *)strDest = *(const SecStrBuf27 *)(const void *)strSrc; \
                break; \
            case 28: \
                *(SecStrBuf28 *)(void *)strDest = *(const SecStrBuf28 *)(const void *)strSrc; \
                break; \
            case 29: \
                *(SecStrBuf29 *)(void *)strDest = *(const SecStrBuf29 *)(const void *)strSrc; \
                break; \
            case 30: \
                *(SecStrBuf30 *)(void *)strDest = *(const SecStrBuf30 *)(const void *)strSrc; \
                break; \
            case 31: \
                *(SecStrBuf31 *)(void *)strDest = *(const SecStrBuf31 *)(const void *)strSrc; \
                break; \
            case 32: \
                *(SecStrBuf32 *)(void *)strDest = *(const SecStrBuf32 *)(const void *)strSrc; \
                break; \
            default: \
                break; \
        } /* END switch */ \
    } else { \
        char *tmpStrDest = (char *)strDest; \
        const char *tmpStrSrc = (const char *)strSrc; \
        switch (srcStrLen) { \
            case 32: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 31: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 30: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 29: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 28: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 27: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 26: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 25: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 24: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 23: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 22: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 21: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 20: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 19: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 18: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 17: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 16: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 15: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 14: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 13: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 12: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 11: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 10: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 9: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 8: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 7: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 6: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 5: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 4: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 3: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 2: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            case 1: \
                *(tmpStrDest++) = *(tmpStrSrc++); \
                /* fall-through */ /* FALLTHRU */ \
            default: \
                break; \
        } \
    } \
} SECUREC_WHILE_ZERO
#endif

/*
 * Check Src Range
 */
static errno_t CheckSrcRange(char *strDest, size_t destMax, const char *strSrc)
{
    size_t tmpDestMax = destMax;
    const char *tmpSrc = strSrc;
    /* use destMax as boundary checker and destMax must be greater than zero */
    while (*(tmpSrc) != '\0' && tmpDestMax > 0) {
        ++tmpSrc;
        --tmpDestMax;
    }
    if (tmpDestMax == 0) {
        strDest[0] = '\0';
        SECUREC_ERROR_INVALID_RANGE("strcpy_s");
        return ERANGE_AND_RESET;
    }
    return EOK;
}

/*
 * Handling errors
 */
errno_t strcpy_error(char *strDest, size_t destMax, const char *strSrc)
{
    if (destMax == 0 || destMax > SECUREC_STRING_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("strcpy_s");
        return ERANGE;
    } else if (strDest == NULL || strSrc == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("strcpy_s");
        if (strDest != NULL) {
            strDest[0] = '\0';
            return EINVAL_AND_RESET;
        }
        return EINVAL;
    }
    return CheckSrcRange(strDest, destMax, strSrc);
}

/*
 * Performance optimization. srcStrLen  include '\0'
 */
static void SecDoStrcpyOpt(char *strDest, const char *strSrc, size_t srcStrLen)
{
#if SECUREC_IN_KERNEL
    SecDoMemcpy(strDest, strSrc, srcStrLen);
#else
    if (srcStrLen > SECUREC_STRCOPY_THRESHOLD_SIZE) {
        SecDoMemcpy(strDest, strSrc, srcStrLen);
    } else {
        SECUREC_SMALL_STR_COPY;
    }
#endif
}

/*
 * <FUNCTION DESCRIPTION>
 *    The strcpy_s function copies the string pointed to  strSrc
 *          (including the terminating null character) into the array pointed to by strDest
 *    The destination string must be large enough to hold the source string,
 *    including the terminating null character. strcpy_s will return EOVERLAP_AND_RESET
 *    if the source and destination strings overlap.
 *
 * <INPUT PARAMETERS>
 *    strDest                          Location of destination string buffer
 *    destMax                        Size of the destination string buffer.
 *    strSrc                            Null-terminated source string buffer.
 *
 * <OUTPUT PARAMETERS>
 *    strDest                         is updated.
 *
 * <RETURN VALUE>
 *    EOK                               Success
 *    EINVAL                          strDest is  NULL and destMax != 0 and destMax <= SECUREC_STRING_MAX_LEN
 *    EINVAL_AND_RESET       strDest !=  NULL and strSrc is NULL and destMax != 0 and destMax <= SECUREC_STRING_MAX_LEN
 *    ERANGE                         destMax is 0 and destMax > SECUREC_STRING_MAX_LEN
 *    ERANGE_AND_RESET      strDest have not enough space  and all other parameters are valid  and not overlap
 *    EOVERLAP_AND_RESET   dest buffer and source buffer are overlapped and all  parameters are valid
 *
 *    If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
errno_t strcpy_s(char *strDest, size_t destMax, const char *strSrc)
{
    if ((destMax > 0 && destMax <= SECUREC_STRING_MAX_LEN && strDest != NULL && strSrc != NULL && strDest != strSrc)) {
        size_t srcStrLen = SecStrMinLen(strSrc, destMax) + 1; /* len  include \0 */
        if (srcStrLen <= destMax) {
            /* use mem overlap check include \0 */
            if (SECUREC_MEMORY_NO_OVERLAP(strDest, strSrc, srcStrLen)) {
                /* performance optimization srcStrLen include '\0' */
                SecDoStrcpyOpt(strDest, strSrc, srcStrLen);
                return EOK;
            } else {
                strDest[0] = '\0';
                SECUREC_ERROR_BUFFER_OVERLAP("strcpy_s");
                return EOVERLAP_AND_RESET;
            }
        }
    }
    return strcpy_error(strDest, destMax, strSrc);
}

#if SECUREC_IN_KERNEL
EXPORT_SYMBOL(strcpy_s);
#endif

