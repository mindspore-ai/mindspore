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

/*
 * Befor this function, the basic parameter checking has been done
 */
static errno_t SecDoStrncat(char *strDest, size_t destMax, const char *strSrc, size_t count)
{
    size_t destLen = SecStrMinLen(strDest, destMax);
    /* The strSrc is no longer optimized. The reason is that when count is small,
     * the efficiency of strnlen is higher than that of self realization.
     */
    size_t srcLen = SecStrMinLen(strSrc, count);

    if (SECUREC_CAT_STRING_IS_OVERLAP(strDest, destLen, strSrc, srcLen)) {
        strDest[0] = '\0';
        if (strDest + destLen <= strSrc && destLen == destMax) {
            SECUREC_ERROR_INVALID_PARAMTER("strncat_s");
            return EINVAL_AND_RESET;
        }
        SECUREC_ERROR_BUFFER_OVERLAP("strncat_s");
        return EOVERLAP_AND_RESET;
    }
    if (srcLen + destLen >= destMax || strDest == strSrc) {
        strDest[0] = '\0';
        if (destLen == destMax) {
            SECUREC_ERROR_INVALID_PARAMTER("strncat_s");
            return EINVAL_AND_RESET;
        }
        SECUREC_ERROR_INVALID_RANGE("strncat_s");
        return ERANGE_AND_RESET;
    }
    SecDoMemcpy(strDest + destLen, strSrc, srcLen);    /* no  terminator */
    *(strDest + destLen + srcLen) = '\0';
    return EOK;
}

/*
 * <FUNCTION DESCRIPTION>
 *    The strncat_s function appends not more than n successive  characters
 *    (not including the terminating null  character)
 *     from the array pointed to by strSrc to the end of the  string pointed to by strDest
 *    The strncat_s function try to append the first D characters of strSrc to
 *    the end of strDest, where D is the lesser of count and the length of strSrc.
 *    If appending those D characters will fit within strDest (whose size is given
 *    as destMax) and still leave room for a null terminator, then those characters
 *    are appended, starting at the original terminating null of strDest, and a
 *    new terminating null is appended; otherwise, strDest[0] is set to the null
 *    character.
 *
 * <INPUT PARAMETERS>
 *    strDest            Null-terminated destination string.
 *    destMax            Size of the destination buffer.
 *    strSrc             Null-terminated source string.
 *    count              Number of character to append, or truncate.
 *
 * <OUTPUT PARAMETERS>
 *    strDest            is updated
 *
 * <RETURN VALUE>
 *    EOK                Success
 *    EINVAL             strDest is  NULL and destMax != 0 and destMax <= SECUREC_STRING_MAX_LEN
 *    EINVAL_AND_RESET      (strDest unterminated  and all other parameters are valid)or
 *                        (strDest !=  NULL and strSrc is NULL and destMax != 0 and destMax <= SECUREC_STRING_MAX_LEN)
 *    ERANGE                         destMax is 0 and destMax > SECUREC_STRING_MAX_LEN
 *    ERANGE_AND_RESET      strDest have not enough space  and all other parameters are valid  and not overlap
 *    EOVERLAP_AND_RESET   dest buffer and source buffer are overlapped and all  parameters are valid
 *
 *    If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
errno_t strncat_s(char *strDest, size_t destMax, const char *strSrc, size_t count)
{
    if (destMax == 0 || destMax > SECUREC_STRING_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("strncat_s");
        return ERANGE;
    }

    if (strDest == NULL || strSrc == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("strncat_s");
        if (strDest != NULL) {
            strDest[0] = '\0';
            return EINVAL_AND_RESET;
        }
        return EINVAL;
    }
    if (count > SECUREC_STRING_MAX_LEN) {
#ifdef  SECUREC_COMPATIBLE_WIN_FORMAT
        if (count == (size_t)(-1)) {
            /* Windows internal functions may pass in -1 when calling this function */
            return SecDoStrncat(strDest, destMax, strSrc, destMax);
        }
#endif
        strDest[0] = '\0';
        SECUREC_ERROR_INVALID_RANGE("strncat_s");
        return ERANGE_AND_RESET;
    }
    return SecDoStrncat(strDest, destMax, strSrc, count);
}

#if SECUREC_IN_KERNEL
EXPORT_SYMBOL(strncat_s);
#endif

