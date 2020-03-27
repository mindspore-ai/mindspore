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

#define SECUREC_INLINE_STR_LEN     1
#define SECUREC_INLINE_STR_LEN_OPT 1
#define SECUREC_INLINE_DO_MEMCPY   1
#include "securecutil.h"

/*
 * Befor this function, the basic parameter checking has been done
 */
static errno_t SecDoStrcat(char *strDest, size_t destMax, const char *strSrc)
{
    size_t destLen = SecStrMinLen(strDest, destMax);
    /* Only optimize strSrc, do not apply this function to strDest */
    size_t srcLen = SecStrMinLenOpt(strSrc, destMax - destLen);

    if (SECUREC_CAT_STRING_IS_OVERLAP(strDest, destLen, strSrc, srcLen)) {
        strDest[0] = '\0';
        if (strDest + destLen <= strSrc && destLen == destMax) {
            SECUREC_ERROR_INVALID_PARAMTER("strcat_s");
            return EINVAL_AND_RESET;
        }
        SECUREC_ERROR_BUFFER_OVERLAP("strcat_s");
        return EOVERLAP_AND_RESET;
    }
    if (srcLen + destLen >= destMax || strDest == strSrc) {
        strDest[0] = '\0';
        if (destLen == destMax) {
            SECUREC_ERROR_INVALID_PARAMTER("strcat_s");
            return EINVAL_AND_RESET;
        }
        SECUREC_ERROR_INVALID_RANGE("strcat_s");
        return ERANGE_AND_RESET;
    }
    SecDoMemcpy(strDest + destLen, strSrc, srcLen + 1); /* single character length  include \0 */
    return EOK;
}

/*
 * <FUNCTION DESCRIPTION>
 *    The strcat_s function appends a copy of the string pointed to by strSrc (including the terminating null character)
 *    to the end of the  string pointed to by strDest.
 *    The initial character of strSrc overwrites the terminating null character of strDest.
 *    strcat_s will return EOVERLAP_AND_RESET if the source and destination strings overlap.
 *
 *    Note that the second parameter is the total size of the buffer, not the
 *    remaining size.
 *
 * <INPUT PARAMETERS>
 *    strDest             Null-terminated destination string buffer.
 *    destMax             Size of the destination string buffer.
 *    strSrc              Null-terminated source string buffer.
 *
 * <OUTPUT PARAMETERS>
 *    strDest             is updated
 *
 * <RETURN VALUE>
 *    EOK                 Success
 *    EINVAL              strDest is  NULL and destMax != 0 and destMax <= SECUREC_STRING_MAX_LEN
 *    EINVAL_AND_RESET    (strDest unterminated  and all other parameters are valid)or
 *                         (strDest != NULL and strSrc is NULL and destMax != 0 and destMax <= SECUREC_STRING_MAX_LEN)
 *    ERANGE              destMax is 0 and destMax > SECUREC_STRING_MAX_LEN
 *    ERANGE_AND_RESET      strDest have not enough space  and all other parameters are valid  and not overlap
 *    EOVERLAP_AND_RESET   dest buffer and source buffer are overlapped and all  parameters are valid
 *
 *    If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
errno_t strcat_s(char *strDest, size_t destMax, const char *strSrc)
{
    if (destMax == 0 || destMax > SECUREC_STRING_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("strcat_s");
        return ERANGE;
    }
    if (strDest == NULL || strSrc == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("strcat_s");
        if (strDest != NULL) {
            strDest[0] = '\0';
            return EINVAL_AND_RESET;
        }
        return EINVAL;
    }
    return SecDoStrcat(strDest, destMax, strSrc);
}

#if SECUREC_IN_KERNEL
EXPORT_SYMBOL(strcat_s);
#endif

