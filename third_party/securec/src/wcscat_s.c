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

#define SECUREC_INLINE_DO_MEMCPY 1

#include "securecutil.h"

/*
 * Befor this function, the basic parameter checking has been done
 */
static errno_t SecDoWcscat(wchar_t *strDest, size_t destMax, const wchar_t *strSrc)
{
    size_t destLen;
    size_t srcLen;
    size_t maxCount; /* Store the maximum available count */

    /* To calculate the length of a wide character, the parameter must be a wide character */
    SECUREC_CALC_WSTR_LEN(strDest, destMax, &destLen);
    maxCount = destMax - destLen;
    SECUREC_CALC_WSTR_LEN(strSrc, maxCount, &srcLen);

    if (SECUREC_CAT_STRING_IS_OVERLAP(strDest, destLen, strSrc, srcLen)) {
        strDest[0] = L'\0';
        if (strDest + destLen <= strSrc && destLen == destMax) {
            SECUREC_ERROR_INVALID_PARAMTER("wcscat_s");
            return EINVAL_AND_RESET;
        }
        SECUREC_ERROR_BUFFER_OVERLAP("wcscat_s");
        return EOVERLAP_AND_RESET;
    }
    if (srcLen + destLen >= destMax || strDest == strSrc) {
        strDest[0] = L'\0';
        if (destLen == destMax) {
            SECUREC_ERROR_INVALID_PARAMTER("wcscat_s");
            return EINVAL_AND_RESET;
        }
        SECUREC_ERROR_INVALID_RANGE("wcscat_s");
        return ERANGE_AND_RESET;
    }
    SecDoMemcpy(strDest + destLen, strSrc, (srcLen + 1) * sizeof(wchar_t)); /* single character length  include \0 */
    return EOK;
}

/*
 * <FUNCTION DESCRIPTION>
 *    The wcscat_s function appends a copy of the wide string pointed to by strSrc
*      (including the terminating null wide character)
 *     to the end of the wide string pointed to by strDest.
 *    The arguments and return value of wcscat_s are wide-character strings.
 *
 *    The wcscat_s function appends strSrc to strDest and terminates the resulting
 *    string with a null character. The initial character of strSrc overwrites the
 *    terminating null character of strDest. wcscat_s will return EOVERLAP_AND_RESET if the
 *    source and destination strings overlap.
 *
 *    Note that the second parameter is the total size of the buffer, not the
 *    remaining size.
 *
 * <INPUT PARAMETERS>
 *    strDest              Null-terminated destination string buffer.
 *    destMax              Size of the destination string buffer.
 *    strSrc               Null-terminated source string buffer.
 *
 * <OUTPUT PARAMETERS>
 *    strDest               is updated
 *
 * <RETURN VALUE>
 *    EOK                   Success
 *    EINVAL                strDest is  NULL and destMax != 0 and destMax <= SECUREC_WCHAR_STRING_MAX_LEN
 *    EINVAL_AND_RESET      (strDest unterminated and all other parameters are valid) or
 *                          (strDest != NULL and strSrc is NULLL and destMax != 0
 *                           and destMax <= SECUREC_WCHAR_STRING_MAX_LEN)
 *    ERANGE                destMax > SECUREC_WCHAR_STRING_MAX_LEN or destMax is 0
 *    ERANGE_AND_RESET      strDest have not enough space  and all other parameters are valid  and not overlap
 *    EOVERLAP_AND_RESET     dest buffer and source buffer are overlapped and all  parameters are valid
 *
 *    If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
errno_t wcscat_s(wchar_t *strDest, size_t destMax, const wchar_t *strSrc)
{
    if (destMax == 0 || destMax > SECUREC_WCHAR_STRING_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("wcscat_s");
        return ERANGE;
    }

    if (strDest == NULL || strSrc == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("wcscat_s");
        if (strDest != NULL) {
            strDest[0] = L'\0';
            return EINVAL_AND_RESET;
        }
        return EINVAL;
    }

    return SecDoWcscat(strDest, destMax, strSrc);
}


