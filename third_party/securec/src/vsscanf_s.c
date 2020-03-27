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

#include "secinput.h"
#if defined(SECUREC_VXWORKS_PLATFORM) && (!defined(SECUREC_SYSAPI4VXWORKS) && !defined(SECUREC_CTYPE_MACRO_ADAPT))
#include <ctype.h>
#endif

/*
 * <NAME>
 *    vsscanf_s
 *
 *
 * <FUNCTION DESCRIPTION>
 *    The vsscanf_s function is equivalent to sscanf_s, with the variable argument list replaced by argList
 *    The vsscanf_s function reads data from buffer into the location given by
 *    each argument. Every argument must be a pointer to a variable with a type
 *    that corresponds to a type specifier in format. The format argument controls
 *    the interpretation of the input fields and has the same form and function
 *    as the format argument for the scanf function.
 *    If copying takes place between strings that overlap, the behavior is undefined.
 *
 * <INPUT PARAMETERS>
 *    buffer                Stored data
 *    format                Format control string, see Format Specifications.
 *    argList               pointer to list of arguments
 *
 * <OUTPUT PARAMETERS>
 *    argList               the converted value stored in user assigned address
 *
 * <RETURN VALUE>
 *    Each of these functions returns the number of fields successfully converted
 *    and assigned; the return value does not include fields that were read but
 *    not assigned. A return value of 0 indicates that no fields were assigned.
 *    return -1 if an error occurs.
 */
int vsscanf_s(const char *buffer, const char *format, va_list argList)
{
    size_t count;               /* If initialization causes  e838 */
    int retVal;
    SecFileStream fStr;

    /* validation section */
    if (buffer == NULL || format == NULL) {
        SECUREC_ERROR_INVALID_PARAMTER("vsscanf_s");
        return SECUREC_SCANF_EINVAL;
    }
    count = strlen(buffer);
    if (count == 0 || count > SECUREC_STRING_MAX_LEN) {
        SecClearDestBuf(buffer, format, argList);
        SECUREC_ERROR_INVALID_PARAMTER("vsscanf_s");
        return SECUREC_SCANF_EINVAL;
    }
#ifdef SECUREC_VXWORKS_PLATFORM
    /*
     * in vxworks platform when buffer is white string, will set first %s argument tu zero.like following useage:
     * "   \v\f\t\r\n", "%s", str, strSize
     * do not check all character, just first and last character then consider it is white string
     */
    if (isspace((int)buffer[0]) && isspace((int)buffer[count - 1])) {
        SecClearDestBuf(buffer, format, argList);
    }
#endif
    SECUREC_INIT_SEC_FILE_STREAM(fStr, SECUREC_MEM_STR_FLAG, NULL, 0, buffer, (int)count);
    retVal = SecInputS(&fStr, format, argList);
    if (retVal < 0) {
        SECUREC_ERROR_INVALID_PARAMTER("vsscanf_s");
        return SECUREC_SCANF_EINVAL;
    }
    return retVal;
}
#if SECUREC_IN_KERNEL
EXPORT_SYMBOL(vsscanf_s);
#endif

