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

#include "securec.h"

/*
 * <FUNCTION DESCRIPTION>
 *    The sprintf_s function is equivalent to the sprintf function
 *    except for the parameter destMax and the explicit runtime-constraints violation
 *    The sprintf_s function formats and stores a series of characters and values
 *    in strDest. Each argument (if any) is converted and output according to
 *    the corresponding format specification in format. The format consists of
 *    ordinary characters and has the same form and function as the format argument
 *    for printf. A null character is appended after the last character written.
 *    If copying occurs between strings that overlap, the behavior is undefined.
 *
 * <INPUT PARAMETERS>
 *    strDest                 Storage location for output.
 *    destMax                Maximum number of characters to store.
 *    format                  Format-control string.
 *    ...                        Optional arguments
 *
 * <OUTPUT PARAMETERS>
 *    strDest                 is updated
 *
 * <RETURN VALUE>
 *    return the number of bytes stored in strDest, not counting the terminating null character.
 *    return -1 if an error occurred.
 *
 * If there is a runtime-constraint violation, strDest[0] will be set to the '\0' when strDest and destMax valid
 */
int sprintf_s(char *strDest, size_t destMax, const char *format, ...)
{
    int ret;                    /* If initialization causes  e838 */
    va_list argList;

    va_start(argList, format);
    ret = vsprintf_s(strDest, destMax, format, argList);
    va_end(argList);
    (void)argList;              /* to clear e438 last value assigned not used , the compiler will optimize this code */

    return ret;
}
#if SECUREC_IN_KERNEL
EXPORT_SYMBOL(sprintf_s);
#endif


