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

#include "securecutil.h"

static void SecTrimCRLF(char *buffer, size_t len)
{
    int i;
    /* No need to determine whether integer overflow exists */
    for (i = (int)(len - 1); i >= 0 && (buffer[i] == '\r' || buffer[i] == '\n'); --i) {
        buffer[i] = '\0';
    }
    return;
}

/*
 * <FUNCTION DESCRIPTION>
 *    The gets_s function reads at most one less than the number of characters
 *    specified by destMax from the stream pointed to by stdin, into the array pointed to by buffer
 *    The line consists of all characters up to and including
 *    the first newline character ('\n'). gets_s then replaces the newline
 *    character with a null character ('\0') before returning the line.
 *    If the first character read is the end-of-file character, a null character
 *    is stored at the beginning of buffer and NULL is returned.
 *
 * <INPUT PARAMETERS>
 *    buffer                         Storage location for input string.
 *    numberOfElements       The size of the buffer.
 *
 * <OUTPUT PARAMETERS>
 *    buffer                         is updated
 *
 * <RETURN VALUE>
 *    buffer                         Successful operation
 *    NULL                           Improper parameter or read fail
 */
char *gets_s(char *buffer, size_t numberOfElements)
{
    size_t len;
#ifdef SECUREC_COMPATIBLE_WIN_FORMAT
    size_t bufferSize = ((numberOfElements == (size_t)-1) ? SECUREC_STRING_MAX_LEN : numberOfElements);
#else
    size_t bufferSize = numberOfElements;
#endif

    if (buffer == NULL || bufferSize == 0 || bufferSize > SECUREC_STRING_MAX_LEN) {
        SECUREC_ERROR_INVALID_PARAMTER("gets_s");
        return NULL;
    }

    if (fgets(buffer, (int)bufferSize, stdin) == NULL) {
        return NULL;
    }

    len = strlen(buffer);
    if (len > 0 && len < bufferSize) {
        SecTrimCRLF(buffer, len);
    }

    return buffer;
}

