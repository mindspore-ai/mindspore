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

#define SECUREC_INLINE_DO_MEMCPY    1
#define SECUREC_FORMAT_OUTPUT_INPUT 1
#ifdef SECUREC_FOR_WCHAR
#undef SECUREC_FOR_WCHAR
#endif

#include "secureprintoutput.h"

#define SECUREC_CHAR(x) x
#define SECUREC_WRITE_MULTI_CHAR  SecWriteMultiChar
#define SECUREC_WRITE_STRING      SecWriteString

#ifndef EOF
#define EOF (-1)
#endif

/* put a char to output */
#define SECUREC_PUTC(c, outStream)    ((--(outStream)->count >= 0) ? \
    (int)((unsigned int)(unsigned char)(*((outStream)->cur++) = (char)(c)) & 0xff) : EOF)
/* to clear e835 */
#define SECUREC_PUTC_ZERO(outStream)    ((--(outStream)->count >= 0) ? \
    ((*((outStream)->cur++) = (char)('\0'))) : EOF)

static void SecWriteMultiChar(char ch, int num, SecPrintfStream *f, int *pnumwritten);
static void SecWriteString(const char *string, int len, SecPrintfStream *f, int *pnumwritten);

#include "output.inl"

/*
 * Wide character formatted output implementation
 */
int SecVsnprintfImpl(char *string, size_t count, const char *format, va_list argList)
{
    SecPrintfStream str;
    int retVal;

    str.count = (int)count; /* this count include \0 character, Must be greater than zero */
    str.cur = string;

    retVal = SecOutputS(&str, format, argList);
    if ((retVal >= 0) && (SECUREC_PUTC_ZERO(&str) != EOF)) {
        return retVal;
    } else if (str.count < 0) {
        /* the buffer was too small; we return truncation */
        string[count - 1] = '\0';
        return SECUREC_PRINTF_TRUNCATE;
    }
    string[0] = '\0'; /* empty the dest strDest */
    return -1;
}

/*
 * Sec write Wide character
 */
static void SecWriteMultiChar(char ch, int num, SecPrintfStream *f, int *pnumwritten)
{
    int count = num;
    while (count-- > 0) {
        if (SECUREC_PUTC(ch, f) == EOF) {
            *pnumwritten = -1;
            break;
        } else {
            *pnumwritten = *pnumwritten + 1;
        }
    }
}

/*
 * Sec write string function
 */
static void SecWriteString(const char *string, int len, SecPrintfStream *f, int *pnumwritten)
{
    const char *str = string;
    int count = len;
    while (count-- > 0) {
        if (SECUREC_PUTC(*str, f) == EOF) {
            *pnumwritten = -1;
            break;
        } else {
            *pnumwritten = *pnumwritten + 1;
            ++str;
        }
    }
}

