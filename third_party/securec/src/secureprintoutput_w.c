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

/* if some platforms don't have wchar.h, dont't include it */
#if !(defined(SECUREC_VXWORKS_PLATFORM))
/* This header file is placed below secinput.h, which will cause tool alarm,
 * but if there is no macro above, it will cause compiling alarm
 */
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#ifndef _CRTIMP_ALTERNATIVE
#define _CRTIMP_ALTERNATIVE     /* comment microsoft *_s function */
#endif
#ifndef __STDC_WANT_SECURE_LIB__
#define __STDC_WANT_SECURE_LIB__ 0
#endif
#endif
#include <wchar.h>
#endif

#define SECUREC_ENABLE_WCHAR_FUNC   0
#define SECUREC_INLINE_DO_MEMCPY    1
#define SECUREC_FORMAT_OUTPUT_INPUT 1
#ifndef SECUREC_FOR_WCHAR
#define SECUREC_FOR_WCHAR
#endif

#include "secureprintoutput.h"

#ifndef WEOF
#define WEOF ((wchar_t)(-1))
#endif

#define SECUREC_CHAR(x) L ## x
#define SECUREC_WRITE_MULTI_CHAR SecWriteMultiCharW
#define SECUREC_WRITE_STRING     SecWriteStringW

static void SecWriteCharW(wchar_t ch, SecPrintfStream *f, int *pnumwritten);
static void SecWriteMultiCharW(wchar_t ch, int num, SecPrintfStream *f, int *pnumwritten);
static void SecWriteStringW(const wchar_t *string, int len, SecPrintfStream *f, int *pnumwritten);
static int SecPutWcharStrEndingZero(SecPrintfStream *str, int zeroCount);


#include "output.inl"

/*
 * Wide character formatted output implementation
 */
int SecVswprintfImpl(wchar_t *string, size_t sizeInWchar, const wchar_t *format, va_list argList)
{
    SecPrintfStream str;
    int retVal; /* If initialization causes  e838 */

    str.cur = (char *)string;
    /* this count include \0 character, Must be greater than zero */
    str.count = (int)(sizeInWchar * sizeof(wchar_t));

    retVal = SecOutputSW(&str, format, argList);
    if ((retVal >= 0) && SecPutWcharStrEndingZero(&str, (int)sizeof(wchar_t))) {
        return (retVal);
    } else if (str.count < 0) {
        /* the buffer was too small; we return truncation */
        string[sizeInWchar - 1] = L'\0';
        return SECUREC_PRINTF_TRUNCATE;
    }
    string[0] = L'\0';
    return -1;
}

/*
 * Output one zero character zero into the SecPrintfStream structure
 */
static int SecPutZeroChar(SecPrintfStream *str)
{
    if (str->count > 0) {
        *(str->cur) = (char)('\0');
        str->count = str->count - 1;
        str->cur = str->cur + 1;
        return 0;
    }
    return -1;
}

/*
 * Output a wide character zero end into the SecPrintfStream structure
 */
static int SecPutWcharStrEndingZero(SecPrintfStream *str, int zeroCount)
{
    int succeed = 0;
    int i = 0;

    while (i < zeroCount && (SecPutZeroChar(str) == 0)) {
        ++i;
    }
    if (i == zeroCount) {
        succeed = 1;
    }
    return succeed;
}


/*
 * Output a wide character into the SecPrintfStream structure
 */
static wchar_t SecPutCharW(wchar_t ch, SecPrintfStream *f)
{
    wchar_t wcRet = 0;
    if (((f)->count -= (int)sizeof(wchar_t)) >= 0) {
        *(wchar_t *)(void *)(f->cur) = ch;
        f->cur += sizeof(wchar_t);
        wcRet = ch;
    } else {
        wcRet = (wchar_t)WEOF;
    }
    return wcRet;
}

/*
 * Output a wide character into the SecPrintfStream structure, returns the number of characters written
 */
static void SecWriteCharW(wchar_t ch, SecPrintfStream *f, int *pnumwritten)
{
    if (SecPutCharW(ch, f) == (wchar_t)WEOF) {
        *pnumwritten = -1;
    } else {
        *pnumwritten = *pnumwritten + 1;
    }
}

/*
 * Output multiple wide character into the SecPrintfStream structure,  returns the number of characters written
 */
static void SecWriteMultiCharW(wchar_t ch, int num, SecPrintfStream *f, int *pnumwritten)
{
    int count = num;
    while (count-- > 0) {
        SecWriteCharW(ch, f, pnumwritten);
        if (*pnumwritten == -1) {
            break;
        }
    }
}

/*
 * Output a wide string into the SecPrintfStream structure,  returns the number of characters written
 */
static void SecWriteStringW(const wchar_t *string, int len, SecPrintfStream *f, int *pnumwritten)
{
    const wchar_t *str = string;
    int count = len;
    while (count-- > 0) {
        SecWriteCharW(*str++, f, pnumwritten);
        if (*pnumwritten == -1) {
            break;
        }
    }
}

