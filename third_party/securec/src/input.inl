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

#ifndef INPUT_INL_5D13A042_DC3F_4ED9_A8D1_882811274C27
#define INPUT_INL_5D13A042_DC3F_4ED9_A8D1_882811274C27

#if SECUREC_IN_KERNEL
#include <linux/ctype.h>
#ifndef EOF
#define EOF  (-1)
#endif
#else
#if !defined(SECUREC_SYSAPI4VXWORKS) && !defined(SECUREC_CTYPE_MACRO_ADAPT)
#include <ctype.h>
#ifdef SECUREC_FOR_WCHAR
#include <wctype.h>             /* for iswspace */
#endif
#endif
#endif

#define SECUREC_NUM_WIDTH_SHORT                 0
#define SECUREC_NUM_WIDTH_INT                   1
#define SECUREC_NUM_WIDTH_LONG                  2
#define SECUREC_NUM_WIDTH_LONG_LONG             3 /* also long double */

#define SECUREC_BUF_EXT_MUL                     2
#define SECUREC_BUFFERED_BLOK_SIZE              1024

#if defined(SECUREC_VXWORKS_PLATFORM) && !defined(va_copy) && !defined(__va_copy)
/* the name is the same as system macro. */
#define __va_copy(d, s) do { \
    size_t size_of_d = (size_t)sizeof(d); \
    size_t size_of_s = (size_t)sizeof(s); \
    if (size_of_d != size_of_s) { \
        (void)memcpy((d), (s), sizeof(va_list)); \
    } else { \
        (void)memcpy(&(d), &(s), sizeof(va_list)); \
    } \
} SECUREC_WHILE_ZERO
#endif


#define SECUREC_MULTI_BYTE_MAX_LEN              6
/* Record a flag for each bit */
#define SECUREC_BRACKET_INDEX(x)                ((unsigned int)(x) >> 3)
#define SECUREC_BRACKET_VALUE(x)                ((unsigned char)(1 << ((unsigned int)(x) & 7)))


/* Compatibility macro name cannot be modifie */
#ifndef UNALIGNED
#if !(defined(_M_IA64)) && !(defined(_M_AMD64))
#define UNALIGNED
#else
#define UNALIGNED __unaligned
#endif
#endif

#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
/* Max 64bit value is 0xffffffffffffffff */
#define SECUREC_MAX_64BITS_VALUE                18446744073709551615ULL
#define SECUREC_MAX_64BITS_VALUE_DIV_TEN        1844674407370955161ULL
#define SECUREC_MAX_64BITS_VALUE_CUT_LAST_DIGIT 18446744073709551610ULL
#define SECUREC_MIN_64BITS_NEG_VALUE            9223372036854775808ULL
#define SECUREC_MAX_64BITS_POS_VALUE            9223372036854775807ULL
#define SECUREC_MIN_32BITS_NEG_VALUE            2147483648ULL
#define SECUREC_MAX_32BITS_POS_VALUE            2147483647ULL
#define SECUREC_MAX_32BITS_VALUE                4294967295ULL
#define SECUREC_MAX_32BITS_VALUE_INC            4294967296ULL
#define SECUREC_MAX_32BITS_VALUE_DIV_TEN        429496729ULL
#define SECUREC_LONG_BIT_NUM                    ((unsigned int)(sizeof(long) << 3U))

#define SECUREC_LONG_HEX_BEYOND_MAX(number)     (((number) >> (SECUREC_LONG_BIT_NUM - 4U)) > 0)
#define SECUREC_LONG_OCTAL_BEYOND_MAX(number)   (((number) >> (SECUREC_LONG_BIT_NUM - 3U)) > 0)

#define SECUREC_QWORD_HEX_BEYOND_MAX(number)    (((number) >> (64U - 4U)) > 0)
#define SECUREC_QWORD_OCTAL_BEYOND_MAX(number)  (((number) >> (64U - 3U)) > 0)

#define SECUREC_LP64_BIT_WIDTH                  64
#define SECUREC_LP32_BIT_WIDTH                  32

#endif

#define SECUREC_CHAR(x)                         (x)
#define SECUREC_BRACE                           '{'     /* [ to { */

#ifdef SECUREC_FOR_WCHAR
#define SECUREC_SCANF_BRACKET_CONDITION(comChr, ch, table, mask) ((comChr) == SECUREC_BRACE && \
    (table) != NULL && \
    (((table)[((unsigned int)(int)(ch) & SECUREC_CHAR_MASK) >> 3] ^ (mask)) & \
    (1 << ((unsigned int)(int)(ch) & 7))))
#else
#define SECUREC_SCANF_BRACKET_CONDITION(comChr, ch, table, mask) ((comChr) == SECUREC_BRACE && \
    (((table)[((unsigned char)(ch) & 0xff) >> 3] ^ (mask)) & (1 << ((unsigned char)(ch) & 7))))
#endif
#define SECUREC_SCANF_STRING_CONDITION(comChr, ch) ((comChr) == SECUREC_CHAR('s') && \
    (!((ch) >= SECUREC_CHAR('\t') && (ch) <= SECUREC_CHAR('\r')) && (ch) != SECUREC_CHAR(' ')))

/* Do not use   |=   optimize this code, it will cause compiling warning */
/* only supports  wide characters with a maximum length of two bytes */
#define SECUREC_BRACKET_SET_BIT(table, ch) do { \
    unsigned int tableIndex = SECUREC_BRACKET_INDEX(((unsigned int)(int)(ch) & SECUREC_CHAR_MASK)); \
    unsigned int tableValue = SECUREC_BRACKET_VALUE(((unsigned int)(int)(ch) & SECUREC_CHAR_MASK)); \
    (table)[tableIndex] = (unsigned char)((table)[tableIndex] | tableValue); \
} SECUREC_WHILE_ZERO

#ifdef SECUREC_FOR_WCHAR
/* table size is 32 x 256 */
#define SECUREC_BRACKET_TABLE_SIZE    8192
#define SECUREC_EOF WEOF
#define SECUREC_MB_LEN 16       /* max. # bytes in multibyte char  ,see MB_LEN_MAX */
/* int to unsigned int clear  e571 */
#define SECUREC_IS_DIGIT(chr)  (!((unsigned int)(int)(chr) & 0xff00) && isdigit(((unsigned int)(int)(chr) & 0x00ff)))
#define SECUREC_IS_XDIGIT(chr) (!((unsigned int)(int)(chr) & 0xff00) && isxdigit(((unsigned int)(int)(chr) & 0x00ff)))
#define SECUREC_IS_SPACE(chr)    iswspace((wint_t)(int)(chr))
#else
#define SECUREC_BRACKET_TABLE_SIZE    32
#define SECUREC_EOF EOF
#define SECUREC_IS_DIGIT(chr)    isdigit((unsigned char)(chr) & 0x00ff)
#define SECUREC_IS_XDIGIT(chr)   isxdigit((unsigned char)(chr) & 0x00ff)
#define SECUREC_IS_SPACE(chr)    isspace((unsigned char)(chr) & 0x00ff)
#endif


static SecInt SecSkipSpaceChar(SecFileStream *stream, int *counter);
static SecInt SecGetChar(SecFileStream *stream, int *counter);
static void SecUnGetChar(SecInt ch, SecFileStream *stream, int *counter);

typedef struct {
#ifdef SECUREC_FOR_WCHAR
    unsigned char *table; /* default NULL */
#else
    unsigned char table[SECUREC_BRACKET_TABLE_SIZE]; /* Array length is large enough in application scenarios */
#endif
    unsigned char mask; /* default 0 */
} SecBracketTable;

#ifdef SECUREC_FOR_WCHAR
#define SECUREC_INIT_BRACKET_TABLE { NULL, 0 }
#else
#define SECUREC_INIT_BRACKET_TABLE { { 0 }, 0 }
#endif

#if SECUREC_ENABLE_SCANF_FLOAT
typedef struct {
    size_t floatStrSize;           /* tialization must be length of buffer in charater */
    size_t floatStrUsedLen;        /* store float string len */
    SecChar buffer[SECUREC_FLOAT_BUFSIZE + 1];
    SecChar *floatStr;            /* Initialization must point to buffer */
    SecChar *allocatedFloatStr;   /* Initialization must be NULL  to store alloced point */
} SecFloatSpec;
#endif

typedef struct {
    SecUnsignedInt64 number64;
    unsigned long number;
    int numberWidth;     /* 0 = SHORT, 1 = int, > 1  long or L_DOUBLE */
    int isInt64Arg;      /* 1 for 64-bit integer, 0 otherwise */
    int negative;        /* 0 is positive */
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    int beyondMax;       /* Non-zero means beyond */
#endif
    void *argPtr;        /* Variable parameter pointer */
    size_t arrayWidth;   /* length of pointer Variable parameter, in charaters */
    int width;           /* width number in format */
    int widthSet;        /* 0 is not set width in format */
    int comChr;          /* Lowercase format conversion characters */
    int oriComChr;       /* store number conversion */
    signed char isWChar; /* -1/0 not wchar, 1 for wchar */
    char suppress;       /* 0 is not have %* in format */
} SecScanSpec;

#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
#define SECUREC_INIT_NUMBER_SPEC { 0, 0, 0, 0, 0, 0, NULL, 0, 0, 0, 0, 0, 0 }
#else
#define SECUREC_INIT_NUMBER_SPEC { 0, 0, 0, 0, 0, 0, NULL, 0, 0, 0, 0, 0 }
#endif

#ifdef SECUREC_FOR_WCHAR
#define SECUREC_GETC fgetwc
#define SECUREC_UN_GETC ungetwc
#define SECUREC_CHAR_MASK 0xffff
#else
#define SECUREC_GETC fgetc
#define SECUREC_UN_GETC ungetc
#define SECUREC_CHAR_MASK 0xff
#endif

/*
 * Determine if it is a 64-bit pointer  function
 * return 0 is not ,1 is 64bit pointer
 */
static int SecIs64BitPtr(size_t sizeOfVoidStar)
{
    /* point size is 4 or 8 , Under the 64 bit system, the value not 0 */
    /* to clear e778 */
    if ((sizeOfVoidStar & sizeof(SecInt64)) != 0) {
        return 1;
    }
    return 0;
}

#if SECUREC_ENABLE_SCANF_FLOAT

/*
 * Convert a floating point string to a floating point number
 */
static void SecAssignFloat(const char *floatStr, int numberWidth, void *argPtr)
{
    char *endPtr = NULL;
    double d;
#if SECUREC_SUPPORT_STRTOLD
    if (numberWidth == SECUREC_NUM_WIDTH_LONG_LONG) {
        long double d2 = strtold(floatStr, &endPtr);
        *(long double UNALIGNED *)(argPtr) = d2;
        return;
    }
#endif
    d = strtod(floatStr, &endPtr);
    if (numberWidth > SECUREC_NUM_WIDTH_INT) {
        *(double UNALIGNED *)(argPtr) = (double)d;
    } else {
        *(float UNALIGNED *)(argPtr) = (float)d;
    }
}

#ifdef SECUREC_FOR_WCHAR
/*
 * Convert a floating point wchar string to a floating point number
 * Success  ret 0
 */
static int SecAssignFloatW(const SecFloatSpec *floatSpec, const  SecScanSpec *spec)
{
    /* convert float string */
    size_t mbsLen;
    size_t tempFloatStrLen = (size_t)(floatSpec->floatStrSize + 1) * sizeof(wchar_t);
    char *tempFloatStr = (char *)SECUREC_MALLOC(tempFloatStrLen);

    if (tempFloatStr == NULL) {
        return -1;
    }
    tempFloatStr[0] = '\0';
    SECUREC_MASK_MSVC_CRT_WARNING
    mbsLen = wcstombs(tempFloatStr, floatSpec->floatStr, tempFloatStrLen - 1);
    SECUREC_END_MASK_MSVC_CRT_WARNING
    if (mbsLen != (size_t)-1) {
        tempFloatStr[mbsLen] = '\0';
        SecAssignFloat(tempFloatStr, spec->numberWidth, spec->argPtr);
    } else {
        SECUREC_FREE(tempFloatStr);
        return -1;
    }
    SECUREC_FREE(tempFloatStr);
    return 0;
}
#endif
/*
 * Splice floating point string
 * return 0 OK
 */
static int SecUpdateFloatString(SecChar ch, SecFloatSpec *floatSpec)
{
    floatSpec->floatStr[floatSpec->floatStrUsedLen++] = ch;    /* ch must be '0' - '9' */
    if (floatSpec->floatStrUsedLen < floatSpec->floatStrSize) {
        return 0;
    }
    if (floatSpec->allocatedFloatStr == NULL) {
        /* add 1 to clear ZERO LENGTH ALLOCATIONS warning */
        size_t oriBufSize = floatSpec->floatStrSize* (SECUREC_BUF_EXT_MUL * sizeof(SecChar)) + 1;
        void *tmpPointer = (void *)SECUREC_MALLOC(oriBufSize);
        if (tmpPointer == NULL) {
            return -1;
        }
        if (memcpy_s(tmpPointer, oriBufSize, floatSpec->floatStr, floatSpec->floatStrSize * sizeof(SecChar)) != EOK) {
            SECUREC_FREE(tmpPointer);   /* This is a dead code, just to meet the coding requirements */
            return -1;
        }
        floatSpec->floatStr = (SecChar *) (tmpPointer);
        floatSpec->allocatedFloatStr = (SecChar *) (tmpPointer); /* use to clear free on stack warning */
        floatSpec->floatStrSize *= SECUREC_BUF_EXT_MUL; /* this is OK, oriBufSize plus 1 just clear warning */
        return 0;
    } else {
        /* LSD 2014.3.6 fix, replace realloc to malloc to avoid heap injection */
        size_t oriBufSize = floatSpec->floatStrSize * sizeof(SecChar);
        size_t nextSize = (oriBufSize * SECUREC_BUF_EXT_MUL) + 1; /* add 1 to clear satic check tool warning */
        /* Prevents integer overflow when calculating the wide character length.
         * The maximum length of SECUREC_MAX_WIDTH_LEN is enough
         */
        if (nextSize <= SECUREC_MAX_WIDTH_LEN) {
            void *tmpPointer = (void *)SECUREC_MALLOC(nextSize);
            if (tmpPointer == NULL) {
                return -1;
            }
            if (memcpy_s(tmpPointer, nextSize, floatSpec->floatStr, oriBufSize) != EOK) {
                SECUREC_FREE(tmpPointer);   /* This is a dead code, just to meet the coding requirements */
                return -1;
            }
            if (memset_s(floatSpec->floatStr, oriBufSize, 0, oriBufSize) != EOK) {
                SECUREC_FREE(tmpPointer);   /* This is a dead code, just to meet the coding requirements */
                return -1;
            }
            SECUREC_FREE(floatSpec->floatStr);

            floatSpec->floatStr = (SecChar *) (tmpPointer);
            floatSpec->allocatedFloatStr = (SecChar *) (tmpPointer);    /* use to clear free on stack warning */
            floatSpec->floatStrSize *= SECUREC_BUF_EXT_MUL; /* this is OK, oriBufSize plus 1 just clear warning */
            return 0;
        }
    }
    return -1;
}
#endif

#ifndef SECUREC_FOR_WCHAR
/* LSD only multi-bytes string need isleadbyte() function */
static int SecIsLeadByte(SecInt ch)
{
    unsigned int c = (unsigned int)ch;
#if !(defined(_MSC_VER) || defined(_INC_WCTYPE))
    return (int)(c & 0x80);
#else
    return (int)isleadbyte((int)(c & 0xff));
#endif
}
#endif

/*
 * Parsing whether it is a wide character
 */
static void SecUpdateWcharFlagByType(SecUnsignedChar ch, SecScanSpec *spec)
{
#if defined(SECUREC_FOR_WCHAR) && (defined(SECUREC_COMPATIBLE_WIN_FORMAT))
    signed char flagForUpperType = -1;
    signed char flagForLowerType = 1;
#else
    signed char flagForUpperType = 1;
    signed char flagForLowerType = -1;
#endif
    /* if no  l or h flag  */
    if (spec->isWChar == 0) {
        if ((ch == SECUREC_CHAR('C')) || (ch == SECUREC_CHAR('S'))) {
            spec->isWChar = flagForUpperType;
        } else {
            spec->isWChar = flagForLowerType;
        }
    }
    return;
}
/*
 * decode  %l %ll
 */
static void SecDecodeScanQualifierL(const SecUnsignedChar **format, SecScanSpec *spec)
{
    const SecUnsignedChar *fmt = *format;
    if (*(fmt + 1) == SECUREC_CHAR('l')) {
        spec->isInt64Arg = 1;
        spec->numberWidth = SECUREC_NUM_WIDTH_LONG_LONG;
        ++fmt;
    } else {
        spec->numberWidth = SECUREC_NUM_WIDTH_LONG;
#if defined(SECUREC_ON_64BITS) && !(defined(SECUREC_COMPATIBLE_WIN_FORMAT))
        /* on window 64 system sizeof long is 32bit */
        spec->isInt64Arg = 1;
#endif
        spec->isWChar = 1;
    }
    *format = fmt;
}

/*
 * decode  %I %I43 %I64 %Id %Ii %Io ...
 * set finishFlag to  1  finish Flag
 */
static void SecDecodeScanQualifierI(const SecUnsignedChar **format, SecScanSpec *spec, int *finishFlag)
{
    const SecUnsignedChar *fmt = *format;
    if ((*(fmt + 1) == SECUREC_CHAR('6')) &&
        (*(fmt + 2) == SECUREC_CHAR('4'))) { /* offset 2 for I64 */
        spec->isInt64Arg = 1;
        *format = *format + 2; /* add 2 to skip I64 point to '4' next loop will inc */
    } else if ((*(fmt + 1) == SECUREC_CHAR('3')) &&
                (*(fmt + 2) == SECUREC_CHAR('2'))) { /* offset 2 for I32 */
        *format = *format + 2; /* add 2 to skip I32 point to '2' next loop will inc */
    } else if ((*(fmt + 1) == SECUREC_CHAR('d')) ||
                (*(fmt + 1) == SECUREC_CHAR('i')) ||
                (*(fmt + 1) == SECUREC_CHAR('o')) ||
                (*(fmt + 1) == SECUREC_CHAR('x')) ||
                (*(fmt + 1) == SECUREC_CHAR('X'))) {
        spec->isInt64Arg = SecIs64BitPtr(sizeof(void *));
    } else {
        /* for %I */
        spec->isInt64Arg = SecIs64BitPtr(sizeof(void *));
        *finishFlag = 1;
    }
}

static int SecDecodeScanWidth(const SecUnsignedChar **format, SecScanSpec *spec)
{
    const SecUnsignedChar *fmt = *format;
    while (SECUREC_IS_DIGIT(*fmt)) {
        spec->widthSet = 1;
        if (SECUREC_MUL_TEN_ADD_BEYOND_MAX(spec->width)) {
            return -1;
        }
        spec->width = (int)SECUREC_MUL_TEN((unsigned int)spec->width) + (unsigned char)(*fmt - SECUREC_CHAR('0'));
        ++fmt;
    }
    *format = fmt;
    return 0;
}

/*
 * init default flags for each format
 */
static void SecSetDefaultScanSpec(SecScanSpec *spec)
{
    spec->number64 = 0;
    spec->number = 0;
    spec->numberWidth = SECUREC_NUM_WIDTH_INT;    /* 0 = SHORT, 1 = int, > 1  long or L_DOUBLE */
    spec->isInt64Arg = 0;                         /* 1 for 64-bit integer, 0 otherwise */
    spec->negative = 0;
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    spec->beyondMax = 0;
#endif
    spec->argPtr = NULL;
    spec->arrayWidth = 0;
    spec->width = 0;
    spec->widthSet = 0;
    spec->comChr = 0;
    spec->isWChar = 0;
    spec->suppress = 0;
}

/*
 * decode qualifier %I %L %h ...
 * set finishFlag to  1  finish Flag
 */
static void  SecDecodeScanQualifier(const SecUnsignedChar **format, SecScanSpec *spec, int *finishFlag)
{
    switch ((int)(unsigned char)(**(format))) {
        case SECUREC_CHAR('F'):    /* fall-through */ /* FALLTHRU */
        case SECUREC_CHAR('N'):
            break;
        case SECUREC_CHAR('h'):
            --spec->numberWidth;  /* h for SHORT , hh for CHAR */
            spec->isWChar = -1;
            break;
#ifdef SECUREC_COMPATIBLE_LINUX_FORMAT
        case SECUREC_CHAR('j'):
            spec->numberWidth = SECUREC_NUM_WIDTH_LONG_LONG;  /* intmax_t or uintmax_t */
            spec->isInt64Arg = 1;
            break;
        case SECUREC_CHAR('t'):    /* fall-through */ /* FALLTHRU */
#endif
        case SECUREC_CHAR('z'):
#ifdef SECUREC_ON_64BITS
            spec->numberWidth = SECUREC_NUM_WIDTH_LONG_LONG;
            spec->isInt64Arg = 1;
#else
            spec->numberWidth = SECUREC_NUM_WIDTH_LONG;
#endif
            break;
        case SECUREC_CHAR('L'):    /* long double */ /* fall-through */ /* FALLTHRU */
        case SECUREC_CHAR('q'):
            spec->numberWidth = SECUREC_NUM_WIDTH_LONG_LONG;
            spec->isInt64Arg = 1;
            break;
        case SECUREC_CHAR('l'):
            SecDecodeScanQualifierL(format, spec);
            break;
        case SECUREC_CHAR('w'):
            spec->isWChar = 1;
            break;
        case SECUREC_CHAR('*'):
            spec->suppress = 1;
            break;
        case SECUREC_CHAR('I'):
            SecDecodeScanQualifierI(format, spec, finishFlag);
            break;
        default:
            *finishFlag = 1;
            break;
    }

}
/*
 * decode width and qualifier in format
 */
static int SecDecodeScanFlag(const SecUnsignedChar **format, SecScanSpec *spec)
{
    const SecUnsignedChar *fmt = *format;
    int finishFlag = 0;

    do {
        ++fmt; /*  first skip % , next  seek fmt */
        /* may %*6d , so put it inside the loop */
        if (SecDecodeScanWidth(&fmt, spec) != 0) {
            return -1;
        }
        SecDecodeScanQualifier(&fmt, spec, &finishFlag);
    } while (finishFlag == 0);
    *format = fmt;
    return 0;
}





/*
 * Judging whether a zeroing buffer is needed according to different formats
 */
static int SecDecodeClearFormat(const SecUnsignedChar *format, int *comChr)
{
    const SecUnsignedChar *fmt = format;
    /* to lowercase */
    int ch = (unsigned char)(*fmt) | (SECUREC_CHAR('a') - SECUREC_CHAR('A'));
    if (!(ch == SECUREC_CHAR('c') || ch == SECUREC_CHAR('s') || ch == SECUREC_BRACE)) {
        return -1;     /* first argument is not a string type */
    }
    if (ch == SECUREC_BRACE) {
#if !(defined(SECUREC_COMPATIBLE_WIN_FORMAT))
        if (*fmt == SECUREC_CHAR('{')) {
            return -1;
        }
#endif
        ++fmt;
        if (*fmt == SECUREC_CHAR('^')) {
            ++fmt;
        }
        if (*fmt == SECUREC_CHAR(']')) {
            ++fmt;
        }
        while ((*fmt != SECUREC_CHAR('\0')) && (*fmt != SECUREC_CHAR(']'))) {
            ++fmt;
        }
        if (*fmt == SECUREC_CHAR('\0')) {
            return -1; /* trunc'd format string */
        }
    }
    *comChr = ch;
    return 0;
}

/*
 * add L'\0' for wchar string , add '\0' for char string
 */
static void SecAddEndingZero(void *ptr, const SecScanSpec *spec)
{
    *(char *)ptr = '\0';
    (void)spec; /* clear not use */
#if SECUREC_HAVE_WCHART
    if (spec->isWChar > 0) {
        *(wchar_t UNALIGNED *)ptr = L'\0';
    }
#endif
}

#ifdef SECUREC_FOR_WCHAR
/*
 *  Clean up the first %s %c buffer to zero for wchar version
 */
void SecClearDestBufW(const wchar_t *buffer, const wchar_t *format, va_list argList)
#else
/*
 *  Clean up the first %s %c buffer to zero for char version
 */
void SecClearDestBuf(const char *buffer, const char *format, va_list argList)
#endif
{

    va_list argListSave;        /* backup for argList value, this variable don't need initialized */
    SecScanSpec spec;
    int comChr = 0;
    const SecUnsignedChar *fmt = (const SecUnsignedChar *)format;
    if (fmt == NULL) {
        return;
    }

    /* find first % */
    while (*fmt != SECUREC_CHAR('\0') && *fmt != SECUREC_CHAR('%')) {
        ++fmt;
    }
    if (*fmt == SECUREC_CHAR('\0')) {
        return;
    }

    SecSetDefaultScanSpec(&spec);
    if (SecDecodeScanFlag(&fmt, &spec) != 0) {
        return;
    }

    /* update wchar flag for %S %C */
    SecUpdateWcharFlagByType(*fmt, &spec);

    if (spec.suppress != 0 || SecDecodeClearFormat(fmt, &comChr) != 0) {
        return;
    }

    if ((buffer != NULL) && (*buffer != SECUREC_CHAR('\0')) && (comChr != SECUREC_CHAR('s'))) {
        /* when buffer not empty just clear %s.
         * example call sscanf by  argment of (" \n", "%s", s, sizeof(s))
         */
        return;
    }
    (void)memset(&argListSave, 0, sizeof(va_list)); /* to clear e530 argListSave not initialized */
#if defined(va_copy)
    va_copy(argListSave, argList);
#elif defined(__va_copy)        /* for vxworks */
    __va_copy(argListSave, argList);
#else
    argListSave = argList;
#endif
    do {
        void *argPtr = (void *)va_arg(argListSave, void *);
        /* Get the next argument - size of the array in characters */
        size_t arrayWidth = ((size_t)(va_arg(argListSave, size_t))) & 0xFFFFFFFFUL;
        va_end(argListSave);
        /* to clear e438 last value assigned not used , the compiler will optimize this code */
        (void)argListSave;
        /* There is no need to judge the upper limit */
        if (arrayWidth == 0 || argPtr == NULL) {
            return;
        }

        /* clear one char */
        SecAddEndingZero(argPtr, &spec);
    } SECUREC_WHILE_ZERO;
    return;

}

/*
 *  Assign number  to output buffer
 */
static void SecAssignNumber(const SecScanSpec *spec)
{
    void *argPtr = spec->argPtr;
    if (spec->isInt64Arg != 0) {
#if defined(SECUREC_VXWORKS_PLATFORM)
#if defined(SECUREC_VXWORKS_PLATFORM_COMP)
        *(SecInt64 UNALIGNED *)argPtr = (SecInt64)(spec->number64);
#else
         /* take number64 as unsigned number unsigned to int clear Compile warning */
        *(SecInt64 UNALIGNED *)argPtr = *(SecUnsignedInt64 *)(&(spec->number64));
#endif
#else
        /* take number64 as unsigned number */
        *(SecInt64 UNALIGNED *)argPtr = (SecInt64)(spec->number64);
#endif
        return;
    }
    if (spec->numberWidth > SECUREC_NUM_WIDTH_INT) {
        /* take number as unsigned number */
        *(long UNALIGNED *)argPtr = (long)(spec->number);
    } else if (spec->numberWidth == SECUREC_NUM_WIDTH_INT) {
        *(int UNALIGNED *)argPtr = (int)(spec->number);
    } else if (spec->numberWidth == SECUREC_NUM_WIDTH_SHORT) {
        /* take number as unsigned number */
        *(short UNALIGNED *)argPtr = (short)(spec->number);
    } else {  /* < 0 for hh format modifier */
        /* take number as unsigned number */
        *(char UNALIGNED *)argPtr = (char)(spec->number);
    }
}

#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
/*
 *  Judge the long bit width
 */
static int SecIsLongBitEqual(int bitNum)
{
    return (unsigned int)bitNum == SECUREC_LONG_BIT_NUM;
}
#endif
/*
 * Convert hexadecimal characters to decimal value
 */
static int SecHexValueOfChar(SecInt ch)
{
    /* use isdigt Causing tool false alarms */
    return (int)((ch >= '0' && ch <= '9') ? ((unsigned char)ch - '0') :
            ((((unsigned char)ch | (unsigned char)('a' - 'A')) - ('a')) + 10)); /* Adding 10 is to hex value */
}



/*
 * Parse decimal character to integer for 32bit .
 */
static void SecDecodeNumberDecimal(SecInt ch, SecScanSpec *spec)
{
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    unsigned long decimalEdge = SECUREC_MAX_32BITS_VALUE_DIV_TEN;
#ifdef SECUREC_ON_64BITS
    if (SecIsLongBitEqual(SECUREC_LP64_BIT_WIDTH)) {
        decimalEdge = (unsigned long)SECUREC_MAX_64BITS_VALUE_DIV_TEN;
    }
#else
    if (SecIsLongBitEqual(SECUREC_LP32_BIT_WIDTH)) {
        decimalEdge = SECUREC_MAX_32BITS_VALUE_DIV_TEN;
    }
#endif
    if (spec->number > decimalEdge) {
        spec->beyondMax = 1;
    }
#endif
    spec->number = SECUREC_MUL_TEN(spec->number);
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    if (spec->number == SECUREC_MUL_TEN(decimalEdge)) {
        SecUnsignedInt64 number64As = (unsigned long)SECUREC_MAX_64BITS_VALUE - spec->number;
        if (number64As < (SecUnsignedInt64)((SecUnsignedInt)ch - SECUREC_CHAR('0'))) {
            spec->beyondMax = 1;
        }
    }
#endif
    spec->number += (unsigned long)((SecUnsignedInt)ch - SECUREC_CHAR('0'));

}


/*
 * Parse Hex character to integer for 32bit .
 */
static void SecDecodeNumberHex(SecInt ch, SecScanSpec *spec)
{
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    if (SECUREC_LONG_HEX_BEYOND_MAX(spec->number)) {
        spec->beyondMax = 1;
    }
#endif
    spec->number = SECUREC_MUL_SIXTEEN(spec->number);
    spec->number += (unsigned long)(unsigned int)SecHexValueOfChar(ch);
}


/*
 * Parse Octal character to integer for 32bit .
 */
static void SecDecodeNumberOctal(SecInt ch, SecScanSpec *spec)
{
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    if (SECUREC_LONG_OCTAL_BEYOND_MAX(spec->number)) {
        spec->beyondMax = 1;
    }
#endif
    spec->number = SECUREC_MUL_EIGHT(spec->number);
    spec->number += (unsigned long)((SecUnsignedInt)ch - SECUREC_CHAR('0'));
}


#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
/* Compatible with integer negative values other than int */
static void SecFinishNumberNegativeOther(int comChr, int numberWidth, SecScanSpec *spec)
{
    if ((comChr == SECUREC_CHAR('d')) || (comChr == SECUREC_CHAR('i'))) {
        if (spec->number > (unsigned long)(1ULL << (SECUREC_LONG_BIT_NUM - 1))) {
            spec->number = (unsigned long)(1ULL << (SECUREC_LONG_BIT_NUM - 1));
        } else {
            spec->number = (unsigned long)(-(long)spec->number);
        }
        if (spec->beyondMax != 0) {
            if (numberWidth < SECUREC_NUM_WIDTH_INT) {
                spec->number = 0;
            } else if (numberWidth == SECUREC_NUM_WIDTH_LONG) {
                spec->number = ((unsigned long)(1UL << (SECUREC_LONG_BIT_NUM - 1)));
            }
        }
    } else { /* o, u, x, X, p */
        spec->number = (unsigned long)(-(long)spec->number);
        if (spec->beyondMax != 0) {
            spec->number |= (unsigned long)SECUREC_MAX_64BITS_VALUE;
        }
    }
}
/* Compatible processing of integer negative numbers */
static void SecFinishNumberNegativeInt(int comChr, SecScanSpec *spec)
{
    if ((comChr == SECUREC_CHAR('d')) || (comChr == SECUREC_CHAR('i'))) {
#ifdef SECUREC_ON_64BITS
        if (SecIsLongBitEqual(SECUREC_LP64_BIT_WIDTH)) {
            if ((spec->number > SECUREC_MIN_64BITS_NEG_VALUE)) {
                spec->number = 0;
            } else {
                spec->number = (unsigned int)(-(int)spec->number);
            }
        }
#else
        if (SecIsLongBitEqual(SECUREC_LP32_BIT_WIDTH)) {
            if ((spec->number > SECUREC_MIN_32BITS_NEG_VALUE)) {
                spec->number = SECUREC_MIN_32BITS_NEG_VALUE;
            } else {
                spec->number = (unsigned int)(-(int)spec->number);
            }
        }
#endif
        if (spec->beyondMax != 0) {
#ifdef SECUREC_ON_64BITS
            if (SecIsLongBitEqual(SECUREC_LP64_BIT_WIDTH)) {
                spec->number = 0;
            }
#else
            if (SecIsLongBitEqual(SECUREC_LP32_BIT_WIDTH)) {
                spec->number = SECUREC_MIN_32BITS_NEG_VALUE;
            }
#endif
        }
    } else {            /* o, u, x, X ,p */
#ifdef SECUREC_ON_64BITS
        if (spec->number > SECUREC_MAX_32BITS_VALUE_INC) {
            spec->number = SECUREC_MAX_32BITS_VALUE;
        } else {
            spec->number = (unsigned int)(-(int)spec->number);
        }
#else
        spec->number = (unsigned int)(-(int)spec->number);
#endif
        if (spec->beyondMax != 0) {
            spec->number |= (unsigned long)SECUREC_MAX_64BITS_VALUE;
        }
    }
}

/* Compatible with integer positive values other than int */
static void SecFinishNumberPositiveOther(int comChr, int numberWidth, SecScanSpec *spec)
{
    if (comChr == SECUREC_CHAR('d') || comChr == SECUREC_CHAR('i')) {
        if (spec->number > ((unsigned long)(1UL << (SECUREC_LONG_BIT_NUM - 1)) - 1)) {
            spec->number = ((unsigned long)(1UL << (SECUREC_LONG_BIT_NUM - 1)) - 1);
        }
        if ((spec->beyondMax != 0 && numberWidth < SECUREC_NUM_WIDTH_INT)) {
            spec->number |= (unsigned long)SECUREC_MAX_64BITS_VALUE;
        }
        if (spec->beyondMax != 0 && numberWidth == SECUREC_NUM_WIDTH_LONG) {
            spec->number = ((unsigned long)(1UL << (SECUREC_LONG_BIT_NUM - 1)) - 1);
        }
    } else {
        if (spec->beyondMax != 0) {
            spec->number |= (unsigned long)SECUREC_MAX_64BITS_VALUE;
        }
    }
}

/* Compatible processing of integer positive numbers */
static void SecFinishNumberPositiveInt(int comChr, SecScanSpec *spec)
{
    if ((comChr == SECUREC_CHAR('d')) || (comChr == SECUREC_CHAR('i'))) {
#ifdef SECUREC_ON_64BITS
        if (SecIsLongBitEqual(SECUREC_LP64_BIT_WIDTH)) {
            if (spec->number > SECUREC_MAX_64BITS_POS_VALUE) {
                spec->number |= (unsigned long)SECUREC_MAX_64BITS_VALUE;
            }
        }
        if (spec->beyondMax != 0 && SecIsLongBitEqual(SECUREC_LP64_BIT_WIDTH)) {
            spec->number |= (unsigned long)SECUREC_MAX_64BITS_VALUE;
        }
#else
        if (SecIsLongBitEqual(SECUREC_LP32_BIT_WIDTH)) {
            if (spec->number > SECUREC_MAX_32BITS_POS_VALUE) {
                spec->number = SECUREC_MAX_32BITS_POS_VALUE;
            }
        }
        if (spec->beyondMax != 0 && SecIsLongBitEqual(SECUREC_LP32_BIT_WIDTH)) {
            spec->number = SECUREC_MAX_32BITS_POS_VALUE;
        }
#endif
    } else {            /* o,u,x,X,p */
        if (spec->beyondMax != 0) {
            spec->number = SECUREC_MAX_32BITS_VALUE;
        }
    }
}

#endif


/*
 * Parse decimal character to integer for 64bit .
 */
static void SecDecodeNumber64Decimal(SecInt ch, SecScanSpec *spec)
{
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    if (spec->number64 > SECUREC_MAX_64BITS_VALUE_DIV_TEN) {
        spec->beyondMax = 1;
    }
#endif
    spec->number64 = SECUREC_MUL_TEN(spec->number64);
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    if (spec->number64 == SECUREC_MAX_64BITS_VALUE_CUT_LAST_DIGIT) {
        SecUnsignedInt64 number64As = (SecUnsignedInt64)SECUREC_MAX_64BITS_VALUE - spec->number64;
        if (number64As < (SecUnsignedInt64)((SecUnsignedInt)ch - SECUREC_CHAR('0'))) {
            spec->beyondMax = 1;
        }
    }
#endif
    spec->number64 += (SecUnsignedInt64)((SecUnsignedInt)ch - SECUREC_CHAR('0'));
}

/*
 * Parse Hex character to integer for 64bit .
 */
static void SecDecodeNumber64Hex(SecInt ch, SecScanSpec *spec)
{
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    if (SECUREC_QWORD_HEX_BEYOND_MAX(spec->number64)) {
        spec->beyondMax = 1;
    }
#endif
    spec->number64 = SECUREC_MUL_SIXTEEN(spec->number64);
    spec->number64 += (SecUnsignedInt64)(unsigned int)SecHexValueOfChar(ch);

}

/*
 * Parse Octal character to integer for 64bit .
 */
static void SecDecodeNumber64Octal(SecInt ch, SecScanSpec *spec)
{
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    if (SECUREC_QWORD_OCTAL_BEYOND_MAX(spec->number64)) {
        spec->beyondMax = 1;
    }
#endif
    spec->number64 = SECUREC_MUL_EIGHT(spec->number64);
    spec->number64 += (SecUnsignedInt64)((SecUnsignedInt)ch - SECUREC_CHAR('0'));
}

#define SECUREC_DECODE_NUMBER_FUNC_NUM 2
/* Function name cannot add address symbol, causing 546 alarm */
static void (*g_secDecodeNumberHex[SECUREC_DECODE_NUMBER_FUNC_NUM])(SecInt ch, SecScanSpec *spec) = \
    { SecDecodeNumberHex, SecDecodeNumber64Hex };
static void (*g_secDecodeNumberOctal[SECUREC_DECODE_NUMBER_FUNC_NUM])(SecInt ch, SecScanSpec *spec) = \
    { SecDecodeNumberOctal, SecDecodeNumber64Octal };
static void (*g_secDecodeNumberDecimal[SECUREC_DECODE_NUMBER_FUNC_NUM])(SecInt ch, SecScanSpec *spec) = \
    { SecDecodeNumberDecimal, SecDecodeNumber64Decimal };

/*
 * Parse 64-bit integer formatted input, return 0 when ch is a number.
 */
static int SecDecodeNumber(SecInt ch, SecScanSpec *spec)
{
    if (spec->comChr == SECUREC_CHAR('x') || spec->comChr == SECUREC_CHAR('p')) {
        if (SECUREC_IS_XDIGIT(ch)) {
            (*g_secDecodeNumberHex[spec->isInt64Arg])(ch, spec);
        } else {
            return -1;
        }
        return 0;
    }
    if (!(SECUREC_IS_DIGIT(ch))) {
        return -1;
    }
    if (spec->comChr == SECUREC_CHAR('o')) {
        if (ch < SECUREC_CHAR('8')) {
            (*g_secDecodeNumberOctal[spec->isInt64Arg])(ch, spec);
        } else {
            return -1;
        }
    } else { /* comChr is 'd' */
        (*g_secDecodeNumberDecimal[spec->isInt64Arg])(ch, spec);
    }
    return 0;
}


/*
 * Complete the final 32-bit integer formatted input
 */
static void SecFinishNumber(SecScanSpec *spec)
{
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    if (spec->negative != 0) {
        if (spec->numberWidth == SECUREC_NUM_WIDTH_INT) {
            SecFinishNumberNegativeInt(spec->oriComChr, spec);
        } else {
            SecFinishNumberNegativeOther(spec->oriComChr, spec->numberWidth, spec);
        }
    } else {
        if (spec->numberWidth == SECUREC_NUM_WIDTH_INT) {
            SecFinishNumberPositiveInt(spec->oriComChr, spec);
        } else {
            SecFinishNumberPositiveOther(spec->oriComChr, spec->numberWidth, spec);
        }
    }
#else
    if (spec->negative != 0) {
#if defined(__hpux)
        if (spec->oriComChr != SECUREC_CHAR('p')) {
            spec->number = (unsigned long)(-(long)spec->number);
        }
#else
        spec->number = (unsigned long)(-(long)spec->number);
#endif
    }
#endif
    return;
}

/*
 * Complete the final 64-bit integer formatted input
 */
static void SecFinishNumber64(SecScanSpec *spec)
{
#if (defined(SECUREC_COMPATIBLE_LINUX_FORMAT) && !(defined(SECUREC_ON_UNIX)))
    if (spec->negative != 0) {
        if (spec->oriComChr == (SECUREC_CHAR('d')) || (spec->oriComChr == SECUREC_CHAR('i'))) {
            if (spec->number64 > SECUREC_MIN_64BITS_NEG_VALUE) {
                spec->number64 = SECUREC_MIN_64BITS_NEG_VALUE;
            } else {
                spec->number64 = (SecUnsignedInt64)(-(SecInt64)spec->number64);
            }
            if (spec->beyondMax != 0) {
                spec->number64 = SECUREC_MIN_64BITS_NEG_VALUE;
            }
        } else {                /* o, u, x, X, p */
            spec->number64 = (SecUnsignedInt64)(-(SecInt64)spec->number64);
            if (spec->beyondMax != 0) {
                spec->number64 = SECUREC_MAX_64BITS_VALUE;
            }
        }
    } else {
        if ((spec->oriComChr == SECUREC_CHAR('d')) || (spec->oriComChr == SECUREC_CHAR('i'))) {
            if (spec->number64 > SECUREC_MAX_64BITS_POS_VALUE) {
                spec->number64 = SECUREC_MAX_64BITS_POS_VALUE;
            }
            if (spec->beyondMax != 0) {
                spec->number64 = SECUREC_MAX_64BITS_POS_VALUE;
            }
        } else {
            if (spec->beyondMax != 0) {
                spec->number64 = SECUREC_MAX_64BITS_VALUE;
            }
        }
    }
#else
    if (spec->negative != 0) {
#if defined(__hpux)
        if (spec->oriComChr != SECUREC_CHAR('p')) {
            spec->number64 = (SecUnsignedInt64)(-(SecInt64)spec->number64);
        }
#else
        spec->number64 = (SecUnsignedInt64)(-(SecInt64)spec->number64);
#endif
    }
#endif
    return;
}
static void (*g_secFinishNumber[SECUREC_DECODE_NUMBER_FUNC_NUM])(SecScanSpec *spec) = \
    { SecFinishNumber, SecFinishNumber64 };

#if SECUREC_ENABLE_SCANF_FILE

/*
 *  Adjust the pointer position of the file stream
 */
static void SecSeekStream(SecFileStream *stream)
{
    if ((stream->count == 0) && feof(stream->pf)) {
        /* file pointer at the end of file, don't need to seek back */
        stream->base[0] = '\0';
        return;
    }
    /* LSD seek to original position, bug fix 2014 1 21 */
    if (fseek(stream->pf, stream->oriFilePos, SEEK_SET)) {
        /* seek failed, ignore it */
        stream->oriFilePos = 0;
        return;
    }

    if (stream->fileRealRead > 0) { /* LSD bug fix. when file reach to EOF, don't seek back */
#if (defined(SECUREC_COMPATIBLE_WIN_FORMAT))
        int loops;
        for (loops = 0; loops < (stream->fileRealRead / SECUREC_BUFFERED_BLOK_SIZE); ++loops) {
            if (fread(stream->base, (size_t)1, (size_t)SECUREC_BUFFERED_BLOK_SIZE,
                stream->pf) != SECUREC_BUFFERED_BLOK_SIZE) {
                break;
            }
        }
        if ((stream->fileRealRead % SECUREC_BUFFERED_BLOK_SIZE) != 0) {
            size_t ret = fread(stream->base, (size_t)((unsigned int)stream->fileRealRead % SECUREC_BUFFERED_BLOK_SIZE),
                               (size_t)1, stream->pf);
            if ((ret == 1 || ret == 0) && (ftell(stream->pf) < stream->oriFilePos + stream->fileRealRead)) {
                (void)fseek(stream->pf, stream->oriFilePos + stream->fileRealRead, SEEK_SET);
            }
        }

#else
        /* in linux like system */
        if (fseek(stream->pf, stream->oriFilePos + stream->fileRealRead, SEEK_SET)) {
            /* seek failed, ignore it */
            stream->oriFilePos = 0;
        }
#endif
    }

    return;
}

/*
 *  Adjust the pointer position of the file stream and free memory
 */
static void SecAdjustStream(SecFileStream *stream)
{
    if (stream != NULL && (stream->flag & SECUREC_FILE_STREAM_FLAG) && stream->base != NULL) {
        SecSeekStream(stream);
        SECUREC_FREE(stream->base);
        stream->base = NULL;
    }
    return;
}
#endif

static void SecSkipSpaceFormat(const SecUnsignedChar **format)
{
    const SecUnsignedChar *fmt = *format;
    while (SECUREC_IS_SPACE(*fmt)) {
        ++fmt;
    }
    *format = fmt;
}
#ifndef SECUREC_FOR_WCHAR
/*
 * Handling multi-character characters
 */
static int SecDecodeLeadByte(SecInt ch, const SecUnsignedChar **format, SecFileStream *stream, int *counter)
{
#if SECUREC_HAVE_MBTOWC
    char temp[SECUREC_MULTI_BYTE_MAX_LEN];
    const SecUnsignedChar *fmt = *format;
    wchar_t tempWChar = L'\0';
    int ch2 = SecGetChar(stream, counter);
    if (*fmt == SECUREC_CHAR('\0') || (int)(*fmt) != (ch2)) {
        /* LSD in console mode, ungetc twice may cause problem */
        SecUnGetChar(ch2, stream, counter);
        SecUnGetChar(ch, stream, counter);
        return -1;
    }
    ++fmt;
    if (MB_CUR_MAX >= SECUREC_UTF8_BOM_HEADER_SIZE &&
        (((unsigned char)ch & SECUREC_UTF8_LEAD_1ST) == SECUREC_UTF8_LEAD_1ST) &&
        (((unsigned char)ch2 & SECUREC_UTF8_LEAD_2ND) == SECUREC_UTF8_LEAD_2ND)) {
        /* this char is very likely to be a UTF-8 char */
        int ch3 = SecGetChar(stream, counter);
        temp[0] = (char)ch;
        temp[1] = (char)ch2; /* 1 index of second character */
        temp[2] = (char)ch3; /* 2 index of third character */
        temp[3] = '\0';      /* 3 of string terminator position */

        if (mbtowc(&tempWChar, temp, sizeof(temp)) > 0) {
            /* succeed */
            if (*fmt == SECUREC_CHAR('\0') || (int)(*fmt) != (int)ch3) {
                SecUnGetChar(ch3, stream, counter);
                return -1;
            }
            ++fmt;
            *counter = *counter - 1;
        } else {
            SecUnGetChar(ch3, stream, counter);
        }
    }
    *counter = *counter - 1;    /* only count as one character read */
    *format = fmt;
    return 0;
#else
    SecUnGetChar(ch, stream, counter);
    (void)format;
    return -1;
#endif
}
#endif



/*
 *  Resolving sequence of characters from %[ format
 */
static int SecSetupBracketTable(const SecUnsignedChar **format, SecBracketTable *bracketTable)
{
    const SecUnsignedChar *fmt = *format;
    SecUnsignedChar prevChar = 0;
    SecUnsignedChar expCh;
    SecUnsignedChar last = 0;
#if !(defined(SECUREC_COMPATIBLE_WIN_FORMAT))
    if (*fmt == SECUREC_CHAR('{')) {
        return -1;
    }
#endif
    /* for building "table" data */
    ++fmt; /* skip [ */
    bracketTable->mask = 0;
    if (*fmt == SECUREC_CHAR('^')) {
        ++fmt;
        bracketTable->mask = (unsigned char)0xff;
    }
    if (*fmt == SECUREC_CHAR(']')) {
        prevChar = SECUREC_CHAR(']');
        ++fmt;
        SECUREC_BRACKET_SET_BIT(bracketTable->table, SECUREC_CHAR(']'));
    }
    while (*fmt != SECUREC_CHAR('\0') && *fmt != SECUREC_CHAR(']')) {
        expCh = *fmt++;
        if (expCh != SECUREC_CHAR('-') || prevChar == 0 || *fmt == SECUREC_CHAR(']')) {
            /* normal character */
            prevChar = expCh;
            SECUREC_BRACKET_SET_BIT(bracketTable->table, expCh);
        } else {
            /* for %[a-z] */
            expCh = *fmt++;   /* get end of range */
            if (prevChar < expCh) { /* %[a-z] */
                last = expCh;
            } else {
                prevChar = expCh;
#if (defined(SECUREC_COMPATIBLE_WIN_FORMAT))
                /* %[z-a] */
                last = prevChar;

#else
                SECUREC_BRACKET_SET_BIT(bracketTable->table, SECUREC_CHAR('-'));
                SECUREC_BRACKET_SET_BIT(bracketTable->table, expCh);
                continue;
#endif
            }
            /* format %[a-\xff] last is 0xFF, condition (rnch <= last) cause dead loop */
            for (expCh = prevChar; expCh < last; ++expCh) {
                SECUREC_BRACKET_SET_BIT(bracketTable->table, expCh);
            }
            SECUREC_BRACKET_SET_BIT(bracketTable->table, last);
            prevChar = 0;
        }
    }
    *format = fmt;
    return 0;
}


#ifdef SECUREC_FOR_WCHAR
static int SecInputForWchar(SecInt ch, SecScanSpec *spec)
{
    void *endPtr = spec->argPtr;
    if (spec->isWChar > 0) {
        *(wchar_t UNALIGNED *)endPtr = (wchar_t)ch;
        endPtr = (wchar_t *)endPtr + 1;
        --spec->arrayWidth;
    } else {
#if SECUREC_HAVE_WCTOMB
        int temp;
        char tmpBuf[SECUREC_MB_LEN + 1];
        SECUREC_MASK_MSVC_CRT_WARNING temp = wctomb(tmpBuf, (wchar_t)ch);
        SECUREC_END_MASK_MSVC_CRT_WARNING
        if (temp <= 0 || ((size_t)(unsigned int)temp) > sizeof(tmpBuf)) {
            /* if wctomb  error, then ignore character */
            return 0;
        }
        if (((size_t)(unsigned int)temp) > spec->arrayWidth) {
            return -1;
        }
        if (memcpy_s(endPtr, spec->arrayWidth, tmpBuf, (size_t)(unsigned int)temp) != EOK) {
            return -1;
        }
        endPtr = (char *)endPtr + temp;
        spec->arrayWidth -= (size_t)(unsigned int)temp;
#else
        return -1;
#endif
    }
    spec->argPtr = endPtr;
    return 0;
}
#endif


#ifndef SECUREC_FOR_WCHAR
static int SecInputForChar(SecInt ch, SecScanSpec *spec, SecFileStream *stream, int *charCount)
{
    void *endPtr = spec->argPtr;
    if (spec->isWChar > 0) {
        wchar_t tempWChar = L'?';   /* set default char as ? */
#if SECUREC_HAVE_MBTOWC
        char temp[SECUREC_MULTI_BYTE_MAX_LEN + 1];
        temp[0] = (char)ch;
        temp[1] = '\0';
#if defined(SECUREC_COMPATIBLE_WIN_FORMAT)
        if (SecIsLeadByte(ch)) {
            temp[1] = (char)SecGetChar(stream, charCount);
            temp[2] = '\0'; /* 2 of string terminator position */
        }
        if (mbtowc(&tempWChar, temp, sizeof(temp)) <= 0) {
            /* no string termination error for tool */
            tempWChar = L'?';
        }
#else
        if (SecIsLeadByte(ch)) {
            int convRes = 0;
            int di = 1;
            /* in Linux like system, the string is encoded in UTF-8 */
            while (convRes <= 0 && di < (int)MB_CUR_MAX && di < SECUREC_MULTI_BYTE_MAX_LEN) {
                temp[di++] = (char)SecGetChar(stream, charCount);
                temp[di] = '\0';
                convRes = mbtowc(&tempWChar, temp, sizeof(temp));
            }
            if (convRes <= 0) {
                tempWChar = L'?';
            }
        } else {
            if (mbtowc(&tempWChar, temp, sizeof(temp)) <= 0) {
                /* no string termination error for tool */
                tempWChar = L'?';
            }
        }
#endif
#endif /* SECUREC_HAVE_MBTOWC */
        *(wchar_t UNALIGNED *)endPtr = tempWChar;
        /* just copy L'?' if mbtowc fails, errno is set by mbtowc */
        endPtr = (wchar_t *)endPtr + 1;
        --spec->arrayWidth;
        (void)charCount;
        (void)stream;
    } else {
        *(char *)endPtr = (char)ch;
        endPtr = (char *)endPtr + 1;
        --spec->arrayWidth;
    }
    spec->argPtr = endPtr;
    return 0;
}
#endif


#if SECUREC_ENABLE_SCANF_FLOAT

/* no not use localeconv()->decimal_pointif  onlay support  '.' */
#define SECURE_IS_FLOAT_DECIMAL(ch) ((ch) == SECUREC_CHAR('.'))
/*
 * init SecFloatSpec befor parse format
 */
static void SecInitFloatSpec(SecFloatSpec *floatSpec)
{
    floatSpec->floatStr = floatSpec->buffer;
    floatSpec->allocatedFloatStr = NULL;
    floatSpec->floatStrSize = sizeof(floatSpec->buffer) / sizeof(floatSpec->buffer[0]);
    floatSpec->floatStr = floatSpec->buffer;
    floatSpec->floatStrUsedLen = 0;
}

static void SecClearFloatSpec(SecFloatSpec *floatSpec, int *doneCount)
{
     /* LSD 2014.3.6 add, clear the stack data */
    if (memset_s(floatSpec->buffer, sizeof(floatSpec->buffer), 0,
        sizeof(floatSpec->buffer)) != EOK) {
        *doneCount = 0;  /* This is a dead code, just to meet the coding requirements */
    }
    if (floatSpec->allocatedFloatStr != NULL) {
        /* pFloatStr can be alloced in SecUpdateFloatString function, clear and free it */
        if (memset_s(floatSpec->allocatedFloatStr, floatSpec->floatStrSize * sizeof(SecChar), 0,
            floatSpec->floatStrSize * sizeof(SecChar)) != EOK) {
            *doneCount = 0; /* This is a dead code, just to meet the coding requirements */
        }
        SECUREC_FREE(floatSpec->allocatedFloatStr);
        floatSpec->allocatedFloatStr = NULL;
        floatSpec->floatStr = NULL;
    }
}


/*
 * scan value of exponent.
 * return 0 OK
 */
static int SecInputFloatE(SecFileStream *stream, SecScanSpec *spec, SecFloatSpec *floatSpec, int *charCount)
{
    SecInt ch = SecGetChar(stream, charCount);
    if (ch == SECUREC_CHAR('+') || ch == SECUREC_CHAR('-')) {
        if (ch == SECUREC_CHAR('-') && SecUpdateFloatString((SecChar)'-', floatSpec) != 0) {
            return -1;
        }
        if (spec->width != 0) {
            ch = SecGetChar(stream, charCount);
            --spec->width;
        }
    }

    while (SECUREC_IS_DIGIT(ch) && spec->width-- != 0) {
        if (SecUpdateFloatString((SecChar)ch, floatSpec) != 0) {
            return -1;
        }
        ch = SecGetChar(stream, charCount);
    }
    return 0;
}

/*
 * scan %f.
 * return 0 OK
 */
static int SecInputFloat(SecFileStream *stream, SecScanSpec *spec, SecFloatSpec *floatSpec, int *charCount)
{
    int started = -1;
    SecInt ch = SecGetChar(stream, charCount);

    floatSpec->floatStrUsedLen = 0;
    if (ch == SECUREC_CHAR('-')) {
        floatSpec->floatStr[floatSpec->floatStrUsedLen++] = SECUREC_CHAR('-');
        --spec->width;
        ch = SecGetChar(stream, charCount);
    } else if (ch == SECUREC_CHAR('+')) {
        --spec->width;
        ch = SecGetChar(stream, charCount);
    }

    if (spec->widthSet == 0) {    /* must care width */
        spec->width = -1; /* -1 is unlimited */
    }

    /* now get integral part */
    while (SECUREC_IS_DIGIT(ch) && spec->width-- != 0) {
        started = 0;
        /* ch must be '0' - '9' */
        if (SecUpdateFloatString((SecChar)ch, floatSpec) != 0) {
            return -1;
        }
        ch = SecGetChar(stream, charCount);
    }

    /* now get fractional part */
    if (SECURE_IS_FLOAT_DECIMAL((SecChar)ch) && spec->width-- != 0) {
        /* now check for decimal */
        if (SecUpdateFloatString((SecChar)ch, floatSpec) != 0) {
            return -1;
        }
        ch = SecGetChar(stream, charCount);
        while (SECUREC_IS_DIGIT(ch) && spec->width-- != 0) {
            started = 0;
            if (SecUpdateFloatString((SecChar)ch, floatSpec) != 0) {
                return -1;
            }
            ch = SecGetChar(stream, charCount);
        }
    }

    /* now get exponent part */
    if (started == 0 && (ch == SECUREC_CHAR('e') || ch == SECUREC_CHAR('E')) && spec->width-- != 0) {
        if (SecUpdateFloatString((SecChar)'e', floatSpec) != 0) {
            return -1;
        }
        if (SecInputFloatE(stream, spec, floatSpec, charCount) != 0) {
            return -1;
        }
    }
    /* un set the last character that is not a floating point number */
    SecUnGetChar(ch, stream, charCount);
    /* Make sure  have a string terminator, buffer is large enough */
    floatSpec->floatStr[floatSpec->floatStrUsedLen] = SECUREC_CHAR('\0');
    return started;

}
#endif

/*
 * scan digital part of %d %i %o %u %x %p.
 * return 0 OK
 */
static int SecInputNumberDigital(SecInt firstCh, SecFileStream *stream, SecScanSpec *spec, int *charCount)
{
    SecInt ch = firstCh;
    int loopFlag = 0;
    int started = -1;
    while (loopFlag == 0) {
        /* decode ch to number */
        loopFlag = SecDecodeNumber(ch, spec);
        if (loopFlag == 0) {
            started = 0;
            if (spec->widthSet != 0 && --spec->width == 0) {
                loopFlag = 1;
            } else {
                ch = SecGetChar(stream, charCount);
            }
        } else {
            SecUnGetChar(ch, stream, charCount);
        }
    }

    /* Handling integer negative numbers and beyond max */
    (*g_secFinishNumber[spec->isInt64Arg])(spec);
    return started;

}

/*
 * scan %d %i %o %u %x %p.
 * return 0 OK
 */
static int SecInputNumber(SecFileStream *stream, SecScanSpec *spec, int *charCount)
{
    SecInt ch = SecGetChar(stream, charCount);

    if (ch == SECUREC_CHAR('+') || ch == SECUREC_CHAR('-')) {
        if (ch == SECUREC_CHAR('-')) {
            spec->negative = 1;
        }
        if (spec->widthSet != 0 && --spec->width == 0) {
            return -1;
        } else {
            ch = SecGetChar(stream, charCount);
        }
    }

    if (spec->oriComChr == SECUREC_CHAR('i')) {
        /* i could be d, o, or x, use d as default */
        spec->comChr = SECUREC_CHAR('d');
    }

    if (spec->oriComChr == SECUREC_CHAR('x') || spec->oriComChr == SECUREC_CHAR('i')) {
        if (ch != SECUREC_CHAR('0')) {
            /* scan number */
            return SecInputNumberDigital(ch, stream, spec, charCount);
        }
        /* now input string may be 0x123 or 0X123 or just 0 */
        /* get next char */
        ch = SecGetChar(stream, charCount);
        if ((SecChar)(ch) == SECUREC_CHAR('x') || (SecChar)ch == SECUREC_CHAR('X')) {
            spec->comChr = SECUREC_CHAR('x');
            ch = SecGetChar(stream, charCount);
            /* length of 0x is 2 */
            if (spec->widthSet != 0 && spec->width <= (1 + 1)) {
                /* length not enough for "0x" */
                return -1;
            }
            spec->width -= 2; /* Subtract 2 for the length of "0x" */
        } else {
            if (spec->oriComChr != SECUREC_CHAR('x')) {
                spec->comChr = SECUREC_CHAR('o');
            }
            /* unset the character after 0 back to stream, input only '0' result is OK */
            SecUnGetChar(ch, stream, charCount);
            ch = SECUREC_CHAR('0');
        }
    }
    /* scan number */
    return SecInputNumberDigital(ch, stream, spec, charCount);
}
/*
 * scan %c %s %[
 * return 0 OK
 */
static int SecInputString(SecFileStream *stream, SecScanSpec *spec,
    const SecBracketTable *bracketTable, int *charCount, int *doneCount)
{
    void *startPtr = spec->argPtr;
    int suppressed= 0;
    int errNoMem = 0;

    while (spec->widthSet == 0 || spec->width-- != 0) {
        SecInt ch = SecGetChar(stream, charCount);
        /* char  condition or string condition and bracket condition.
         * only supports  wide characters with a maximum length of two bytes
         */
        if ((ch != SECUREC_EOF) && (spec->comChr == SECUREC_CHAR('c') ||
            SECUREC_SCANF_STRING_CONDITION(spec->comChr, ch) ||
            SECUREC_SCANF_BRACKET_CONDITION(spec->comChr, ch, bracketTable->table, bracketTable->mask))) {
            if (spec->suppress != 0) {
                /* Used to identify processed data for %*
                 * use endPtr to identify will cause 613, so use suppressed
                 */
                suppressed = 1;
                continue;
            }
            /* now suppress is not set */
            if (spec->arrayWidth == 0) {
                errNoMem = 1; /* We have exhausted the user's buffer */
                break;
            }
#ifdef SECUREC_FOR_WCHAR
            errNoMem = SecInputForWchar(ch, spec);
#else
            errNoMem = SecInputForChar(ch, spec, stream, charCount);
#endif
            if (errNoMem != 0) {
                break;
            }
        } else {
            SecUnGetChar(ch, stream, charCount);
            break;
        }
    }

    if (errNoMem != 0) {
        /* In case of error, blank out the input buffer */
        if (spec->suppress == 0) {
            SecAddEndingZero(startPtr, spec);
        }
        return -1;
    }

    /* No input was scanned */
    if ((spec->suppress != 0 && suppressed == 0) ||
        (spec->suppress == 0 && startPtr == spec->argPtr)) {
        return -1;
    }

    if (spec->suppress == 0) {
        if (spec->comChr != 'c') {
            /* null-terminate strings */
            SecAddEndingZero(spec->argPtr, spec);
        }
        *doneCount = *doneCount + 1;
    }
    return 0;
}

#ifdef SECUREC_FOR_WCHAR
/*
 * alloce buffer for wchar version of %[.
 * return 0 OK
 */
static int SecAllocBracketTable(SecBracketTable *bracketTable)
{
    if (bracketTable->table == NULL) {
        /* table should be freed after use */
        bracketTable->table = (unsigned char *)SECUREC_MALLOC(SECUREC_BRACKET_TABLE_SIZE);
        if (bracketTable->table == NULL) {
            return -1;
        }
    }
    return 0;
}

/*
 * free buffer for wchar version of %[
 */
static void SecFreeBracketTable(SecBracketTable *bracketTable)
{
    if (bracketTable->table != NULL) {
        SECUREC_FREE(bracketTable->table);
        bracketTable->table = NULL;
    }
}
#endif

#ifdef SECUREC_FOR_WCHAR
/*
 *  Formatting input core functions for wchar version.Called by a function such as vsscanf_s
 */
int SecInputSW(SecFileStream *stream, const wchar_t *cFormat, va_list argList)
#else
/*
 * Formatting input core functions for char version.Called by a function such as vswscanf_s
 */
int SecInputS(SecFileStream *stream, const char *cFormat, va_list argList)
#endif
{
    const SecUnsignedChar *format = (const SecUnsignedChar *)cFormat;
    SecBracketTable bracketTable = SECUREC_INIT_BRACKET_TABLE;
    SecScanSpec spec;
    SecInt ch = 0;
    int charCount = 0;
    int doneCount = 0;
    int formatError = 0;
    int paraIsNull = 0;
#if SECUREC_ENABLE_SCANF_FLOAT
    SecFloatSpec floatSpec;
#endif
    int match = 0;
    int errRet = 0;
#if SECUREC_ENABLE_SCANF_FLOAT
    SecInitFloatSpec(&floatSpec);
#endif
    /* format must not NULL */
    /* use err < 1 to claer 845 */
    while (errRet < 1 && *format != SECUREC_CHAR('\0')) {
        /* skip space in format and space in input */
        if (SECUREC_IS_SPACE(*format)) {
            SecInt nonSpaceChar = SecSkipSpaceChar(stream, &charCount);
            /* eat all space chars and put fist no space char backup */
            SecUnGetChar(nonSpaceChar, stream, &charCount);
            SecSkipSpaceFormat(&format);
            continue;
        }

        if (*format != SECUREC_CHAR('%')) {
            ch = SecGetChar(stream, &charCount);
            if ((int)(*format++) != (int)(ch)) {
                SecUnGetChar(ch, stream, &charCount);
                ++errRet; /* use plus to clear 845 */
                continue;
            }
#ifndef SECUREC_FOR_WCHAR
            if (SecIsLeadByte(ch) && SecDecodeLeadByte(ch, &format, stream, &charCount) != 0) {
                ++errRet;
                continue;
            }
#endif
            /* for next %n */
            if ((ch == SECUREC_EOF) && ((*format != SECUREC_CHAR('%')) || (*(format + 1) != SECUREC_CHAR('n')))) {
                break;
            }
            continue;
        }

        /* now *format is % */
        /* set default value for each % */
        SecSetDefaultScanSpec(&spec);
        if (SecDecodeScanFlag(&format, &spec) != 0) {
            formatError = 1;
            ++errRet;
            continue;
        }
        /* update wchar flag for %S %C */
        SecUpdateWcharFlagByType(*format, &spec);

#if SECUREC_HAVE_WCHART == 0
        /* in kernel not support wide char */
        if (spec.isWChar > 0) {
            formatError = 1;
            ++errRet;
            continue;
        }
#endif
        if (spec.widthSet != 0 && spec.width == 0) {
            /* 0 width in format */
            ++errRet;
            continue;
        }

        spec.comChr = (unsigned char)(*format) | (SECUREC_CHAR('a') - SECUREC_CHAR('A')); /* to lowercase */
        spec.oriComChr = spec.comChr;

        if (spec.comChr != SECUREC_CHAR('n')) {
            if (spec.comChr != SECUREC_CHAR('c') && spec.comChr != SECUREC_BRACE) {
                ch = SecSkipSpaceChar(stream, &charCount);
            } else {
                ch = SecGetChar(stream, &charCount);
            }
            if (ch == SECUREC_EOF) {
                ++errRet;
                continue;
            }
        }

        /* now no 0 width in format and get one char from input */
        switch (spec.comChr) {
            case SECUREC_CHAR('c'): /* also 'C' */
                /* fall-through */ /* FALLTHRU */
            case SECUREC_CHAR('s'): /* also 'S': */
                /* fall-through */ /* FALLTHRU */
            case SECUREC_BRACE:
                /* check dest buffer and size */
                if (spec.suppress == 0) {
                    spec.argPtr = (void *)va_arg(argList, void *);
                    if (spec.argPtr == NULL) {
                        paraIsNull = 1;
                        ++errRet;
                        continue;
                    }
                    /* Get the next argument - size of the array in characters */
#ifdef SECUREC_ON_64BITS
                    spec.arrayWidth = ((size_t)(va_arg(argList, size_t))) & 0xFFFFFFFFUL;
#else /* !SECUREC_ON_64BITS */
                    spec.arrayWidth = (size_t)va_arg(argList, size_t);
#endif
                    if (spec.arrayWidth == 0 || (spec.isWChar <= 0 && spec.arrayWidth > SECUREC_STRING_MAX_LEN) ||
                        (spec.isWChar > 0 && spec.arrayWidth > SECUREC_WCHAR_STRING_MAX_LEN)) {
                        /* do not clear buffer just go error */
                        ++errRet;
                        continue;
                    }
                    /* One element is needed for '\0' for %s and %[ */
                    if (spec.comChr != SECUREC_CHAR('c')) {
                        --spec.arrayWidth;
                    }
                } else {
                    /*  Set argPtr to  NULL  is necessary, in supress mode we don't use argPtr to store data */
                    spec.argPtr = NULL;
                }

                if (spec.comChr == 'c') {
                    if (spec.widthSet == 0) {
                        spec.widthSet = 1;
                        spec.width = 1;
                    }
                } else if (spec.comChr == SECUREC_BRACE) {
                    /* malloc  when  first %[ is meet  for wchar version */
#ifdef SECUREC_FOR_WCHAR
                    if (SecAllocBracketTable(&bracketTable) != 0) {
                        ++errRet;
                        continue;
                    }

#endif
                    (void)memset(bracketTable.table, 0, (size_t)SECUREC_BRACKET_TABLE_SIZE);
                    if (SecSetupBracketTable(&format, &bracketTable) != 0) {
                        ++errRet;
                        continue;
                    }

                    if (*format == SECUREC_CHAR('\0')) {
                        if (spec.suppress == 0 && spec.arrayWidth > 0) {
                            SecAddEndingZero(spec.argPtr, &spec);
                        }
                        ++errRet;
                        /* truncated format */
                        continue;
                    }

                }
                /* un set last char to stream */
                SecUnGetChar(ch, stream, &charCount);
                /* scanset completed.  Now read string */
                if (SecInputString(stream, &spec, &bracketTable, &charCount, &doneCount) != 0) {
                    ++errRet;
                    continue;
                }
                break;
            case SECUREC_CHAR('p'):
                /* make %hp same as %p */
                spec.numberWidth = SECUREC_NUM_WIDTH_INT;
#ifdef SECUREC_ON_64BITS
                spec.isInt64Arg = 1;
#endif
                /* fall-through */ /* FALLTHRU */
            case SECUREC_CHAR('o'):    /* fall-through */ /* FALLTHRU */
            case SECUREC_CHAR('u'):    /* fall-through */ /* FALLTHRU */
            case SECUREC_CHAR('d'):    /* fall-through */ /* FALLTHRU */
            case SECUREC_CHAR('i'):    /* fall-through */ /* FALLTHRU */
            case SECUREC_CHAR('x'):
                /* un set last char to stream */
                SecUnGetChar(ch, stream, &charCount);
                if (SecInputNumber(stream, &spec, &charCount) != 0) {
                    ++errRet;
                    continue;
                }
                if (spec.suppress == 0) {
                    spec.argPtr = (void *)va_arg(argList, void *);
                    if (spec.argPtr == NULL) {
                        paraIsNull = 1;
                        ++errRet;
                        continue;
                    }
                    SecAssignNumber(&spec);
                    ++doneCount;
                }
                break;
            case SECUREC_CHAR('n'):    /* char count */
                if (spec.suppress == 0) {
                    spec.argPtr = (void *)va_arg(argList, void *);
                    if (spec.argPtr == NULL) {
                        paraIsNull = 1;
                        ++errRet;
                        continue;
                    }
                    spec.number = (unsigned long)(unsigned int)charCount;
                    spec.isInt64Arg = 0;
                    SecAssignNumber(&spec);
                }
                break;
            case SECUREC_CHAR('e'):    /* fall-through */ /* FALLTHRU */
            case SECUREC_CHAR('f'):    /* fall-through */ /* FALLTHRU */
            case SECUREC_CHAR('g'):    /* scan a float */
#if SECUREC_ENABLE_SCANF_FLOAT
                /* un set last char to stream */
                SecUnGetChar(ch, stream, &charCount);
                if (SecInputFloat(stream, &spec, &floatSpec, &charCount) != 0) {
                    ++errRet;
                    continue;
                }
                if (spec.suppress == 0) {
                    spec.argPtr = (void *)va_arg(argList, void *);
                    if (spec.argPtr == NULL) {
                        ++errRet;
                        paraIsNull = 1;
                        continue;
                    }
#ifdef SECUREC_FOR_WCHAR
                    if (SecAssignFloatW(&floatSpec, &spec) != 0) {
                        ++errRet;
                        continue;
                    }
#else
                    SecAssignFloat(floatSpec.floatStr, spec.numberWidth, spec.argPtr);
#endif
                    ++doneCount;
                }

                break;
#else /* SECUREC_ENABLE_SCANF_FLOAT */
                ++errRet;
                continue;
#endif
            default:
                if ((int)(*format) != (int)ch) {
                    SecUnGetChar(ch, stream, &charCount);
                    formatError = 1;
                    ++errRet;
                    continue;
                } else {
                    --match;
                }
        }

        ++match;
        ++format;
        if ((ch == SECUREC_EOF) && ((*format != SECUREC_CHAR('%')) || (*(format + 1) != SECUREC_CHAR('n')))) {
            break;
        }
    }

#ifdef SECUREC_FOR_WCHAR
    SecFreeBracketTable(&bracketTable);
#endif

#if SECUREC_ENABLE_SCANF_FLOAT
    SecClearFloatSpec(&floatSpec, &doneCount);
#endif

#if SECUREC_ENABLE_SCANF_FILE
    SecAdjustStream(stream);
#endif

    if (ch == SECUREC_EOF) {
        return ((doneCount || match) ? doneCount : SECUREC_SCANF_EINVAL);
    } else if (formatError != 0 || paraIsNull != 0) {
        /* Invalid Input Format or parameter */
        return SECUREC_SCANF_ERROR_PARA;
    }

    return doneCount;
}

#if SECUREC_ENABLE_SCANF_FILE

#if defined(SECUREC_NO_STD_UNGETC)
/*
 *  Get char  from stdin or buffer
 */
static SecInt SecGetCharFromStdin(SecFileStream *stream)
{
    SecInt ch;
    if (stream->fUnget == 1) {
        ch = (SecInt) stream->lastChar;
        stream->fUnget = 0;
    } else {
        ch = SECUREC_GETC(stream->pf);
        stream->lastChar = (unsigned int)ch;
    }
    return ch;
}
#else
/*
 *  Get char  from stdin or buffer use std function
 */
static SecInt SecGetCharFromStdin(const SecFileStream *stream)
{
    SecInt ch;
    ch = SECUREC_GETC(stream->pf);
    return ch;
}
#endif

static void SecSkipBomHeader(SecFileStream *stream)
{
#ifdef SECUREC_FOR_WCHAR
    if (stream->count >= SECUREC_BOM_HEADER_SIZE &&
        (((unsigned char)(stream->base[0]) == SECUREC_BOM_HEADER_LE_1ST &&
        (unsigned char)(stream->base[1]) == SECUREC_BOM_HEADER_LE_2ST) ||
        ((unsigned char)(stream->base[0]) == SECUREC_BOM_HEADER_BE_1ST &&
        (unsigned char)(stream->base[1]) == SECUREC_BOM_HEADER_BE_2ST))) {

        /* the stream->count must be a  multiple of  sizeof(SecChar),
         * otherwise this function will return SECUREC_EOF when read the last character
         */
        if ((stream->count - SECUREC_BOM_HEADER_SIZE) % (int)sizeof(SecChar) != 0) {
            int ret = (int)fread(stream->base + stream->count, (size_t)1,
                                 (size_t)SECUREC_BOM_HEADER_SIZE, stream->pf);
            if (ret > 0 && ret <= SECUREC_BUFFERED_BLOK_SIZE) {
                stream->count += ret;
            }
        }
        /* it's BOM header, skip */
        stream->count -= SECUREC_BOM_HEADER_SIZE;
        stream->cur += SECUREC_BOM_HEADER_SIZE;
    }
#else
    if (stream->count >= SECUREC_UTF8_BOM_HEADER_SIZE &&
        (unsigned char)(stream->base[0]) == SECUREC_UTF8_BOM_HEADER_1ST &&
        (unsigned char)(stream->base[1]) == SECUREC_UTF8_BOM_HEADER_2ND &&
        (unsigned char)(stream->base[2]) == SECUREC_UTF8_BOM_HEADER_3RD) { /* 2 offset of third head character */
        /* it's BOM header, skip */
        stream->count -= SECUREC_UTF8_BOM_HEADER_SIZE;
        stream->cur += SECUREC_UTF8_BOM_HEADER_SIZE;
    }
#endif
}
/*
 *  Get char  from file stream or buffer
 */
static SecInt SecGetCharFromFile(SecFileStream *stream)
{
    SecInt ch;
    if (stream->count == 0) {
        int firstReadOnFile = 0;
        /* load file to buffer */
        if (stream->base == NULL) {
            stream->base = (char *)SECUREC_MALLOC(SECUREC_BUFFERED_BLOK_SIZE + 1);
            if (stream->base == NULL) {
                return SECUREC_EOF;
            }
            stream->base[SECUREC_BUFFERED_BLOK_SIZE] = '\0';   /* for tool Warning string null */
        }
        /* LSD add 2014.3.21 */
        if (stream->oriFilePos == SECUREC_UNINITIALIZED_FILE_POS) {
            stream->oriFilePos = ftell(stream->pf);   /* save original file read position */
            firstReadOnFile = 1;
        }
        stream->count = (int)fread(stream->base, (size_t)1, (size_t)SECUREC_BUFFERED_BLOK_SIZE, stream->pf);
        stream->base[SECUREC_BUFFERED_BLOK_SIZE] = '\0';   /* for tool Warning string null */
        if (stream->count == 0 || stream->count > SECUREC_BUFFERED_BLOK_SIZE) {
            return SECUREC_EOF;
        }
        stream->cur = stream->base;
        stream->flag |= SECUREC_LOAD_FILE_TO_MEM_FLAG;
        if (firstReadOnFile != 0) {
            SecSkipBomHeader(stream);
        }
    }
    /* according  wchar_t has two bytes */
    ch = (SecInt)((stream->count -= (int)sizeof(SecChar)) >= 0 ? \
                  (SecInt)(SECUREC_CHAR_MASK & \
                  (unsigned int)(int)(*((const SecChar *)(const void *)stream->cur))) : SECUREC_EOF);
    stream->cur += sizeof(SecChar);

    if (ch != SECUREC_EOF && stream->base != NULL) {
        stream->fileRealRead += (int)sizeof(SecChar);
    }
    return ch;
}
#endif

/*
 *  Get char  for wchar version
 */
static SecInt SecGetChar(SecFileStream *stream, int *counter)
{
    SecInt ch = SECUREC_EOF;
#if SECUREC_ENABLE_SCANF_FILE
    if ((stream->flag & SECUREC_FROM_STDIN_FLAG) > 0) {
        ch = SecGetCharFromStdin(stream);
    } else if ((stream->flag & SECUREC_FILE_STREAM_FLAG) > 0) {
        ch = SecGetCharFromFile(stream);
    }
#endif
    if ((stream->flag & SECUREC_MEM_STR_FLAG) > 0) {
        /* according  wchar_t has two bytes */
        ch = (SecInt)((stream->count -= (int)sizeof(SecChar)) >= 0 ? \
                      (SecInt)(SECUREC_CHAR_MASK & \
                      (unsigned int)(int)(*((const SecChar *)(const void *)stream->cur))) : SECUREC_EOF);
        stream->cur += sizeof(SecChar);
    }
    *counter = *counter + 1;
    return ch;
}

/*
 *  Unget Public realizatio char  for wchar and char version
 */
static void SecUnGetCharImpl(SecInt ch, SecFileStream *stream)
{
    if ((stream->flag & SECUREC_FROM_STDIN_FLAG) > 0) {
#if SECUREC_ENABLE_SCANF_FILE
#if defined(SECUREC_NO_STD_UNGETC)
        stream->lastChar = (unsigned int)ch;
        stream->fUnget = 1;
#else
        (void)SECUREC_UN_GETC(ch, stream->pf);
#endif
#else
        (void)ch; /* to clear e438 last value assigned not used , the compiler will optimize this code */
#endif
    } else if ((stream->flag & SECUREC_MEM_STR_FLAG) || (stream->flag & SECUREC_LOAD_FILE_TO_MEM_FLAG) > 0) {
        if (stream->cur > stream->base) {
            stream->cur -= sizeof(SecChar);
            stream->count += (int)sizeof(SecChar);
        }
    }
#if SECUREC_ENABLE_SCANF_FILE
    if ((stream->flag & SECUREC_FILE_STREAM_FLAG) > 0 && stream->base) {
        stream->fileRealRead -= (int)sizeof(SecChar);
    }
#endif
}

/*
 *  Unget char  for char version
 */
static void SecUnGetChar(SecInt ch, SecFileStream *stream, int *counter)
{
    if (ch != SECUREC_EOF) {
        SecUnGetCharImpl(ch, stream);
    }
    *counter = *counter - 1;
}

/*
 *  Skip space char by isspace
 */
static SecInt SecSkipSpaceChar(SecFileStream *stream, int *counter)
{
    SecInt ch;
    do {
        ch = SecGetChar(stream, counter);
    } while (ch != SECUREC_EOF && SECUREC_IS_SPACE(ch));
    return ch;
}
#endif /* __INPUT_INL__5D13A042_DC3F_4ED9_A8D1_882811274C27 */

