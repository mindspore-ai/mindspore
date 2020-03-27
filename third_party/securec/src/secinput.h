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

#ifndef SEC_INPUT_H_E950DA2C_902F_4B15_BECD_948E99090D9C
#define SEC_INPUT_H_E950DA2C_902F_4B15_BECD_948E99090D9C
#include "securecutil.h"

#define SECUREC_SCANF_EINVAL             (-1)
#define SECUREC_SCANF_ERROR_PARA         (-2)

/* for internal stream flag */
#define SECUREC_MEM_STR_FLAG             0X01
#define SECUREC_FILE_STREAM_FLAG         0X02
#define SECUREC_FROM_STDIN_FLAG          0X04
#define SECUREC_LOAD_FILE_TO_MEM_FLAG    0X08

#define SECUREC_UNINITIALIZED_FILE_POS   (-1)
#define SECUREC_BOM_HEADER_SIZE          2
#define SECUREC_BOM_HEADER_BE_1ST        0xFEU
#define SECUREC_BOM_HEADER_BE_2ST        0xFFU
#define SECUREC_BOM_HEADER_LE_1ST        0xFFU
#define SECUREC_BOM_HEADER_LE_2ST        0xFEU
#define SECUREC_UTF8_BOM_HEADER_SIZE     3
#define SECUREC_UTF8_BOM_HEADER_1ST      0xEFU
#define SECUREC_UTF8_BOM_HEADER_2ND      0xBBU
#define SECUREC_UTF8_BOM_HEADER_3RD      0xBFU
#define SECUREC_UTF8_LEAD_1ST            0xE0
#define SECUREC_UTF8_LEAD_2ND            0x80

typedef struct {
    unsigned int flag;          /* mark the properties of input stream */
    int count;                  /* the size of buffered string in bytes */
    const char *cur;            /* the pointer to next read position */
    char *base;                 /* the pointer to the header of buffered string */
#if SECUREC_ENABLE_SCANF_FILE
    FILE *pf;                   /* the file pointer */
    long oriFilePos;            /* the original position of file offset when fscanf is called */
    int fileRealRead;
#if defined(SECUREC_NO_STD_UNGETC)
    unsigned int lastChar;      /* the char code of last input */
    int fUnget;                 /* the boolean flag of pushing a char back to read stream */
#endif
#endif
} SecFileStream;


#define SECUREC_INIT_SEC_FILE_STREAM_COMMON(fileStream, streamFlag, curPtr, strCount) do { \
    (fileStream).flag = (streamFlag); \
    (fileStream).count = (strCount); \
    (fileStream).cur = (curPtr); \
    (fileStream).base = NULL; \
} SECUREC_WHILE_ZERO

#if SECUREC_ENABLE_SCANF_FILE
#if defined(SECUREC_NO_STD_UNGETC)
/* This initialization for eliminating redundant initialization.
 * Compared with the previous version initialization 0,
 * the current code causes the binary size to increase by some bytes
 */
#define SECUREC_INIT_SEC_FILE_STREAM(fileStream, streamFlag, stream, filePos, curPtr, strCount) do { \
    SECUREC_INIT_SEC_FILE_STREAM_COMMON((fileStream), (streamFlag), (curPtr), (strCount)); \
    (fileStream).pf = (stream); \
    (fileStream).oriFilePos = (filePos); \
    (fileStream).fileRealRead = 0; \
    (fileStream).lastChar = 0; \
    (fileStream).fUnget = 0; \
} SECUREC_WHILE_ZERO
#else
#define SECUREC_INIT_SEC_FILE_STREAM(fileStream, streamFlag, stream, filePos, curPtr, strCount) do { \
    SECUREC_INIT_SEC_FILE_STREAM_COMMON((fileStream), (streamFlag), (curPtr), (strCount)); \
    (fileStream).pf = (stream); \
    (fileStream).oriFilePos = (filePos); \
    (fileStream).fileRealRead = 0; \
} SECUREC_WHILE_ZERO
#endif
#else /* No SECUREC_ENABLE_SCANF_FILE */
#define SECUREC_INIT_SEC_FILE_STREAM(fileStream, streamFlag, stream, filePos, curPtr, strCount) do { \
    SECUREC_INIT_SEC_FILE_STREAM_COMMON((fileStream), (streamFlag), (curPtr), (strCount)); \
} SECUREC_WHILE_ZERO
#endif

#ifdef __cplusplus
extern "C" {
#endif

    extern int SecInputS(SecFileStream *stream, const char *cFormat, va_list argList);
    extern void SecClearDestBuf(const char *buffer, const char *format, va_list argList);
#if SECUREC_IN_KERNEL == 0
    extern int SecInputSW(SecFileStream *stream, const wchar_t *cFormat, va_list argList);
    extern void SecClearDestBufW(const wchar_t *buffer, const wchar_t *format, va_list argList);
#endif
/* 20150105 For software and hardware decoupling,such as UMG */
#if defined(SECUREC_SYSAPI4VXWORKS)
#ifdef feof
#undef feof
#endif
    extern int feof(FILE *stream);
#endif

#if defined(SECUREC_SYSAPI4VXWORKS) || defined(SECUREC_CTYPE_MACRO_ADAPT)
#ifndef isspace
#define isspace(c) (((c) == ' ') || ((c) == '\t') || ((c) == '\r') || ((c) == '\n'))
#endif
#ifndef iswspace
#define iswspace(c) (((c) == L' ') || ((c) == L'\t') || ((c) == L'\r') || ((c) == L'\n'))
#endif
#ifndef isascii
#define isascii(c) (((unsigned char)(c)) <= 0x7f)
#endif
#ifndef isupper
#define isupper(c) ((c) >= 'A' && (c) <= 'Z')
#endif
#ifndef islower
#define islower(c) ((c) >= 'a' && (c) <= 'z')
#endif
#ifndef isalpha
#define isalpha(c) (isupper(c) || (islower(c)))
#endif
#ifndef isdigit
#define isdigit(c) ((c) >= '0' && (c) <= '9')
#endif
#ifndef isxupper
#define isxupper(c) ((c) >= 'A' && (c) <= 'F')
#endif
#ifndef isxlower
#define isxlower(c) ((c) >= 'a' && (c) <= 'f')
#endif
#ifndef isxdigit
#define isxdigit(c) (isdigit(c) || isxupper(c) || isxlower(c))
#endif
#endif

#ifdef __cplusplus
}
#endif
/* Reserved file operation macro interface */
#define SECUREC_LOCK_FILE(s)
#define SECUREC_UNLOCK_FILE(s)
#define SECUREC_LOCK_STDIN(i, s)
#define SECUREC_UNLOCK_STDIN(i, s)
#endif


