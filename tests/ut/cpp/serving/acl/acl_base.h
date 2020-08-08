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

#ifndef ACL_STUB_INC_ACL_BASE
#define ACL_STUB_INC_ACL_BASE
#include <stdint.h>
#include <stddef.h>

typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclrtContext;
typedef int aclError;
typedef uint16_t aclFloat16;
typedef struct aclDataBuffer aclDataBuffer;
typedef struct aclTensorDesc aclTensorDesc;

const int ACL_ERROR_NONE = 0;

typedef enum {
  ACL_DT_UNDEFINED = -1,
  ACL_FLOAT = 0,
  ACL_FLOAT16 = 1,
  ACL_INT8 = 2,
  ACL_INT32 = 3,
  ACL_UINT8 = 4,
  ACL_INT16 = 6,
  ACL_UINT16 = 7,
  ACL_UINT32 = 8,
  ACL_INT64 = 9,
  ACL_UINT64 = 10,
  ACL_DOUBLE = 11,
  ACL_BOOL = 12,
} aclDataType;

typedef enum {
  ACL_FORMAT_UNDEFINED = -1,
  ACL_FORMAT_NCHW = 0,
  ACL_FORMAT_NHWC = 1,
  ACL_FORMAT_ND = 2,
  ACL_FORMAT_NC1HWC0 = 3,
  ACL_FORMAT_FRACTAL_Z = 4,
  ACL_FORMAT_FRACTAL_NZ = 29,

} aclFormat;

typedef enum {
  ACL_DEBUG,
  ACL_INFO,
  ACL_WARNING,
  ACL_ERROR,
} aclLogLevel;

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size);
aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer);
void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer);
uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer);
size_t aclDataTypeSize(aclDataType dataType);

aclTensorDesc *aclCreateTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);
void aclDestroyTensorDesc(const aclTensorDesc *desc);
aclDataType aclGetTensorDescType(const aclTensorDesc *desc);
aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc);
size_t aclGetTensorDescSize(const aclTensorDesc *desc);
size_t aclGetTensorDescElementCount(const aclTensorDesc *desc);
size_t aclGetTensorDescNumDims(const aclTensorDesc *desc);
int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index);

void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line, const char *fmt, ...);

#define ACL_APP_LOG(level, fmt, ...) aclAppLog(level, __FUNCTION__, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#endif