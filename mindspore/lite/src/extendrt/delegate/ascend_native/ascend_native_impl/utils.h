/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_UTILS_H_

#include "acl/acl.h"
#include "acl/acl_rt.h"

#define PIPE_CAST 2
#define BLOCK_CAST (4 * 1024)
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ALIGN32(size) ((((size) + 32 - 1) / 32) * 32)
#define ALIGN(size, len) ((((size) + (len)-1) / (len)) * (len))
#define ALIGN_BY_TYPE(size, size_of_type, bytes) \
  ((((size) + ((bytes) / (size_of_type)) - 1) / ((bytes) / (size_of_type))) * ((bytes) / (size_of_type)))

constexpr float kFloatMSEC = 1000.0f;
const int USEC = 1000000;
const int MSEC = 1000;
#define CUBE_CORE_NUM 1
#define VEC_CORE_NUM 1
namespace mindspore::ascend_native {

void *MallocDevice(size_t size, void *ctx);
void FreeDevice(void *ptr, void *ctx);
void SyncDevice(void *stream, void *ctx);
void PrintFp16(void *x, size_t elem_num, void *stream, void *ctx);
void PrintFp32(void *x, size_t elem_num, void *stream, void *ctx);
void PrintInt32(void *x, size_t elem_num, void *stream, void *ctx);
void *CreateStream(void *context);
void *CreateContext(int32_t deviceId = 0);
void *MallocCopy(void *src, size_t size, void *ctx);
int InitAcl();
int FinalizeAcl();
void DestroyStream(void *stream, void *context);
void DestroyCtx(void *context);
void SetContext(void *ctx);
void printChecksumFp32(void *x, size_t elem_num, void *stream);
void printChecksumInt32(void *x, size_t elem_num, void *stream);
void PrintInfo(void *stream);
template <typename T>
void printVector(void *x, int elem_num, void *q);
void printChecksumFp16(void *x, size_t elem_num, void *stream, void *ctx);
void PrintFp16Host(void *x, size_t elem_num, void *stream, void *ctx);
void CopyHTD(void *dst, void *src, size_t size, void *ctx);
void CopyDTH(void *dst, void *src, size_t size, void *ctx);
uint64_t GetTimeUs();

class MMExtra {
 public:
  uint32_t bmm_num_ = 1;
  uint32_t lda_ = -1;
  uint32_t ldb_ = -1;
  uint32_t ldc_ = -1;
};

#define CHECK_ACL(x)                                                                  \
  do {                                                                                \
    aclError __ret = x;                                                               \
    if (__ret != ACL_ERROR_NONE) {                                                    \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
    }                                                                                 \
  } while (0);
}  // namespace mindspore::ascend_native
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_UTILS_H_
