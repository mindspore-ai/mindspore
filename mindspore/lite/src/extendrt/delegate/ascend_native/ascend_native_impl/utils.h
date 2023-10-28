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

#include <stddef.h>

#define ASCEND_UB_BUFFER_SIZE 241280
#define ASCEND_HALF_UB_BUFFER_SIZE 120640
#define ASCEND_THIRD_UB_BUFFER_SIZE 80432
#define ASCEND_CROSS_BUF_SIZE 256
#define ASCEND_UB_NUM_ELEM (ASCEND_UB_BUFFER_SIZE / sizeof(T))
// 120640 for FP16
#define ASCEND_CROSS_NUM_ELEM (ASCEND_CROSS_BUF_SIZE / sizeof(T))
// 128 for FP16
#define MAX_CROSS_ELEMS (ASCEND_UB_NUM_ELEM / (ASCEND_CROSS_NUM_ELEM))
#define ACEND_LB_UB_BUFFER_SIZE = 25600;
// 938 for FP16
#define CeilDiv(x, y) (((x) + (y)-1) / (y))

namespace mindspore::ascend_native {
void *CreateStream();
void CopyHostFp32ToDeviceFp16(void *src, void **dst_ptr, size_t elem_num, void *stream);
void CopyHostFp32ToDeviceFp32(void *src, void **dst_ptr, size_t elem_num, void *stream);
void CopyHostFp16ToDeviceFp16(void *src, void **dst_ptr, size_t elem_num, void *stream);
void CopyHostFp16ToDeviceFp32(void *src, void **dst_ptr, size_t elem_num, void *stream);
void CopyDeviceFp16ToHostFp32(void *src, void *dst, size_t elem_num, void *stream);
void CopyDeviceFp32ToHostFp32(void *src, void *dst, size_t elem_num, void *stream);
void CopyDeviceFp16ToHostFp16(void *src, void *dst, size_t elem_num, void *stream);
void CopyDeviceFp32ToHostFp16(void *src, void *dst, size_t elem_num, void *stream);
void *MallocDevice(size_t size, void *stream);
void *MallocCopy(void *src, size_t size, void *stream);
void FreeDevice(void *ptr, void *stream);
void SyncDevice(void *stream);
void PrintFp16(void *x, size_t elem_num, void *stream);
void PrintFp32(void *x, size_t elem_num, void *stream);
void PrintInt32(void *x, size_t elem_num, void *stream);
void printChecksumFp16(void *x, size_t elem_num, void *stream);
void printChecksumFp32(void *x, size_t elem_num, void *stream);
void printChecksumInt32(void *x, size_t elem_num, void *stream);
void PreapreVSL(void *batch_valid_len, int batch_size, int *token_num, void *q);
void PrintInfo(void *stream);
template <typename T>
void printVector(void *x, int elem_num, void *q);
void GetTokenNum(void *batch_valid_len, int batch_size, int *token_num, void *q);

}  // namespace mindspore::ascend_native
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_UTILS_H_
