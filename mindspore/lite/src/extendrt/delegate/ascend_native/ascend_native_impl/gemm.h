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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_GEMM_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_GEMM_H_

#include <stddef.h>

namespace mindspore::ascend_native {
void GemmFp32(void *queue, size_t m_size, size_t n_size, size_t k_size, float ALPHA, void *d_hA, size_t lda, void *d_hB,
              size_t ldb, float BETA, void *d_fC, size_t ldc, size_t core_num);

void GemmFp16(void *queue, bool ta, bool tb, size_t m_size, size_t n_size, size_t k_size, float ALPHA, void *d_hA,
              size_t lda, void *d_hB, size_t ldb, float BETA, void *d_fC, size_t ldc, size_t core_num);

void BGemmFp16(void *queue, bool ta, bool tb, size_t m_size, size_t n_size, size_t k_size, float ALPHA, void *d_hA,
               size_t lda, void *d_hB, size_t ldb, float BETA, void *d_fC, size_t ldc, int repeats, size_t core_num);
}  // namespace mindspore::ascend_native
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_GEMM_H_
