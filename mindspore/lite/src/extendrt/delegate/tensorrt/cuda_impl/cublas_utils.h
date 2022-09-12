/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_CUDA_IMPL_CUBLAS_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_CUDA_IMPL_CUBLAS_UTILS_H_

#include <cublasLt.h>
#include <cublas_v2.h>
#include "src/extendrt/delegate/tensorrt/cuda_impl/cuda_helper.h"
#include "src/common/log_util.h"

// cublas API error checking
#define CUBLAS_CHECK_VOID(err)                        \
  do {                                                \
    cublasStatus_t cublas_err = (err);                \
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {        \
      MS_LOG(ERROR) << "cublas error " << cublas_err; \
      return;                                         \
    }                                                 \
  } while (0)

#define CUBLAS_CHECK(err)                             \
  do {                                                \
    cublasStatus_t cublas_err = (err);                \
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {        \
      MS_LOG(ERROR) << "cublas error " << cublas_err; \
      return -1;                                      \
    }                                                 \
  } while (0)

namespace mindspore::lite {
// a: m * n
// params order: m, n
void Cublas2DTranspose(const float *in_addr, float *out_addr, const int *params, cublasHandle_t cublas_handle);

// a: m * k, b: k * n, c: m * n
// params order: m, n, k
// operations order: trans_a, trans_b
// data_types: type_a, type_b, type_c, compute type
void CublasMM1Batch(const void *a_addr, const void *b_addr, void *c_addr, const int *params,
                    const cublasOperation_t *operations, const cudaDataType *data_types, cublasHandle_t cublas_handle);

// a: batch * m * k, b: batch * k * n, c: batch * m * n
// params order: m, n, k, batch
// operations order: trans_a, trans_b
// data_types: type_a, type_b, type_c, compute type
void CublasMMBatched(void **a_addrs, void **b_addrs, void **c_addrs, const int *params,
                     const cublasOperation_t *operations, const cudaDataType *data_types, cublasHandle_t cublas_handle);

void CublasGemmWrapper(const void *a_addr, const void *b_addr, void *c_addr, const int *params, const int *lds,
                       const cublasOperation_t *operations, const cudaDataType *data_types, void *alpha, void *beta,
                       cublasHandle_t cublas_handle);
void CublasGemmStridedBatchedWrapper(const void *a_addr, const void *b_addr, void *c_addr, const int *params,
                                     const int *lds, const cublasOperation_t *operations, const int *strides,
                                     const cudaDataType *data_types, void *alpha, void *beta, int batch,
                                     cublasHandle_t cublas_handle);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_CUDA_IMPL_CUBLAS_UTILS_H_
