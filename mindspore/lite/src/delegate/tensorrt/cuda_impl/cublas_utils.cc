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

#include "src/delegate/tensorrt/cuda_impl/cublas_utils.h"

namespace mindspore::lite {
void Cublas2DTranspose(const float *in_addr, float *out_addr, const int *params, cublasHandle_t cublas_handle) {
  const int m = params[0];
  const int n = params[1];
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK_VOID(
    cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, in_addr, n, &beta, out_addr, m, out_addr, m));
}

void CublasMM1Batch(const void *a_addr, const void *b_addr, void *c_addr, const int *params,
                    const cublasOperation_t *operations, const cudaDataType_t *data_types,
                    cublasComputeType_t type_compute, cublasHandle_t cublas_handle) {
  const int m = params[0];
  const int n = params[1];
  const int k = params[2];
  cublasOperation_t trans_a = operations[0];
  cublasOperation_t trans_b = operations[1];
  const int lda = (trans_a == CUBLAS_OP_N) ? k : m;
  const int ldb = (trans_b == CUBLAS_OP_N) ? n : k;
  const int ldc = n;
  cudaDataType_t type_a = data_types[0];
  cudaDataType_t type_b = data_types[1];
  cudaDataType_t type_c = data_types[2];
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK_VOID(cublasGemmEx(cublas_handle, trans_b, trans_a, n, m, k, &alpha, b_addr, type_b, ldb, a_addr, type_a,
                                 lda, &beta, c_addr, type_c, ldc, type_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
}  // namespace mindspore::lite
