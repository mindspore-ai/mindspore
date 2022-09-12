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

#include "src/extendrt/delegate/tensorrt/cuda_impl/cublas_utils.h"

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
                    const cublasOperation_t *operations, const cudaDataType *data_types, cublasHandle_t cublas_handle) {
  const int m = params[0];
  const int n = params[1];
  const int k = params[2];
  cublasOperation_t trans_a = operations[0];
  cublasOperation_t trans_b = operations[1];
  const int lda = (trans_a == CUBLAS_OP_N) ? k : m;
  const int ldb = (trans_b == CUBLAS_OP_N) ? n : k;
  const int ldc = n;
  cudaDataType type_a = data_types[0];
  cudaDataType type_b = data_types[1];
  cudaDataType type_c = data_types[2];
  cudaDataType compute_type = data_types[3];
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK_VOID(cublasGemmEx(cublas_handle, trans_b, trans_a, n, m, k, &alpha, b_addr, type_b, ldb, a_addr, type_a,
                                 lda, &beta, c_addr, type_c, ldc, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void CublasMMBatched(void **a_addrs, void **b_addrs, void **c_addrs, const int *params,
                     const cublasOperation_t *operations, const cudaDataType *data_types,
                     cublasHandle_t cublas_handle) {
  cublasOperation_t trans_a = operations[0];
  cublasOperation_t trans_b = operations[1];
  const int m = params[0];
  const int n = params[1];
  const int k = params[2];
  const int batch = params[3];
  const int lda = (trans_a == CUBLAS_OP_N) ? k : m;
  const int ldb = (trans_b == CUBLAS_OP_N) ? n : k;
  const int ldc = n;
  cudaDataType type_a = data_types[0];
  cudaDataType type_b = data_types[1];
  cudaDataType type_c = data_types[2];
  cudaDataType compute_type = data_types[3];
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK_VOID(cublasGemmBatchedEx(cublas_handle, trans_b, trans_a, n, m, k, &alpha, b_addrs, type_b, ldb, a_addrs,
                                        type_a, lda, &beta, c_addrs, type_c, ldc, batch, compute_type,
                                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void CublasGemmWrapper(const void *a_addr, const void *b_addr, void *c_addr, const int *params, const int *lds,
                       const cublasOperation_t *operations, const cudaDataType *data_types, void *alpha, void *beta,
                       cublasHandle_t cublas_handle) {
  const int m = params[0];
  const int n = params[1];
  const int k = params[2];
  cublasOperation_t trans_a = operations[0];
  cublasOperation_t trans_b = operations[1];
  const int lda = lds[0];
  const int ldb = lds[1];
  const int ldc = lds[2];
  cudaDataType type_a = data_types[0];
  cudaDataType type_b = data_types[1];
  cudaDataType type_c = data_types[2];
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;

  CUBLAS_CHECK_VOID(cublasGemmEx(cublas_handle, trans_a, trans_b, m, n, k, alpha, a_addr, type_a, lda, b_addr, type_b,
                                 ldb, beta, c_addr, type_c, ldc, compute_type, CUBLAS_GEMM_DEFAULT));
}

void CublasGemmStridedBatchedWrapper(const void *a_addr, const void *b_addr, void *c_addr, const int *params,
                                     const int *lds, const cublasOperation_t *operations, const int *strides,
                                     const cudaDataType *data_types, void *alpha, void *beta, int batch,
                                     cublasHandle_t cublas_handle) {
  const int m = params[0];
  const int n = params[1];
  const int k = params[2];
  cublasOperation_t trans_a = operations[0];
  cublasOperation_t trans_b = operations[1];
  const int lda = lds[0];
  const int ldb = lds[1];
  const int ldc = lds[2];
  cudaDataType type_a = data_types[0];
  cudaDataType type_b = data_types[1];
  cudaDataType type_c = data_types[2];
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  const int stride_a = strides[0];
  const int stride_b = strides[1];
  const int stride_c = strides[2];

  CUBLAS_CHECK_VOID(cublasGemmStridedBatchedEx(cublas_handle, trans_a, trans_b, m, n, k, alpha, a_addr, type_a, lda,
                                               stride_a, b_addr, type_b, ldb, stride_b, beta, c_addr, type_c, ldc,
                                               stride_c, batch, compute_type, CUBLAS_GEMM_DEFAULT));
}
}  // namespace mindspore::lite
