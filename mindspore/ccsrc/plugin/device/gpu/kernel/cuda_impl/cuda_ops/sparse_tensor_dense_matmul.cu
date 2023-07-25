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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_tensor_dense_matmul.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T, typename S>
__global__ void SparseTensorDenseMatmul(const size_t values_size_, const size_t out_dim_1, const size_t b_rows,
                                        const size_t b_cols, const T *a_indices, const S *a_values, const S *b,
                                        S *output, bool adj_st, bool adj_dt) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < values_size_ * out_dim_1;
       pos += blockDim.x * gridDim.x) {
    int64_t a_ix = pos / out_dim_1;
    int64_t i = adj_st ? a_indices[2 * a_ix + 1] : a_indices[2 * a_ix];
    int64_t j = pos % out_dim_1;
    int64_t k = adj_st ? a_indices[2 * a_ix] : a_indices[2 * a_ix + 1];
    S a_value = a_values[a_ix];
    S b_value = adj_dt ? b[j * b_cols + k] : b[k * b_cols + j];
    MsAtomicAdd(&output[i * out_dim_1 + j], a_value * b_value);
  }
  return;
}

template <typename T, typename S>
cudaError_t CalSparseTensorDenseMatmul(const size_t values_size_, const size_t out_dim_1, const size_t b_rows,
                                       const size_t b_cols, const T *a_indices, const S *a_values, const S *b,
                                       S *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
                                       cudaStream_t cuda_stream) {
  cudaMemset(output, 0, sizeof(S) * b_rows * out_dim_1);
  SparseTensorDenseMatmul<<<CUDA_BLOCKS(device_id, values_size_ * out_dim_1), CUDA_THREADS(device_id), 0,
                            cuda_stream>>>(values_size_, out_dim_1, b_rows, b_cols, a_indices, a_values, b, output,
                                           adj_st, adj_dt);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int32_t, int32_t>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int32_t *a_indices,
  const int32_t *a_values, const int32_t *b, int32_t *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int32_t, int64_t>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int32_t *a_indices,
  const int64_t *a_values, const int64_t *b, int64_t *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int32_t, half>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int32_t *a_indices,
  const half *a_values, const half *b, half *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int32_t, float>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int32_t *a_indices,
  const float *a_values, const float *b, float *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int32_t, double>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int32_t *a_indices,
  const double *a_values, const double *b, double *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int32_t, Complex<float>>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int32_t *a_indices,
  const Complex<float> *a_values, const Complex<float> *b, Complex<float> *output, bool adj_st, bool adj_dt,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int32_t, Complex<double>>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int32_t *a_indices,
  const Complex<double> *a_values, const Complex<double> *b, Complex<double> *output, bool adj_st, bool adj_dt,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int64_t, int32_t>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int64_t *a_indices,
  const int32_t *a_values, const int32_t *b, int32_t *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int64_t, int64_t>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int64_t *a_indices,
  const int64_t *a_values, const int64_t *b, int64_t *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int64_t, half>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int64_t *a_indices,
  const half *a_values, const half *b, half *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int64_t, float>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int64_t *a_indices,
  const float *a_values, const float *b, float *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int64_t, double>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int64_t *a_indices,
  const double *a_values, const double *b, double *output, bool adj_st, bool adj_dt, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int64_t, Complex<float>>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int64_t *a_indices,
  const Complex<float> *a_values, const Complex<float> *b, Complex<float> *output, bool adj_st, bool adj_dt,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseTensorDenseMatmul<int64_t, Complex<double>>(
  const size_t values_size_, const size_t out_dim_1, const size_t b_rows, const size_t b_cols, const int64_t *a_indices,
  const Complex<double> *a_values, const Complex<double> *b, Complex<double> *output, bool adj_st, bool adj_dt,
  const uint32_t &device_id, cudaStream_t cuda_stream);
