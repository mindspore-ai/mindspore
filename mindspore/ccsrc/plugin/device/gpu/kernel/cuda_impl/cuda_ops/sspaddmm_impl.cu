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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sspaddmm_impl.cuh"
#include "include/cuda_runtime.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void SparseAddSparseKernel(const S *input_indices, const T *input_values, const int64_t input_values_num,
                                      int64_t *y_indices, T *y_values, const int64_t y_values_num, const T *beta) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_values_num; pos += blockDim.x * gridDim.x) {
    y_values[pos] = input_values[pos] * (*beta);
    y_indices[pos] = static_cast<int64_t>(input_indices[pos]);
    y_indices[pos + y_values_num] = static_cast<int64_t>(input_indices[pos + input_values_num]);
  }
  return;
}

template <typename T, typename S>
__global__ void SparseMulDenseKernel(const int64_t input_values_num, const S *mat1_indices, const T *mat1_values,
                                     const int64_t mat2_col, const int64_t y_values_num, T *y_values,
                                     int64_t *y_indices, const int64_t mat1_values_num, const T *mat2,
                                     const int64_t *index) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < mat1_values_num * mat2_col;
    pos += blockDim.x * gridDim.x) {
      int64_t indices_pos = static_cast<int64_t>(pos) / mat2_col;
      int64_t j = static_cast<int64_t>(pos) % mat2_col;
      S col = mat1_indices[indices_pos + mat1_values_num];
      S idx = index[indices_pos];
      T value = mat1_values[indices_pos] * mat2[col * mat2_col + j];
      y_indices[idx * mat2_col + j + input_values_num] = static_cast<int64_t>(mat1_indices[indices_pos]);
      y_indices[idx * mat2_col + j + y_values_num + input_values_num] = j;
      MsAtomicAdd(&y_values[idx * mat2_col + j + input_values_num], value);
  }
  return;
}

template <typename T>
__global__ void MulAlphaKernel(T *y_values, const int64_t size, const int64_t input_values_num, const T *alpha) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    y_values[pos + input_values_num] *= (*alpha);
  }
  return;
}

template <typename T, typename S>
void CalSparseAddSparse(const S *input_indices, const T *input_values, const int64_t input_values_num,
                        int64_t *y_indices, T *y_values, const int64_t y_values_num, const T *beta,
                        const uint32_t &device_id, cudaStream_t cuda_stream) {
  SparseAddSparseKernel<<<CUDA_BLOCKS(device_id, input_values_num),
                          CUDA_THREADS(device_id), 0, cuda_stream>>>(input_indices, input_values, input_values_num,
                                                                     y_indices, y_values, y_values_num, beta);
}

template <typename T, typename S>
void CalSparseMulDense(const S *mat1_indices, const T *mat1_values, const int64_t mat1_values_num, const T *mat2,
                       int64_t *y_indices, T *y_values, const int64_t y_values_num,
                       const int64_t mat2_col, const int64_t input_values_num, const T *alpha, int64_t *index,
                       const uint32_t &device_id, cudaStream_t cuda_stream) {
  SparseMulDenseKernel<<<CUDA_BLOCKS(device_id, mat1_values_num * mat2_col),
                         CUDA_THREADS(device_id), 0, cuda_stream>>>(input_values_num, mat1_indices, mat1_values,
                                                                    mat2_col, y_values_num, y_values, y_indices,
                                                                    mat1_values_num, mat2, index);
  int64_t size = y_values_num - input_values_num;
  MulAlphaKernel<<<CUDA_BLOCKS(device_id, size),
                   CUDA_THREADS(device_id), 0, cuda_stream>>>(y_values, size, input_values_num, alpha);
}

#define GPU_SSPADDMM_EXPORT_REGISTER(T, S)                                                                            \
  template CUDA_LIB_EXPORT void CalSparseAddSparse<T, S>(const S *input_indices, const T *input_values,               \
                                                         const int64_t input_values_num, int64_t *y_indices,          \
                                                         T *y_values, const int64_t y_values_num, const T *beta,      \
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);        \
  template CUDA_LIB_EXPORT void CalSparseMulDense<T, S>(const S *mat1_indices, const T *mat1_values,                  \
                                                        const int64_t mat1_values_num, const T *mat2,                 \
                                                        int64_t *y_indices, T *y_values, const int64_t y_values_num,  \
                                                        const int64_t mat2_col, const int64_t input_values_num,       \
                                                        const T *alpha, int64_t *index, const uint32_t &device_id,    \
                                                        cudaStream_t cuda_stream);


GPU_SSPADDMM_EXPORT_REGISTER(int8_t, int)
GPU_SSPADDMM_EXPORT_REGISTER(int16_t, int)
GPU_SSPADDMM_EXPORT_REGISTER(int, int)
GPU_SSPADDMM_EXPORT_REGISTER(int64_t, int)
GPU_SSPADDMM_EXPORT_REGISTER(uint8_t, int)
GPU_SSPADDMM_EXPORT_REGISTER(uint16_t, int)
GPU_SSPADDMM_EXPORT_REGISTER(uint, int)
GPU_SSPADDMM_EXPORT_REGISTER(uint64_t, int)
GPU_SSPADDMM_EXPORT_REGISTER(half, int)
GPU_SSPADDMM_EXPORT_REGISTER(float, int)
GPU_SSPADDMM_EXPORT_REGISTER(double, int)


GPU_SSPADDMM_EXPORT_REGISTER(int8_t, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(int16_t, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(int, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(int64_t, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(uint8_t, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(uint16_t, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(uint, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(uint64_t, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(half, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(float, int64_t)
GPU_SSPADDMM_EXPORT_REGISTER(double, int64_t)
