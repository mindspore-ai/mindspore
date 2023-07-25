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
#include "lu_unpack_grad_impl.cuh"

template <typename T>
__global__ void TrilExpendWidth(const int64_t size, T *l_grad_input, const int64_t matrix_L_height,
                                const int64_t matrix_L_width, T *l_grad_output, const int64_t lu_data_height,
                                const int64_t lu_data_width) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t matrix_size = lu_data_height * lu_data_width;
    int64_t idx_height = pos % matrix_size / lu_data_width;
    int64_t idx_width = pos % matrix_size % lu_data_width;
    if (idx_width >= idx_height) {
      l_grad_output[pos] = 0;
    } else {
      int64_t in_pos = ((pos / matrix_size) * matrix_L_height + idx_height) * matrix_L_width + idx_width;
      l_grad_output[pos] = l_grad_input[in_pos];
    }
  }
  return;
}

template <typename T>
__global__ void TrilLower(const int64_t size, T *l_grad_input, const int64_t matrix_L_height,
                          const int64_t matrix_L_width, T *l_grad_output, const int64_t lu_data_height,
                          const int64_t lu_data_width) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t matrix_size = lu_data_height * lu_data_width;
    int64_t idx_height = pos % matrix_size / lu_data_width;
    int64_t idx_width = pos % matrix_size % lu_data_width;
    if (idx_width >= idx_height) {
      l_grad_output[pos] = 0;
    } else {
      l_grad_output[pos] = l_grad_input[pos];
    }
  }
  return;
}

template <typename T>
__global__ void TriuExpendHeight(const int64_t size, T *u_grad_input, const int64_t matrix_U_height,
                                 const int64_t matrix_U_width, T *u_grad_output, const int64_t lu_data_height,
                                 const int64_t lu_data_width) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t matrix_size = lu_data_height * lu_data_width;
    int64_t idx_height = pos % matrix_size / lu_data_width;
    int64_t idx_width = pos % matrix_size % lu_data_width;
    if (idx_width >= idx_height) {
      int64_t in_pos = ((pos / matrix_size) * matrix_U_height + idx_height) * matrix_U_width + idx_width;
      u_grad_output[pos] = u_grad_input[in_pos];
    } else {
      u_grad_output[pos] = 0;
    }
  }
  return;
}

template <typename T>
__global__ void TriuUpper(const int64_t size, T *u_grad_input, const int64_t matrix_U_height,
                          const int64_t matrix_U_width, T *u_grad_output, const int64_t lu_data_height,
                          const int64_t lu_data_width) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t matrix_size = lu_data_height * lu_data_width;
    int64_t idx_height = pos % matrix_size / lu_data_width;
    int64_t idx_width = pos % matrix_size % lu_data_width;
    if (idx_height <= idx_width) {
      u_grad_output[pos] = u_grad_input[pos];
    } else {
      u_grad_output[pos] = 0;
    }
  }
  return;
}

template <typename T>
cudaError_t CalTrilExpendWidth(const int64_t size, T *l_grad_input, const int64_t matrix_L_height,
                               const int64_t matrix_L_width, T *l_grad_output, const int64_t lu_data_height,
                               const int64_t lu_data_width, const uint32_t &device_id, cudaStream_t cuda_stream) {
  TrilExpendWidth<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, l_grad_input, matrix_L_height, matrix_L_width, l_grad_output, lu_data_height, lu_data_width);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalTrilLower(const int64_t size, T *l_grad_input, const int64_t matrix_L_height,
                         const int64_t matrix_L_width, T *l_grad_output, const int64_t lu_data_height,
                         const int64_t lu_data_width, const uint32_t &device_id, cudaStream_t cuda_stream) {
  TrilLower<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, l_grad_input, matrix_L_height, matrix_L_width, l_grad_output, lu_data_height, lu_data_width);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalTriuExpendHeight(const int64_t size, T *u_grad_input, const int64_t matrix_U_height,
                                const int64_t matrix_U_width, T *u_grad_output, const int64_t lu_data_height,
                                const int64_t lu_data_width, const uint32_t &device_id, cudaStream_t cuda_stream) {
  TriuExpendHeight<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, u_grad_input, matrix_U_height, matrix_U_width, u_grad_output, lu_data_height, lu_data_width);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalTriuUpper(const int64_t size, T *u_grad_input, const int64_t matrix_U_height,
                         const int64_t matrix_U_width, T *u_grad_output, const int64_t lu_data_height,
                         const int64_t lu_data_width, const uint32_t &device_id, cudaStream_t cuda_stream) {
  TriuUpper<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, u_grad_input, matrix_U_height, matrix_U_width, u_grad_output, lu_data_height, lu_data_width);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalTrilExpendWidth<double>(const int64_t size, double *l_grad_input,
                                                                const int64_t matrix_L_height,
                                                                const int64_t matrix_L_width, double *l_grad_output,
                                                                const int64_t lu_data_height,
                                                                const int64_t lu_data_width, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilExpendWidth<float>(const int64_t size, float *l_grad_input,
                                                               const int64_t matrix_L_height,
                                                               const int64_t matrix_L_width, float *l_grad_output,
                                                               const int64_t lu_data_height,
                                                               const int64_t lu_data_width, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilExpendWidth<half>(const int64_t size, half *l_grad_input,
                                                              const int64_t matrix_L_height,
                                                              const int64_t matrix_L_width, half *l_grad_output,
                                                              const int64_t lu_data_height, const int64_t lu_data_width,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilExpendWidth<int64_t>(const int64_t size, int64_t *l_grad_input,
                                                                 const int64_t matrix_L_height,
                                                                 const int64_t matrix_L_width, int64_t *l_grad_output,
                                                                 const int64_t lu_data_height,
                                                                 const int64_t lu_data_width, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilExpendWidth<int32_t>(const int64_t size, int32_t *l_grad_input,
                                                                 const int64_t matrix_L_height,
                                                                 const int64_t matrix_L_width, int32_t *l_grad_output,
                                                                 const int64_t lu_data_height,
                                                                 const int64_t lu_data_width, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilExpendWidth<int16_t>(const int64_t size, int16_t *l_grad_input,
                                                                 const int64_t matrix_L_height,
                                                                 const int64_t matrix_L_width, int16_t *l_grad_output,
                                                                 const int64_t lu_data_height,
                                                                 const int64_t lu_data_width, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilExpendWidth<int8_t>(const int64_t size, int8_t *l_grad_input,
                                                                const int64_t matrix_L_height,
                                                                const int64_t matrix_L_width, int8_t *l_grad_output,
                                                                const int64_t lu_data_height,
                                                                const int64_t lu_data_width, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilExpendWidth<uint8_t>(const int64_t size, uint8_t *l_grad_input,
                                                                 const int64_t matrix_L_height,
                                                                 const int64_t matrix_L_width, uint8_t *l_grad_output,
                                                                 const int64_t lu_data_height,
                                                                 const int64_t lu_data_width, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalTrilLower<double>(const int64_t size, double *l_grad_input,
                                                          const int64_t matrix_L_height, const int64_t matrix_L_width,
                                                          double *l_grad_output, const int64_t lu_data_height,
                                                          const int64_t lu_data_width, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilLower<float>(const int64_t size, float *l_grad_input,
                                                         const int64_t matrix_L_height, const int64_t matrix_L_width,
                                                         float *l_grad_output, const int64_t lu_data_height,
                                                         const int64_t lu_data_width, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilLower<half>(const int64_t size, half *l_grad_input,
                                                        const int64_t matrix_L_height, const int64_t matrix_L_width,
                                                        half *l_grad_output, const int64_t lu_data_height,
                                                        const int64_t lu_data_width, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilLower<int64_t>(const int64_t size, int64_t *l_grad_input,
                                                           const int64_t matrix_L_height, const int64_t matrix_L_width,
                                                           int64_t *l_grad_output, const int64_t lu_data_height,
                                                           const int64_t lu_data_width, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilLower<int32_t>(const int64_t size, int32_t *l_grad_input,
                                                           const int64_t matrix_L_height, const int64_t matrix_L_width,
                                                           int32_t *l_grad_output, const int64_t lu_data_height,
                                                           const int64_t lu_data_width, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilLower<int16_t>(const int64_t size, int16_t *l_grad_input,
                                                           const int64_t matrix_L_height, const int64_t matrix_L_width,
                                                           int16_t *l_grad_output, const int64_t lu_data_height,
                                                           const int64_t lu_data_width, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilLower<int8_t>(const int64_t size, int8_t *l_grad_input,
                                                          const int64_t matrix_L_height, const int64_t matrix_L_width,
                                                          int8_t *l_grad_output, const int64_t lu_data_height,
                                                          const int64_t lu_data_width, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTrilLower<uint8_t>(const int64_t size, uint8_t *l_grad_input,
                                                           const int64_t matrix_L_height, const int64_t matrix_L_width,
                                                           uint8_t *l_grad_output, const int64_t lu_data_height,
                                                           const int64_t lu_data_width, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalTriuExpendHeight<double>(const int64_t size, double *u_grad_input,
                                                                 const int64_t matrix_U_height,
                                                                 const int64_t matrix_U_width, double *u_grad_output,
                                                                 const int64_t lu_data_height,
                                                                 const int64_t lu_data_width, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuExpendHeight<float>(const int64_t size, float *u_grad_input,
                                                                const int64_t matrix_U_height,
                                                                const int64_t matrix_U_width, float *u_grad_output,
                                                                const int64_t lu_data_height,
                                                                const int64_t lu_data_width, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuExpendHeight<half>(const int64_t size, half *u_grad_input,
                                                               const int64_t matrix_U_height,
                                                               const int64_t matrix_U_width, half *u_grad_output,
                                                               const int64_t lu_data_height,
                                                               const int64_t lu_data_width, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuExpendHeight<int64_t>(const int64_t size, int64_t *u_grad_input,
                                                                  const int64_t matrix_U_height,
                                                                  const int64_t matrix_U_width, int64_t *u_grad_output,
                                                                  const int64_t lu_data_height,
                                                                  const int64_t lu_data_width,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuExpendHeight<int32_t>(const int64_t size, int32_t *u_grad_input,
                                                                  const int64_t matrix_U_height,
                                                                  const int64_t matrix_U_width, int32_t *u_grad_output,
                                                                  const int64_t lu_data_height,
                                                                  const int64_t lu_data_width,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuExpendHeight<int16_t>(const int64_t size, int16_t *u_grad_input,
                                                                  const int64_t matrix_U_height,
                                                                  const int64_t matrix_U_width, int16_t *u_grad_output,
                                                                  const int64_t lu_data_height,
                                                                  const int64_t lu_data_width,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuExpendHeight<int8_t>(const int64_t size, int8_t *u_grad_input,
                                                                 const int64_t matrix_U_height,
                                                                 const int64_t matrix_U_width, int8_t *u_grad_output,
                                                                 const int64_t lu_data_height,
                                                                 const int64_t lu_data_width, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuExpendHeight<uint8_t>(const int64_t size, uint8_t *u_grad_input,
                                                                  const int64_t matrix_U_height,
                                                                  const int64_t matrix_U_width, uint8_t *u_grad_output,
                                                                  const int64_t lu_data_height,
                                                                  const int64_t lu_data_width,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalTriuUpper<double>(const int64_t size, double *u_grad_input,
                                                          const int64_t matrix_U_height, const int64_t matrix_U_width,
                                                          double *u_grad_output, const int64_t lu_data_height,
                                                          const int64_t lu_data_width, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuUpper<float>(const int64_t size, float *u_grad_input,
                                                         const int64_t matrix_U_height, const int64_t matrix_U_width,
                                                         float *u_grad_output, const int64_t lu_data_height,
                                                         const int64_t lu_data_width, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuUpper<half>(const int64_t size, half *u_grad_input,
                                                        const int64_t matrix_U_height, const int64_t matrix_U_width,
                                                        half *u_grad_output, const int64_t lu_data_height,
                                                        const int64_t lu_data_width, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuUpper<int64_t>(const int64_t size, int64_t *u_grad_input,
                                                           const int64_t matrix_U_height, const int64_t matrix_U_width,
                                                           int64_t *u_grad_output, const int64_t lu_data_height,
                                                           const int64_t lu_data_width, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuUpper<int32_t>(const int64_t size, int32_t *u_grad_input,
                                                           const int64_t matrix_U_height, const int64_t matrix_U_width,
                                                           int32_t *u_grad_output, const int64_t lu_data_height,
                                                           const int64_t lu_data_width, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuUpper<int16_t>(const int64_t size, int16_t *u_grad_input,
                                                           const int64_t matrix_U_height, const int64_t matrix_U_width,
                                                           int16_t *u_grad_output, const int64_t lu_data_height,
                                                           const int64_t lu_data_width, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuUpper<int8_t>(const int64_t size, int8_t *u_grad_input,
                                                          const int64_t matrix_U_height, const int64_t matrix_U_width,
                                                          int8_t *u_grad_output, const int64_t lu_data_height,
                                                          const int64_t lu_data_width, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTriuUpper<uint8_t>(const int64_t size, uint8_t *u_grad_input,
                                                           const int64_t matrix_U_height, const int64_t matrix_U_width,
                                                           uint8_t *u_grad_output, const int64_t lu_data_height,
                                                           const int64_t lu_data_width, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
