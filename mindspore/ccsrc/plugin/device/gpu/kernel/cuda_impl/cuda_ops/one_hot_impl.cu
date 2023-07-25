/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "one_hot_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "include/cuda_fp16.h"

template <typename T, typename S>
__global__ void OneHotKernel(size_t size, const S *indices, size_t depth, const T *on_value, const T *off_value,
                             size_t left_dim_size, size_t right_dim_size, T *output) {
  T on_v = *on_value;
  T off_v = *off_value;
  for (size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < size;
       thread_idx += blockDim.x * gridDim.x) {
    if (thread_idx < size) {
      size_t left_idx = (thread_idx / (depth * right_dim_size)) % left_dim_size;
      size_t d_idx = thread_idx / right_dim_size % depth;
      size_t right_idx = thread_idx % right_dim_size;
      size_t input_idx = left_idx * right_dim_size + right_idx;
      size_t output_idx = left_idx * depth * right_dim_size + d_idx * right_dim_size + right_idx;
      if (indices[input_idx] == d_idx) {
        output[output_idx] = on_v;
      } else {
        output[output_idx] = off_v;
      }
    }
  }
}
template <typename T, typename S>
cudaError_t OneHot(const S *indices, size_t depth, const T *on_value, const T *off_value, size_t left_dim_size,
                   size_t right_dim_size, T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t size = left_dim_size * depth * right_dim_size;
  OneHotKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, indices, depth, on_value, off_value, left_dim_size, right_dim_size, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t OneHot<uint8_t, int>(const int *indices, size_t depth, const uint8_t *on_value,
                                                          const uint8_t *off_value, size_t left_dim_size,
                                                          size_t right_dim_size, uint8_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<uint8_t, int64_t>(const int64_t *indices, size_t depth,
                                                              const uint8_t *on_value, const uint8_t *off_value,
                                                              size_t left_dim_size, size_t right_dim_size,
                                                              uint8_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<uint16_t, int>(const int *indices, size_t depth, const uint16_t *on_value,
                                                           const uint16_t *off_value, size_t left_dim_size,
                                                           size_t right_dim_size, uint16_t *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<uint16_t, int64_t>(const int64_t *indices, size_t depth,
                                                               const uint16_t *on_value, const uint16_t *off_value,
                                                               size_t left_dim_size, size_t right_dim_size,
                                                               uint16_t *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<uint32_t, int>(const int *indices, size_t depth, const uint32_t *on_value,
                                                           const uint32_t *off_value, size_t left_dim_size,
                                                           size_t right_dim_size, uint32_t *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<uint32_t, int64_t>(const int64_t *indices, size_t depth,
                                                               const uint32_t *on_value, const uint32_t *off_value,
                                                               size_t left_dim_size, size_t right_dim_size,
                                                               uint32_t *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<uint64_t, int>(const int *indices, size_t depth, const uint64_t *on_value,
                                                           const uint64_t *off_value, size_t left_dim_size,
                                                           size_t right_dim_size, uint64_t *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<uint64_t, int64_t>(const int64_t *indices, size_t depth,
                                                               const uint64_t *on_value, const uint64_t *off_value,
                                                               size_t left_dim_size, size_t right_dim_size,
                                                               uint64_t *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<int8_t, int>(const int *indices, size_t depth, const int8_t *on_value,
                                                         const int8_t *off_value, size_t left_dim_size,
                                                         size_t right_dim_size, int8_t *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<int8_t, int64_t>(const int64_t *indices, size_t depth,
                                                             const int8_t *on_value, const int8_t *off_value,
                                                             size_t left_dim_size, size_t right_dim_size,
                                                             int8_t *output, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<int16_t, int>(const int *indices, size_t depth, const int16_t *on_value,
                                                          const int16_t *off_value, size_t left_dim_size,
                                                          size_t right_dim_size, int16_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<int16_t, int64_t>(const int64_t *indices, size_t depth,
                                                              const int16_t *on_value, const int16_t *off_value,
                                                              size_t left_dim_size, size_t right_dim_size,
                                                              int16_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<int32_t, int>(const int *indices, size_t depth, const int32_t *on_value,
                                                          const int32_t *off_value, size_t left_dim_size,
                                                          size_t right_dim_size, int32_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<int32_t, int64_t>(const int64_t *indices, size_t depth,
                                                              const int32_t *on_value, const int32_t *off_value,
                                                              size_t left_dim_size, size_t right_dim_size,
                                                              int32_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<int64_t, int>(const int *indices, size_t depth, const int64_t *on_value,
                                                          const int64_t *off_value, size_t left_dim_size,
                                                          size_t right_dim_size, int64_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<int64_t, int64_t>(const int64_t *indices, size_t depth,
                                                              const int64_t *on_value, const int64_t *off_value,
                                                              size_t left_dim_size, size_t right_dim_size,
                                                              int64_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<half, int>(const int *indices, size_t depth, const half *on_value,
                                                       const half *off_value, size_t left_dim_size,
                                                       size_t right_dim_size, half *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<half, int64_t>(const int64_t *indices, size_t depth, const half *on_value,
                                                           const half *off_value, size_t left_dim_size,
                                                           size_t right_dim_size, half *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<float, int>(const int *indices, size_t depth, const float *on_value,
                                                        const float *off_value, size_t left_dim_size,
                                                        size_t right_dim_size, float *output, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<float, int64_t>(const int64_t *indices, size_t depth, const float *on_value,
                                                            const float *off_value, size_t left_dim_size,
                                                            size_t right_dim_size, float *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<double, int>(const int *indices, size_t depth, const double *on_value,
                                                         const double *off_value, size_t left_dim_size,
                                                         size_t right_dim_size, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<double, int64_t>(const int64_t *indices, size_t depth,
                                                             const double *on_value, const double *off_value,
                                                             size_t left_dim_size, size_t right_dim_size,
                                                             double *output, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<bool, int>(const int *indices, size_t depth, const bool *on_value,
                                                       const bool *off_value, size_t left_dim_size,
                                                       size_t right_dim_size, bool *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<bool, int64_t>(const int64_t *indices, size_t depth, const bool *on_value,
                                                           const bool *off_value, size_t left_dim_size,
                                                           size_t right_dim_size, bool *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<Complex<float>, int>(const int *indices, size_t depth,
                                                                 const Complex<float> *on_value,
                                                                 const Complex<float> *off_value, size_t left_dim_size,
                                                                 size_t right_dim_size, Complex<float> *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<Complex<float>, int64_t>(const int64_t *indices, size_t depth,
                                                                     const Complex<float> *on_value,
                                                                     const Complex<float> *off_value,
                                                                     size_t left_dim_size, size_t right_dim_size,
                                                                     Complex<float> *output, const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t OneHot<Complex<double>, int>(const int *indices, size_t depth,
                                                                  const Complex<double> *on_value,
                                                                  const Complex<double> *off_value,
                                                                  size_t left_dim_size, size_t right_dim_size,
                                                                  Complex<double> *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
OneHot<Complex<double>, int64_t>(const int64_t *indices, size_t depth, const Complex<double> *on_value,
                                 const Complex<double> *off_value, size_t left_dim_size, size_t right_dim_size,
                                 Complex<double> *output, const uint32_t &device_id, cudaStream_t cuda_stream);
