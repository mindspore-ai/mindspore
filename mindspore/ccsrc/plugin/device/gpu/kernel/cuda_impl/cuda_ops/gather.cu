/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "include/cuda_fp16.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
__global__ void GatherKernel(const T *input, const S *index, T *output, const size_t dim_before_axis,
                             const size_t dim_at_axis_input, const size_t dim_at_axis_output,
                             const size_t dim_after_axis) {
  size_t num = dim_before_axis * dim_at_axis_output * dim_after_axis;
  size_t i, k;
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num; id += blockDim.x * gridDim.x) {
    i = id / (dim_at_axis_output * dim_after_axis);
    k = id % dim_after_axis;

    S j = index[id];
    if (j < 0) {
      j += static_cast<S>(dim_at_axis_input);
    }
    CUDA_KERNEL_ASSERT(j >= 0);
    size_t j_read = static_cast<size_t>(j);
    CUDA_KERNEL_ASSERT(j_read < dim_at_axis_input);
    size_t read_id = i * dim_at_axis_input * dim_after_axis + j_read * dim_after_axis + k;
    output[id] = input[read_id];
  }
  return;
}
template <typename T, typename S>
cudaError_t Gather(const T *input, const S *index, T *output, const size_t dim_before_axis,
                   const size_t dim_at_axis_input, const size_t dim_at_axis_output, const size_t dim_after_axis,
                   cudaStream_t stream, uint32_t device_id) {
  size_t size = dim_before_axis * dim_at_axis_output * dim_after_axis;
  GatherKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(
    input, index, output, dim_before_axis, dim_at_axis_input, dim_at_axis_output, dim_after_axis);
  CHECK_CUDA_LAUNCH_SUCCESS();
  return cudaSuccess;
}

template CUDA_LIB_EXPORT cudaError_t Gather<Complex<double>, int>(const Complex<double> *input, const int *index,
                                                                  Complex<double> *output, const size_t dim_before_axis,
                                                                  const size_t dim_at_axis_input,
                                                                  const size_t dim_at_axis_output,
                                                                  const size_t dim_after_axis, cudaStream_t stream,
                                                                  uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<Complex<double>, int64_t>(
  const Complex<double> *input, const int64_t *index, Complex<double> *output, const size_t dim_before_axis,
  const size_t dim_at_axis_input, const size_t dim_at_axis_output, const size_t dim_after_axis, cudaStream_t stream,
  uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<Complex<float>, int>(const Complex<float> *input, const int *index,
                                                                 Complex<float> *output, const size_t dim_before_axis,
                                                                 const size_t dim_at_axis_input,
                                                                 const size_t dim_at_axis_output,
                                                                 const size_t dim_after_axis, cudaStream_t stream,
                                                                 uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<Complex<float>, int64_t>(
  const Complex<float> *input, const int64_t *index, Complex<float> *output, const size_t dim_before_axis,
  const size_t dim_at_axis_input, const size_t dim_at_axis_output, const size_t dim_after_axis, cudaStream_t stream,
  uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<double, int>(const double *input, const int *index, double *output,
                                                         const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                         const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                         cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<double, int64_t>(const double *input, const int64_t *index, double *output,
                                                             const size_t dim_before_axis,
                                                             const size_t dim_at_axis_input,
                                                             const size_t dim_at_axis_output,
                                                             const size_t dim_after_axis, cudaStream_t stream,
                                                             uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<float, int>(const float *input, const int *index, float *output,
                                                        const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                        const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                        cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<float, int64_t>(
  const float *input, const int64_t *index, float *output, const size_t dim_before_axis, const size_t dim_at_axis_input,
  const size_t dim_at_axis_output, const size_t dim_after_axis, cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<half, int>(const half *input, const int *index, half *output,
                                                       const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                       const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                       cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<half, int64_t>(const half *input, const int64_t *index, half *output,
                                                           const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                           const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                           cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<int64_t, int>(const int64_t *input, const int *index, int64_t *output,
                                                          const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                          const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                          cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<int64_t, int64_t>(const int64_t *input, const int64_t *index,
                                                              int64_t *output, const size_t dim_before_axis,
                                                              const size_t dim_at_axis_input,
                                                              const size_t dim_at_axis_output,
                                                              const size_t dim_after_axis, cudaStream_t stream,
                                                              uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<int, int>(const int *input, const int *index, int *output,
                                                      const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                      const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                      cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<int, int64_t>(const int *input, const int64_t *index, int *output,
                                                          const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                          const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                          cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<int16_t, int>(const int16_t *input, const int *index, int16_t *output,
                                                          const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                          const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                          cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<int16_t, int64_t>(const int16_t *input, const int64_t *index,
                                                              int16_t *output, const size_t dim_before_axis,
                                                              const size_t dim_at_axis_input,
                                                              const size_t dim_at_axis_output,
                                                              const size_t dim_after_axis, cudaStream_t stream,
                                                              uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<int8_t, int>(const int8_t *input, const int *index, int8_t *output,
                                                         const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                         const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                         cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<int8_t, int64_t>(const int8_t *input, const int64_t *index, int8_t *output,
                                                             const size_t dim_before_axis,
                                                             const size_t dim_at_axis_input,
                                                             const size_t dim_at_axis_output,
                                                             const size_t dim_after_axis, cudaStream_t stream,
                                                             uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<unsigned char, int>(const unsigned char *input, const int *index,
                                                                unsigned char *output, const size_t dim_before_axis,
                                                                const size_t dim_at_axis_input,
                                                                const size_t dim_at_axis_output,
                                                                const size_t dim_after_axis, cudaStream_t stream,
                                                                uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<unsigned char, int64_t>(const unsigned char *input, const int64_t *index,
                                                                    unsigned char *output, const size_t dim_before_axis,
                                                                    const size_t dim_at_axis_input,
                                                                    const size_t dim_at_axis_output,
                                                                    const size_t dim_after_axis, cudaStream_t stream,
                                                                    uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<bool, int>(const bool *input, const int *index, bool *output,
                                                       const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                       const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                       cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<bool, int64_t>(const bool *input, const int64_t *index, bool *output,
                                                           const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                           const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                           cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<uint16_t, int>(const uint16_t *input, const int *index, uint16_t *output,
                                                           const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                           const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                           cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<uint16_t, int64_t>(const uint16_t *input, const int64_t *index,
                                                               uint16_t *output, const size_t dim_before_axis,
                                                               const size_t dim_at_axis_input,
                                                               const size_t dim_at_axis_output,
                                                               const size_t dim_after_axis, cudaStream_t stream,
                                                               uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<uint32_t, int>(const uint32_t *input, const int *index, uint32_t *output,
                                                           const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                           const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                           cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<uint32_t, int64_t>(const uint32_t *input, const int64_t *index,
                                                               uint32_t *output, const size_t dim_before_axis,
                                                               const size_t dim_at_axis_input,
                                                               const size_t dim_at_axis_output,
                                                               const size_t dim_after_axis, cudaStream_t stream,
                                                               uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<uint64_t, int>(const uint64_t *input, const int *index, uint64_t *output,
                                                           const size_t dim_before_axis, const size_t dim_at_axis_input,
                                                           const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                           cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT cudaError_t Gather<uint64_t, int64_t>(const uint64_t *input, const int64_t *index,
                                                               uint64_t *output, const size_t dim_before_axis,
                                                               const size_t dim_at_axis_input,
                                                               const size_t dim_at_axis_output,
                                                               const size_t dim_after_axis, cudaStream_t stream,
                                                               uint32_t device_id);
