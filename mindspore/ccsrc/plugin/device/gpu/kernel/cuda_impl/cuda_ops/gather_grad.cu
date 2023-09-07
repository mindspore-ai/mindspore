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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather_grad.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
__global__ void GatherGradKernel(const size_t num, const T *index, const S *grad, S *output,
                                 const size_t dim_before_axis, const size_t dim_at_axis_index,
                                 const size_t dim_at_axis_output, const size_t dim_after_axis) {
  size_t i, k;

  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num; id += blockDim.x * gridDim.x) {
    i = id / (dim_at_axis_index * dim_after_axis);
    k = id % dim_after_axis;

    T j = index[id];
    if (j < 0) {
      j += static_cast<T>(dim_at_axis_output);
    }
    CUDA_KERNEL_ASSERT(j >= 0);
    size_t j_read = static_cast<size_t>(j);
    CUDA_KERNEL_ASSERT(j_read < dim_at_axis_output);
    size_t read_id = i * dim_at_axis_output * dim_after_axis + j_read * dim_after_axis + k;
    MsAtomicAdd(output + read_id, grad[id]);
  }
  return;
}

template <typename T, typename S>
cudaError_t GatherGrad(const T *index, const S *grad, S *output, const size_t dim_before_axis,
                       const size_t dim_at_axis_index, const size_t dim_at_axis_output, const size_t dim_after_axis,
                       cudaStream_t stream) {
  size_t size = dim_before_axis * dim_at_axis_index * dim_after_axis;
  GatherGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(size, index, grad, output, dim_before_axis,
                                                                 dim_at_axis_index, dim_at_axis_output, dim_after_axis);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, Complex<double>>(
  const int *index, const Complex<double> *grad, Complex<double> *output, const size_t dim_before_axis,
  const size_t dim_at_axis_index, const size_t dim_at_axis_output, const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, Complex<double>>(
  const int64_t *index, const Complex<double> *grad, Complex<double> *output, const size_t dim_before_axis,
  const size_t dim_at_axis_index, const size_t dim_at_axis_output, const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, Complex<float>>(
  const int *index, const Complex<float> *grad, Complex<float> *output, const size_t dim_before_axis,
  const size_t dim_at_axis_index, const size_t dim_at_axis_output, const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, Complex<float>>(
  const int64_t *index, const Complex<float> *grad, Complex<float> *output, const size_t dim_before_axis,
  const size_t dim_at_axis_index, const size_t dim_at_axis_output, const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, double>(const int *index, const double *grad, double *output,
                                                             const size_t dim_before_axis,
                                                             const size_t dim_at_axis_index,
                                                             const size_t dim_at_axis_output,
                                                             const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, double>(const int64_t *index, const double *grad,
                                                                 double *output, const size_t dim_before_axis,
                                                                 const size_t dim_at_axis_index,
                                                                 const size_t dim_at_axis_output,
                                                                 const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, float>(const int *index, const float *grad, float *output,
                                                            const size_t dim_before_axis,
                                                            const size_t dim_at_axis_index,
                                                            const size_t dim_at_axis_output,
                                                            const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, float>(const int64_t *index, const float *grad, float *output,
                                                                const size_t dim_before_axis,
                                                                const size_t dim_at_axis_index,
                                                                const size_t dim_at_axis_output,
                                                                const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, half>(const int *index, const half *grad, half *output,
                                                           const size_t dim_before_axis, const size_t dim_at_axis_index,
                                                           const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                           cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, half>(const int64_t *index, const half *grad, half *output,
                                                               const size_t dim_before_axis,
                                                               const size_t dim_at_axis_index,
                                                               const size_t dim_at_axis_output,
                                                               const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, int>(const int *index, const int *grad, int *output,
                                                          const size_t dim_before_axis, const size_t dim_at_axis_index,
                                                          const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, int>(const int64_t *index, const int *grad, int *output,
                                                              const size_t dim_before_axis,
                                                              const size_t dim_at_axis_index,
                                                              const size_t dim_at_axis_output,
                                                              const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, int8_t>(const int *index, const int8_t *grad, int8_t *output,
                                                             const size_t dim_before_axis,
                                                             const size_t dim_at_axis_index,
                                                             const size_t dim_at_axis_output,
                                                             const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, int8_t>(const int64_t *index, const int8_t *grad,
                                                                 int8_t *output, const size_t dim_before_axis,
                                                                 const size_t dim_at_axis_index,
                                                                 const size_t dim_at_axis_output,
                                                                 const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, int16_t>(const int *index, const int16_t *grad, int16_t *output,
                                                              const size_t dim_before_axis,
                                                              const size_t dim_at_axis_index,
                                                              const size_t dim_at_axis_output,
                                                              const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, int16_t>(const int64_t *index, const int16_t *grad,
                                                                  int16_t *output, const size_t dim_before_axis,
                                                                  const size_t dim_at_axis_index,
                                                                  const size_t dim_at_axis_output,
                                                                  const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, int64_t>(const int *index, const int64_t *grad, int64_t *output,
                                                              const size_t dim_before_axis,
                                                              const size_t dim_at_axis_index,
                                                              const size_t dim_at_axis_output,
                                                              const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, int64_t>(const int64_t *index, const int64_t *grad,
                                                                  int64_t *output, const size_t dim_before_axis,
                                                                  const size_t dim_at_axis_index,
                                                                  const size_t dim_at_axis_output,
                                                                  const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, unsigned char>(const int *index, const unsigned char *grad,
                                                                    unsigned char *output, const size_t dim_before_axis,
                                                                    const size_t dim_at_axis_index,
                                                                    const size_t dim_at_axis_output,
                                                                    const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, unsigned char>(
  const int64_t *index, const unsigned char *grad, unsigned char *output, const size_t dim_before_axis,
  const size_t dim_at_axis_index, const size_t dim_at_axis_output, const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, uint64_t>(const int *index, const uint64_t *grad, uint64_t *output,
                                                               const size_t dim_before_axis,
                                                               const size_t dim_at_axis_index,
                                                               const size_t dim_at_axis_output,
                                                               const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, uint64_t>(const int64_t *index, const uint64_t *grad,
                                                                   uint64_t *output, const size_t dim_before_axis,
                                                                   const size_t dim_at_axis_index,
                                                                   const size_t dim_at_axis_output,
                                                                   const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, uint32_t>(const int *index, const uint32_t *grad, uint32_t *output,
                                                               const size_t dim_before_axis,
                                                               const size_t dim_at_axis_index,
                                                               const size_t dim_at_axis_output,
                                                               const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, uint32_t>(const int64_t *index, const uint32_t *grad,
                                                                   uint32_t *output, const size_t dim_before_axis,
                                                                   const size_t dim_at_axis_index,
                                                                   const size_t dim_at_axis_output,
                                                                   const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, uint16_t>(const int *index, const uint16_t *grad, uint16_t *output,
                                                               const size_t dim_before_axis,
                                                               const size_t dim_at_axis_index,
                                                               const size_t dim_at_axis_output,
                                                               const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, uint16_t>(const int64_t *index, const uint16_t *grad,
                                                                   uint16_t *output, const size_t dim_before_axis,
                                                                   const size_t dim_at_axis_index,
                                                                   const size_t dim_at_axis_output,
                                                                   const size_t dim_after_axis, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int, bool>(const int *index, const bool *grad, bool *output,
                                                           const size_t dim_before_axis, const size_t dim_at_axis_index,
                                                           const size_t dim_at_axis_output, const size_t dim_after_axis,
                                                           cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherGrad<int64_t, bool>(const int64_t *index, const bool *grad, bool *output,
                                                               const size_t dim_before_axis,
                                                               const size_t dim_at_axis_index,
                                                               const size_t dim_at_axis_output,
                                                               const size_t dim_after_axis, cudaStream_t stream);
