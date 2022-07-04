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

#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_grad_grad_with_argmax_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T, typename I>
__global__ void MaxPoolGradGradWithArgmax(const T *grad, const I *argmax, const int input_stride,
                                          const int output_stride, const int output_elements, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (output_elements); pos += blockDim.x * gridDim.x) {
    const int posn = pos / output_stride;
    output[pos] = grad[posn * input_stride + argmax[pos]];
  }
}

template <typename T, typename I>
void CalMaxPoolGradGradWithArgmax(const T *grad, const I *argmax, const int batch, const int input_stride,
                                  const int output_stride, T *output, const uint32_t &device_id,
                                  cudaStream_t cuda_stream) {
  const int output_elements = batch * output_stride;
  MaxPoolGradGradWithArgmax<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    grad, argmax, input_stride, output_stride, output_elements, output);
}

template CUDA_LIB_EXPORT void CalMaxPoolGradGradWithArgmax<float, int32_t>(const float *grad, const int32_t *argmax,
                                                                           const int batch, const int input_stride,
                                                                           const int output_stride, float *output,
                                                                           const uint32_t &device_id,
                                                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolGradGradWithArgmax<float, int64_t>(const float *grad, const int64_t *argmax,
                                                                           const int batch, const int input_stride,
                                                                           const int output_stride, float *output,
                                                                           const uint32_t &device_id,
                                                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolGradGradWithArgmax<half, int32_t>(const half *grad, const int32_t *argmax,
                                                                          const int batch, const int input_stride,
                                                                          const int output_stride, half *output,
                                                                          const uint32_t &device_id,
                                                                          cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPoolGradGradWithArgmax<half, int64_t>(const half *grad, const int64_t *argmax,
                                                                          const int batch, const int input_stride,
                                                                          const int output_stride, half *output,
                                                                          const uint32_t &device_id,
                                                                          cudaStream_t cuda_stream);
