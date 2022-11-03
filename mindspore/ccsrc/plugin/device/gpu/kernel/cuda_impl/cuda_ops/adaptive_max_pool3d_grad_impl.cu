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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_max_pool3d_grad_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void AdaptiveMaxPool3DGradKernel(const T *input_grad, const S *input_argmax, const int output_stride,
                                            const int argmax_stride, const int batch, T *output_data) {
  for (size_t n = blockIdx.x * blockDim.x + threadIdx.x; n < batch; n += blockDim.x * gridDim.x) {
    for (int64_t i = 0; i < argmax_stride; ++i) {
      int32_t maxp = input_argmax[i + n * argmax_stride] + n * output_stride;
      MsAtomicAdd(output_data + static_cast<int>(maxp), input_grad[i + n * argmax_stride]);
    }
  }
  return;
}

template <typename T, typename S>
void CalAdaptiveMaxPool3DGrad(const T *input_grad, const S *input_argmax, const int output_stride,
                              const int argmax_stride, const int batch, T *output_data, const uint32_t &device_id,
                              cudaStream_t cuda_stream) {
  AdaptiveMaxPool3DGradKernel<<<CUDA_BLOCKS(device_id, batch), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_grad, input_argmax, output_stride, argmax_stride, batch, output_data);
}

template CUDA_LIB_EXPORT void CalAdaptiveMaxPool3DGrad<half, int>(const half *input_grad, const int *input_argmax,
                                                                  const int output_stride, const int argmax_stride,
                                                                  const int batch, half *output_data,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalAdaptiveMaxPool3DGrad<float, int>(const float *input_grad, const int *input_argmax,
                                                                   const int output_stride, const int argmax_stride,
                                                                   const int batch, float *output_data,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalAdaptiveMaxPool3DGrad<double, int>(const double *input_grad, const int *input_argmax,
                                                                    const int output_stride, const int argmax_stride,
                                                                    const int batch, double *output_data,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalAdaptiveMaxPool3DGrad<half, int64_t>(
  const half *input_grad, const int64_t *input_argmax, const int output_stride, const int argmax_stride,
  const int batch, half *output_data, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalAdaptiveMaxPool3DGrad<float, int64_t>(
  const float *input_grad, const int64_t *input_argmax, const int output_stride, const int argmax_stride,
  const int batch, float *output_data, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalAdaptiveMaxPool3DGrad<double, int64_t>(
  const double *input_grad, const int64_t *input_argmax, const int output_stride, const int argmax_stride,
  const int batch, double *output_data, const uint32_t &device_id, cudaStream_t cuda_stream);
