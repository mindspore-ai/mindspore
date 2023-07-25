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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "softmarginloss_grad_impl.cuh"

template <typename T>
__global__ void SoftMarginLossGradReductionNoneKernel(const T *prediction, const T *target, const T *dout,
                                                      const size_t input_size, const T norm, T *gradient) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    gradient[i] = norm * target[i] * exp(-target[i] * prediction[i]) /
                  (static_cast<T>(1) + exp(-target[i] * prediction[i])) * dout[i];
  }
  return;
}

template <>
__global__ void SoftMarginLossGradReductionNoneKernel(const half *prediction, const half *target, const half *dout,
                                                      const size_t input_size, const half norm, half *gradient) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    half multiply = (target[i] * prediction[i]);
    gradient[i] = (multiply > __float2half(0))
                    ? (norm * target[i] * hexp(-multiply) / (__float2half(1) + hexp(-multiply)) * dout[i])
                    : (norm * target[i] / (__float2half(1) + hexp(multiply)) * dout[i]);
  }
  return;
}

template <typename T>
__global__ void SoftMarginLossGradReductionOtherKernel(const T *prediction, const T *target, const T *dout,
                                                       const size_t input_size, const T norm, T *gradient) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    gradient[i] = norm * target[i] * exp(-target[i] * prediction[i]) /
                  (static_cast<T>(1) + exp(-target[i] * prediction[i])) * dout[0];
  }
  return;
}

template <>
__global__ void SoftMarginLossGradReductionOtherKernel(const half *prediction, const half *target, const half *dout,
                                                       const size_t input_size, const half norm, half *gradient) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    half multiply = (target[i] * prediction[i]);
    gradient[i] = (multiply > __float2half(0))
                    ? (norm * target[i] * hexp(-multiply) / (__float2half(1) + hexp(-multiply)) * dout[0])
                    : (norm * target[i] / (__float2half(1) + hexp(multiply)) * dout[0]);
  }
  return;
}

template <typename T>
cudaError_t SoftMarginLossGrad(const T *prediction, const T *target, const T *dout, const size_t input_size,
                               const T norm, const ReductionMode &reduction, T *gradient, const uint32_t &device_id,
                               cudaStream_t cuda_stream) {
  if (reduction == ReductionMode::kNone) {
    SoftMarginLossGradReductionNoneKernel<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0,
                                            cuda_stream>>>(prediction, target, dout, input_size, norm, gradient);
  } else {
    SoftMarginLossGradReductionOtherKernel<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0,
                                             cuda_stream>>>(prediction, target, dout, input_size, norm, gradient);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t SoftMarginLossGrad(const float *prediction, const float *target, const float *dout,
                                                        const size_t input_size, const float norm,
                                                        const ReductionMode &reduction, float *gradient,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t SoftMarginLossGrad(const half *prediction, const half *target, const half *dout,
                                                        const size_t input_size, const half norm,
                                                        const ReductionMode &reduction, half *gradient,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t SoftMarginLossGrad(const double *prediction, const double *target,
                                                        const double *dout, const size_t input_size, const double norm,
                                                        const ReductionMode &reduction, double *gradient,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
