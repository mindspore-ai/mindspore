/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "momentum_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementswise_op_impl.cuh"

template <typename T, typename S, typename G>
struct MomentumUpdateVariableFunctor {
  const S *learning_rate_;
  const S *momentum_;
  MomentumUpdateVariableFunctor(const S *learning_rate, const S *momentum) {
    this->learning_rate_ = learning_rate;
    this->momentum_ = momentum;
  }
  __device__ __forceinline__ void operator()(T *variable, T *accumulation, const G *gradient) const {
    accumulation[0] = momentum_[0] * accumulation[0] + gradient[0];
    variable[0] -= learning_rate_[0] * accumulation[0];
  }
};

template <typename T, typename S, typename G>
struct MomentumUpdateVariableWithNesterovFunctor {
  const S *learning_rate_;
  const S *momentum_;
  MomentumUpdateVariableWithNesterovFunctor(const S *learning_rate, const S *momentum) {
    this->learning_rate_ = learning_rate;
    this->momentum_ = momentum;
  }
  __device__ __forceinline__ void operator()(T *variable, T *accumulation, const G *gradient) const {
    accumulation[0] = momentum_[0] * accumulation[0] + gradient[0];
    variable[0] -= gradient[0] * learning_rate_[0] + accumulation[0] * momentum_[0] * learning_rate_[0];
  }
};

template <>
struct MomentumUpdateVariableFunctor<half, float, half> {
  const float *learning_rate_;
  const float *momentum_;
  MomentumUpdateVariableFunctor(const float *learning_rate, const float *momentum) {
    this->learning_rate_ = learning_rate;
    this->momentum_ = momentum;
  }
  __device__ __forceinline__ void operator()(half *variable, half *accumulation, const half *gradient) const {
    accumulation[0] = __float2half(momentum_[0]) * accumulation[0] + gradient[0];
    variable[0] -= __float2half(learning_rate_[0]) * accumulation[0];
  }
};

template <>
struct MomentumUpdateVariableWithNesterovFunctor<half, float, half> {
  const float *learning_rate_;
  const float *momentum_;
  MomentumUpdateVariableWithNesterovFunctor(const float *learning_rate, const float *momentum) {
    this->learning_rate_ = learning_rate;
    this->momentum_ = momentum;
  }
  __device__ __forceinline__ void operator()(half *variable, half *accumulation, const half *gradient) const {
    accumulation[0] = __float2half(momentum_[0]) * accumulation[0] + gradient[0];
    variable[0] -= gradient[0] * __float2half(learning_rate_[0]) +
                   accumulation[0] * __float2half(momentum_[0]) * __float2half(learning_rate_[0]);
  }
};

template <>
struct MomentumUpdateVariableFunctor<float, float, half> {
  const float *learning_rate_;
  const float *momentum_;
  MomentumUpdateVariableFunctor(const float *learning_rate, const float *momentum) {
    this->learning_rate_ = learning_rate;
    this->momentum_ = momentum;
  }
  __device__ __forceinline__ void operator()(float *variable, float *accumulation, const half *gradient) const {
    accumulation[0] = momentum_[0] * accumulation[0] + __half2float(gradient[0]);
    variable[0] -= learning_rate_[0] * accumulation[0];
  }
};

template <>
struct MomentumUpdateVariableWithNesterovFunctor<float, float, half> {
  const float *learning_rate_;
  const float *momentum_;
  MomentumUpdateVariableWithNesterovFunctor(const float *learning_rate, const float *momentum) {
    this->learning_rate_ = learning_rate;
    this->momentum_ = momentum;
  }
  __device__ __forceinline__ void operator()(float *variable, float *accumulation, const half *gradient) const {
    accumulation[0] = momentum_[0] * accumulation[0] + __half2float(gradient[0]);
    variable[0] -= __half2float(gradient[0]) * learning_rate_[0] + accumulation[0] * momentum_[0] * learning_rate_[0];
  }
};

template <typename T, typename G>
struct FusedMomentumWeightDecayScaleFunctor {
  const T *learning_rate_;
  const T *momentum_;
  const T *weight_decay_;
  const T *scale_;
  FusedMomentumWeightDecayScaleFunctor(const T *learning_rate, const T *momentum, const T *weight_decay,
                                       const T *scale) {
    this->learning_rate_ = learning_rate;
    this->momentum_ = momentum;
    this->weight_decay_ = weight_decay;
    this->scale_ = scale;
  }
  __device__ __forceinline__ void operator()(T *variable, T *accumulation, const G *gradient) const {
    T grad = (variable[0] * weight_decay_[0] + static_cast<T>(gradient[0])) * scale_[0];
    accumulation[0] = momentum_[0] * accumulation[0] + grad;
    variable[0] -= learning_rate_[0] * accumulation[0];
  }
};

template <typename T, typename G>
struct FusedMomentumScaleFunctor {
  const T *learning_rate_;
  const T *momentum_;
  const T *scale_;
  FusedMomentumScaleFunctor(const T *learning_rate, const T *momentum, const T *scale) {
    this->learning_rate_ = learning_rate;
    this->momentum_ = momentum;
    this->scale_ = scale;
  }
  __device__ __forceinline__ void operator()(T *variable, T *accumulation, const G *gradient) const {
    accumulation[0] = momentum_[0] * accumulation[0] + static_cast<T>(gradient[0]) * scale_[0];
    variable[0] -= learning_rate_[0] * accumulation[0];
  }
};

template <typename T, typename G>
struct FusedWeightDecayMomentumFunctor {
  const T *learning_rate_;
  const T *momentum_;
  const T *weight_decay_;
  FusedWeightDecayMomentumFunctor(const T *learning_rate, const T *momentum, const T *weight_decay) {
    this->learning_rate_ = learning_rate;
    this->momentum_ = momentum;
    this->weight_decay_ = weight_decay;
  }
  __device__ __forceinline__ void operator()(T *variable, T *accumulation, const G *gradient) const {
    T grad = variable[0] * weight_decay_[0] + static_cast<T>(gradient[0]);
    accumulation[0] = momentum_[0] * accumulation[0] + grad;
    variable[0] -= learning_rate_[0] * accumulation[0];
  }
};

template <typename T, typename S, typename G>
void MomentumUpdateVariable(const size_t size, T *variable, T *accumulation, const S *learning_rate, const G *gradient,
                            const S *momentum, bool use_nesterov, cudaStream_t cuda_stream) {
  if (use_nesterov) {
    MomentumUpdateVariableWithNesterovFunctor<T, S, G> functor{learning_rate, momentum};
    cuda::elementwise::UnaryInputBinaryOutput(functor, (uint)(size), variable, accumulation, gradient, cuda_stream);
  } else {
    MomentumUpdateVariableFunctor<T, S, G> functor{learning_rate, momentum};
    cuda::elementwise::UnaryInputBinaryOutput(functor, (uint)(size), variable, accumulation, gradient, cuda_stream);
  }
}

template <typename T, typename G>
void FusedWeightDecayScaleMomentum(const size_t size, T *weight_decay, T *scale, T *variable, T *accumulation,
                                   const T *learning_rate, const G *gradient, const T *momentum,
                                   cudaStream_t cuda_stream) {
  FusedMomentumWeightDecayScaleFunctor<T, G> functor{learning_rate, momentum, weight_decay, scale};
  cuda::elementwise::UnaryInputBinaryOutput(functor, (uint)(size), variable, accumulation, gradient, cuda_stream);
}

template <typename T, typename G>
void FusedScaleMomentum(const size_t size, T *scale, T *variable, T *accumulation, const T *learning_rate,
                        const G *gradient, const T *momentum, cudaStream_t cuda_stream) {
  FusedMomentumScaleFunctor<T, G> functor{learning_rate, momentum, scale};
  cuda::elementwise::UnaryInputBinaryOutput(functor, (uint)(size), variable, accumulation, gradient, cuda_stream);
}

template <typename T, typename G>
void FusedWeightDecayMomentum(const size_t size, T *weight_decay, T *variable, T *accumulation, const T *learning_rate,
                              const G *gradient, const T *momentum, cudaStream_t cuda_stream) {
  FusedWeightDecayMomentumFunctor<T, G> functor{learning_rate, momentum, weight_decay};
  cuda::elementwise::UnaryInputBinaryOutput(functor, (uint)(size), variable, accumulation, gradient, cuda_stream);
}

// CombineFusedScaleMomentum
template <typename T, typename S>
__global__ void CombineFusedMomentumScaleKernel(const size_t num, const size_t *element_num, T **scale, T **variable,
                                                T **accumulation, T **learning_rate, S **gradient, T **momentum) {
  for (size_t idx = 0; idx < num; idx++) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (element_num[idx]); i += blockDim.x * gridDim.x) {
      accumulation[idx][i] = momentum[idx][0] * accumulation[idx][i] + static_cast<T>(gradient[idx][i]) * scale[idx][0];
      variable[idx][i] -= learning_rate[idx][0] * accumulation[idx][i];
    }
  }
}

template <typename T, typename S>
void CombineFusedScaleMomentum(const size_t max, const size_t num, const size_t *elements, T **scale, T **variable,
                               T **accumulation, T **learning_rate, S **gradient, T **momentum,
                               cudaStream_t cuda_stream) {
  size_t thread_per_block = 256;
  size_t block_per_grid = (max + thread_per_block - 1) / thread_per_block;
  CombineFusedMomentumScaleKernel<<<block_per_grid, thread_per_block, 0, cuda_stream>>>(
    num, elements, scale, variable, accumulation, learning_rate, gradient, momentum);
}
// end CombineFusedScaleMomentum

// CombineFusedWeightDecayScaleMomentum
template <typename T, typename S>
__global__ void CombineFusedMomentumWeightDecayScaleKernel(const size_t num, const size_t *element_num,
                                                           T **weight_decay, T **scale, T **variable, T **accumulation,
                                                           T **learning_rate, S **gradient, T **momentum) {
  for (size_t idx = 0; idx < num; idx++) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (element_num[idx]); i += blockDim.x * gridDim.x) {
      T grad = (variable[idx][i] * weight_decay[idx][0] + static_cast<T>(gradient[idx][i])) * scale[idx][0];
      accumulation[idx][i] = momentum[idx][0] * accumulation[idx][i] + grad;
      variable[idx][i] -= learning_rate[idx][0] * accumulation[idx][i];
    }
  }
}

template <typename T, typename S>
void CombineFusedWeightDecayScaleMomentum(const size_t max, const size_t num, const size_t *element_num,
                                          T **weight_decay, T **scale, T **variable, T **accumulation,
                                          T **learning_rate, S **gradient, T **momentum, cudaStream_t cuda_stream) {
  size_t thread_per_block = 256;
  size_t block_per_grid = (max + thread_per_block - 1) / thread_per_block;
  CombineFusedMomentumWeightDecayScaleKernel<<<block_per_grid, thread_per_block, 0, cuda_stream>>>(
    num, element_num, weight_decay, scale, variable, accumulation, learning_rate, gradient, momentum);
}
// end CombineFusedWeightDecayScaleMomentum
template CUDA_LIB_EXPORT void MomentumUpdateVariable<float, float, float>(const size_t size, float *variable,
                                                                          float *accumulation,
                                                                          const float *learning_rate,
                                                                          const float *gradient, const float *momentum,
                                                                          bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<half, half, half>(const size_t size, half *variable,
                                                                       half *accumulation, const half *learning_rate,
                                                                       const half *gradient, const half *momentum,
                                                                       bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<half, float, half>(const size_t size, half *variable,
                                                                        half *accumulation, const float *learning_rate,
                                                                        const half *gradient, const float *momentum,
                                                                        bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<float, float, half>(const size_t size, float *variable,
                                                                         float *accumulation,
                                                                         const float *learning_rate,
                                                                         const half *gradient, const float *momentum,
                                                                         bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<int8_t, int8_t, int8_t>(
  const size_t size, int8_t *variable, int8_t *accumulation, const int8_t *learning_rate, const int8_t *gradient,
  const int8_t *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<uint8_t, uint8_t, uint8_t>(
  const size_t size, uint8_t *variable, uint8_t *accumulation, const uint8_t *learning_rate, const uint8_t *gradient,
  const uint8_t *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<int16_t, int16_t, int16_t>(
  const size_t size, int16_t *variable, int16_t *accumulation, const int16_t *learning_rate, const int16_t *gradient,
  const int16_t *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<uint16_t, uint16_t, uint16_t>(
  const size_t size, uint16_t *variable, uint16_t *accumulation, const uint16_t *learning_rate,
  const uint16_t *gradient, const uint16_t *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<uint32_t, uint32_t, uint32_t>(
  const size_t size, uint32_t *variable, uint32_t *accumulation, const uint32_t *learning_rate,
  const uint32_t *gradient, const uint32_t *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<int32_t, int32_t, int32_t>(
  const size_t size, int32_t *variable, int32_t *accumulation, const int32_t *learning_rate, const int32_t *gradient,
  const int32_t *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<int64_t, int64_t, int64_t>(
  const size_t size, int64_t *variable, int64_t *accumulation, const int64_t *learning_rate, const int64_t *gradient,
  const int64_t *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<uint64_t, uint64_t, uint64_t>(
  const size_t size, uint64_t *variable, uint64_t *accumulation, const uint64_t *learning_rate,
  const uint64_t *gradient, const uint64_t *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<double, double, double>(
  const size_t size, double *variable, double *accumulation, const double *learning_rate, const double *gradient,
  const double *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<Complex<float>, Complex<float>, Complex<float>>(
  const size_t size, Complex<float> *variable, Complex<float> *accumulation, const Complex<float> *learning_rate,
  const Complex<float> *gradient, const Complex<float> *momentum, bool use_nesterov, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MomentumUpdateVariable<Complex<double>, Complex<double>, Complex<double>>(
  const size_t size, Complex<double> *variable, Complex<double> *accumulation, const Complex<double> *learning_rate,
  const Complex<double> *gradient, const Complex<double> *momentum, bool use_nesterov, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void FusedWeightDecayScaleMomentum(const size_t element_num, float *weight_decay, float *scale,
                                                            float *variable, float *accumulation,
                                                            const float *learning_rate, const float *gradient,
                                                            const float *momentum, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void FusedWeightDecayScaleMomentum(const size_t element_num, float *weight_decay, float *scale,
                                                            float *variable, float *accumulation,
                                                            const float *learning_rate, const half *gradient,
                                                            const float *momentum, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void FusedWeightDecayMomentum(const size_t element_num, float *weight_decay, float *variable,
                                                       float *accumulation, const float *learning_rate,
                                                       const float *gradient, const float *momentum,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void FusedWeightDecayMomentum(const size_t element_num, float *weight_decay, float *variable,
                                                       float *accumulation, const float *learning_rate,
                                                       const half *gradient, const float *momentum,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void FusedScaleMomentum(const size_t element_num, float *scale, float *variable,
                                                 float *accumulation, const float *learning_rate, const float *gradient,
                                                 const float *momentum, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void FusedScaleMomentum(const size_t element_num, float *scale, float *variable,
                                                 float *accumulation, const float *learning_rate, const half *gradient,
                                                 const float *momentum, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CombineFusedWeightDecayScaleMomentum(
  const size_t max, const size_t num, const size_t *elements, float **weight_decay, float **scale, float **variable,
  float **accumulation, float **learning_rate, float **gradient, float **momentum, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CombineFusedWeightDecayScaleMomentum(
  const size_t max, const size_t num, const size_t *elements, float **weight_decay, float **scale, float **variable,
  float **accumulation, float **learning_rate, half **gradient, float **momentum, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CombineFusedScaleMomentum(const size_t max, const size_t num, const size_t *elements,
                                                        float **scale, float **variable, float **accumulation,
                                                        float **learning_rate, float **gradient, float **momentum,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CombineFusedScaleMomentum(const size_t max, const size_t num, const size_t *elements,
                                                        float **scale, float **variable, float **accumulation,
                                                        float **learning_rate, half **gradient, float **momentum,
                                                        cudaStream_t cuda_stream);
