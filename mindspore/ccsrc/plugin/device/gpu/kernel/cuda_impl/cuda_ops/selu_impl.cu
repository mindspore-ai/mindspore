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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/selu_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementswise_op_impl.cuh"
template <typename T, typename IsInteger = void>
struct SeLUFunctor {
  T scale_;
  T scale_dot_alpha_;
  explicit SeLUFunctor(float scale, float scale_dot_alpha)
      : scale_(static_cast<T>(scale)), scale_dot_alpha_(static_cast<T>(scale_dot_alpha)) {}
  __device__ __forceinline__ T operator()(T x) const { return x >= T(0) ? scale_ * x : scale_dot_alpha_ * expm1(x); }
};

template <typename T>
struct SeLUFunctor<T, typename std::enable_if<std::is_integral<T>::value>::type> {
  float scale_;
  float scale_dot_alpha_;
  explicit SeLUFunctor(float scale, float scale_dot_alpha) : scale_(scale), scale_dot_alpha_(scale_dot_alpha) {}
  __device__ __forceinline__ T operator()(T tx) const {
    auto x = static_cast<float>(tx);
    return tx >= 0 ? scale_ * x : scale_dot_alpha_ * expm1(x);
  }
};

template <>
struct SeLUFunctor<half> {
  half scale_;
  half scale_dot_alpha_;
  explicit SeLUFunctor(float scale, float scale_dot_alpha)
      : scale_(static_cast<half>(scale)), scale_dot_alpha_(static_cast<half>(scale_dot_alpha)) {}
  __device__ __forceinline__ half operator()(half x) const {
    return x >= half(0) ? scale_ * x : scale_dot_alpha_ * static_cast<half>(expm1(__half2float(x)));
  }
};

template <typename T>
void CalculateSeLU(const T *input, size_t input_elements, float scale_dot_alpha, float scale, T *output,
                   const uint32_t &device_id, cudaStream_t cuda_stream) {
  SeLUFunctor<T> functor{scale, scale_dot_alpha};
  cuda::elementwise::Unary(functor, (uint)(input_elements), output, input, cuda_stream);
}

template CUDA_LIB_EXPORT void CalculateSeLU<double>(const double *input, size_t input_elements, float scale_dot_alpha,
                                                    float scale, double *output, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateSeLU<float>(const float *input, size_t input_elements, float scale_dot_alpha,
                                                   float scale, float *output, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateSeLU<half>(const half *input, size_t input_elements, float scale_dot_alpha,
                                                  float scale, half *output, const uint32_t &device_id,
                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateSeLU<int8_t>(const int8_t *input, size_t input_elements, float scale_dot_alpha,
                                                    float scale, int8_t *output, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateSeLU<int32_t>(const int32_t *input, size_t input_elements, float scale_dot_alpha,
                                                     float scale, int32_t *output, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
