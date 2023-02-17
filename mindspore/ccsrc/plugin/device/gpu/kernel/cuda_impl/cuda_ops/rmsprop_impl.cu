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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/rmsprop_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

__device__ __forceinline__ Complex<float> SqrtFunc(Complex<float> input) {
  Complex<float> output;
  float ctheta;
  float Pi = 3.14159265358979323846;
  if (input.real() > 0.0) {
    ctheta = atan(input.imag() / input.real());
  }
  if (input.real() == 0.0 && input.imag() > 0.0) {
      ctheta = Pi / 2.0;
  }
  if (input.real() == 0.0 && input.imag() < 0.0) {
    ctheta = -1.0 * Pi / 2.0;
  }
  if (input.real() < 0.0 && input.imag() >= 0.0) {
    ctheta = atan(input.imag() / input.real()) + Pi;
  }
  if (input.real() < 0.0 && input.imag() < 0.0) {
    ctheta = atan(input.imag() / input.real()) - Pi;
  }
  output.real(sqrt(abs(input)) * cos(ctheta / 2.0));
  output.imag(sqrt(abs(input)) * sin(ctheta / 2.0));
  return output;
}

__device__ __forceinline__ Complex<double> SqrtFunc(Complex<double> input) {
  Complex<double> output;
  double ctheta;
  double Pi = 3.14159265358979323846;
  if (input.real() > 0.0) {
    ctheta = atan(input.imag() / input.real());
  }
  if (input.real() == 0.0 && input.imag() > 0.0) {
      ctheta = Pi / 2.0;
  }
  if (input.real() == 0.0 && input.imag() < 0.0) {
    ctheta = -1.0 * Pi / 2.0;
  }
  if (input.real() < 0.0 && input.imag() >= 0.0) {
    ctheta = atan(input.imag() / input.real()) + Pi;
  }
  if (input.real() < 0.0 && input.imag() < 0.0) {
    ctheta = atan(input.imag() / input.real()) - Pi;
  }
  output.real(sqrt(abs(input)) * cos(ctheta / 2.0));
  output.imag(sqrt(abs(input)) * sin(ctheta / 2.0));
  return output;
}

template <typename T>
__global__ void RmsPropKernel(const size_t batch_size, const size_t input_elements, const T *learning_rate,
                              const T *decay, const T *momentum, const T *epsilon, T *variable, T *mean_square,
                              T *moment, T *gradients, const size_t size) {
  auto all_elements = batch_size * input_elements;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (all_elements); i += blockDim.x * gridDim.x) {
    auto batch = i / input_elements;
    mean_square[i] = decay[0] * mean_square[i] + (static_cast<T>(1.0) - decay[0]) * gradients[i] * gradients[i];
    moment[i] = momentum[0] * moment[i] + learning_rate[batch] *
                static_cast<T>(rsqrt(static_cast<double>(mean_square[i] + epsilon[0]))) * gradients[i];
    variable[i] -= moment[i];
  }
}

template <>
__global__ void RmsPropKernel(const size_t batch_size, const size_t input_elements,
                              const Complex<double> *learning_rate, const Complex<double> *decay,
                              const Complex<double> *momentum, const Complex<double> *epsilon,
                              Complex<double> *variable, Complex<double> *mean_square, Complex<double> *moment,
                              Complex<double> *gradients, const size_t size) {
  auto all_elements = batch_size * input_elements;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (all_elements); i += blockDim.x * gridDim.x) {
    auto batch = i / input_elements;
    mean_square[i] = decay[0] * mean_square[i] + (static_cast<Complex<double>>(1.0) - decay[batch]) *
                     gradients[i] * gradients[i];
    moment[i] = momentum[0] * moment[i] + learning_rate[batch] / SqrtFunc(mean_square[i] + epsilon[0]) * gradients[i];
    variable[i] -= moment[i];
  }
}

template <>
__global__ void RmsPropKernel(const size_t batch_size, const size_t input_elements,
                              const Complex<float> *learning_rate, const Complex<float> *decay,
                              const Complex<float> *momentum, const Complex<float> *epsilon, Complex<float> *variable,
                              Complex<float> *mean_square, Complex<float> *moment, Complex<float> *gradients,
                              const size_t size) {
  auto all_elements = batch_size * input_elements;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (all_elements); i += blockDim.x * gridDim.x) {
    auto batch = i / input_elements;
    mean_square[i] = decay[0] * mean_square[i] + (static_cast<Complex<float>>(1.0) - decay[batch]) * gradients[i] *
                     gradients[i];
    moment[i] = momentum[0] * moment[i] + learning_rate[batch] / SqrtFunc(mean_square[i] + epsilon[0]) * gradients[i];
    variable[i] -= moment[i];
  }
}

template <typename T>
void RmsProp(const size_t batch_size, const size_t input_elements, const T *learning_rate, const T *decay,
             const T *momentum, const T *epsilon, T *variable, T *mean_square, T *moment, T *gradients,
             const size_t size, cudaStream_t cuda_stream) {
  RmsPropKernel<<<GET_BLOCKS(input_elements), GET_THREADS, 0, cuda_stream>>>(batch_size, input_elements, learning_rate,
                                                                             decay, momentum, epsilon, variable,
                                                                             mean_square, moment, gradients, size);
}

template <typename T>
__global__ void RmsPropCenterKernel(const size_t batch_size, const size_t input_elements, const T *learning_rate,
                                    const T *decay, const T *momentum, const T *epsilon, T *variable, T *mean_gradients,
                                    T *mean_square, T *moment, T *gradients, const size_t size) {
  auto all_elements = batch_size * input_elements;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (all_elements); i += blockDim.x * gridDim.x) {
    auto batch = i / input_elements;
    mean_gradients[i] =
      decay[batch] * mean_gradients[i] + (static_cast<T>(1.0) - decay[batch]) * gradients[i];
    mean_square[i] = decay[batch] * mean_square[i] + (static_cast<T>(1.0) - decay[batch]) * gradients[i] * gradients[i];
    moment[i] = momentum[batch] * moment[i] +
                learning_rate[batch] *
                  static_cast<T>(rsqrt(
                    static_cast<double>(mean_square[i] - mean_gradients[i] * mean_gradients[i] + epsilon[batch]))) *
                  gradients[i];
    variable[i] -= moment[i];
  }
}

template <>
__global__ void RmsPropCenterKernel(const size_t batch_size, const size_t input_elements, const float *learning_rate,
                                    const float *decay, const float *momentum, const float *epsilon, float *variable,
                                    float *mean_gradients, float *mean_square, float *moment, float *gradients,
                                    const size_t size) {
  auto all_elements = batch_size * input_elements;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (all_elements); i += blockDim.x * gridDim.x) {
    auto batch = i / input_elements;
    mean_gradients[i] =
      decay[batch] * mean_gradients[i] + (1.0 - decay[batch]) * gradients[i];
    mean_square[i] = decay[batch] * mean_square[i] + (1.0 - decay[batch]) * gradients[i] * gradients[i];
    moment[i] = momentum[batch] * moment[i] +
                learning_rate[batch] *
                rsqrt(mean_square[i] - mean_gradients[i] * mean_gradients[i] + epsilon[batch]) * gradients[i];
    variable[i] -= moment[i];
  }
}

template <>
__global__ void RmsPropCenterKernel(const size_t batch_size, const size_t input_elements,
                                    const Complex<double> *learning_rate, const Complex<double> *decay,
                                    const Complex<double> *momentum, const Complex<double> *epsilon,
                                    Complex<double> *variable, Complex<double> *mean_gradients,
                                    Complex<double> *mean_square, Complex<double> *moment, Complex<double> *gradients,
                                    const size_t size) {
  auto all_elements = batch_size * input_elements;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (all_elements); i += blockDim.x * gridDim.x) {
    auto batch = i / input_elements;
    mean_gradients[i] =
      decay[batch] * mean_gradients[i] + (static_cast<Complex<double>>(1.0) - decay[batch]) * gradients[i];
    mean_square[i] =
      decay[batch] * mean_square[i] + (static_cast<Complex<double>>(1.0) - decay[batch]) * gradients[i] * gradients[i];
    moment[i] = momentum[batch] * moment[i] +
                learning_rate[batch] /
                  SqrtFunc(mean_square[i] - mean_gradients[i] * mean_gradients[i] + epsilon[batch]) * gradients[i];
    variable[i] -= moment[i];
  }
}

template <>
__global__ void RmsPropCenterKernel(const size_t batch_size, const size_t input_elements,
                                    const Complex<float> *learning_rate, const Complex<float> *decay,
                                    const Complex<float> *momentum, const Complex<float> *epsilon,
                                    Complex<float> *variable, Complex<float> *mean_gradients,
                                    Complex<float> *mean_square, Complex<float> *moment, Complex<float> *gradients,
                                    const size_t size) {
  auto all_elements = batch_size * input_elements;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (all_elements); i += blockDim.x * gridDim.x) {
    auto batch = i / input_elements;
    mean_gradients[i] =
      decay[batch] * mean_gradients[i] + (static_cast<Complex<float>>(1.0) - decay[batch]) * gradients[i];
    mean_square[i] =
      decay[batch] * mean_square[i] + (static_cast<Complex<float>>(1.0) - decay[batch]) * gradients[i] * gradients[i];
    moment[i] = momentum[batch] * moment[i] +
                learning_rate[batch] /
                  SqrtFunc(mean_square[i] - mean_gradients[i] * mean_gradients[i] + epsilon[batch]) * gradients[i];
    variable[i] -= moment[i];
  }
}

template <>
__global__ void RmsPropCenterKernel(const size_t batch_size, const size_t input_elements, const half *learning_rate,
                                    const half *decay, const half *momentum, const half *epsilon, half *variable,
                                    half *mean_gradients, half *mean_square, half *moment, half *gradients,
                                    const size_t size) {
  auto all_elements = batch_size * input_elements;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (all_elements); i += blockDim.x * gridDim.x) {
    auto batch = i / input_elements;
    mean_gradients[i] =
      decay[batch] * mean_gradients[i] + (static_cast<half>(1.0) - decay[batch]) * gradients[i];
    mean_square[i] = decay[batch] * mean_square[i] + (static_cast<half>(1.0) - decay[batch]) *
                     gradients[i] * gradients[i];
    moment[i] = momentum[batch] * moment[i] + learning_rate[batch] *
                static_cast<half>(rsqrt(static_cast<float>(mean_square[i] - mean_gradients[i] * mean_gradients[i] +
                epsilon[batch]))) * gradients[i];
    variable[i] -= moment[i];
  }
}

template <typename T>
void RmsPropCenter(const size_t batch_size, const size_t input_elements, const T *learning_rate, const T *decay,
                   const T *momentum, const T *epsilon, T *variable, T *mean_gradients, T *mean_square, T *moment,
                   T *gradients, const size_t size, cudaStream_t cuda_stream) {
  RmsPropCenterKernel<<<GET_BLOCKS(input_elements), GET_THREADS, 0, cuda_stream>>>(
    batch_size, input_elements, learning_rate, decay, momentum, epsilon, variable, mean_gradients, mean_square, moment,
    gradients, size);
}

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements, const float *learning_rate,
                                      const float *decay, const float *momentum, const float *epsilon, float *variable,
                                      float *mean_square, float *moment, float *gradients, const size_t size,
                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements, const half *learning_rate,
                                      const half *decay, const half *momentum, const half *epsilon, half *variable,
                                      half *mean_square, half *moment, half *gradients, const size_t size,
                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements, const double *learning_rate,
                                      const double *decay, const double *momentum, const double *epsilon,
                                      double *variable, double *mean_square, double *moment, double *gradients,
                                      const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements, const int8_t *learning_rate,
                                      const int8_t *decay, const int8_t *momentum, const int8_t *epsilon,
                                      int8_t *variable, int8_t *mean_square, int8_t *moment, int8_t *gradients,
                                      const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements,
                                      const int16_t *learning_rate, const int16_t *decay, const int16_t *momentum,
                                      const int16_t *epsilon, int16_t *variable, int16_t *mean_square, int16_t *moment,
                                      int16_t *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements,
                                      const int32_t *learning_rate, const int32_t *decay, const int32_t *momentum,
                                      const int32_t *epsilon, int32_t *variable, int32_t *mean_square, int32_t *moment,
                                      int32_t *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements,
                                      const int64_t *learning_rate, const int64_t *decay, const int64_t *momentum,
                                      const int64_t *epsilon, int64_t *variable, int64_t *mean_square, int64_t *moment,
                                      int64_t *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements,
                                      const uint8_t *learning_rate, const uint8_t *decay, const uint8_t *momentum,
                                      const uint8_t *epsilon, uint8_t *variable, uint8_t *mean_square, uint8_t *moment,
                                      uint8_t *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements,
                                      const uint16_t *learning_rate, const uint16_t *decay, const uint16_t *momentum,
                                      const uint16_t *epsilon, uint16_t *variable, uint16_t *mean_square,
                                      uint16_t *moment, uint16_t *gradients, const size_t size,
                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements,
                                      const uint32_t *learning_rate, const uint32_t *decay, const uint32_t *momentum,
                                      const uint32_t *epsilon, uint32_t *variable, uint32_t *mean_square,
                                      uint32_t *moment, uint32_t *gradients, const size_t size,
                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements,
                                      const uint64_t *learning_rate, const uint64_t *decay, const uint64_t *momentum,
                                      const uint64_t *epsilon, uint64_t *variable, uint64_t *mean_square,
                                      uint64_t *moment, uint64_t *gradients, const size_t size,
                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements,
                                      const Complex<float> *learning_rate, const Complex<float> *decay,
                                      const Complex<float> *momentum, const Complex<float> *epsilon,
                                      Complex<float> *variable, Complex<float> *mean_square, Complex<float> *moment,
                                      Complex<float> *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsProp(const size_t batch_size, const size_t input_elements,
                                      const Complex<double> *learning_rate, const Complex<double> *decay,
                                      const Complex<double> *momentum, const Complex<double> *epsilon,
                                      Complex<double> *variable, Complex<double> *mean_square, Complex<double> *moment,
                                      Complex<double> *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const float *learning_rate, const float *decay, const float *momentum,
                                            const float *epsilon, float *variable, float *mean_gradients,
                                            float *mean_square, float *moment, float *gradients, const size_t size,
                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const half *learning_rate, const half *decay, const half *momentum,
                                            const half *epsilon, half *variable, half *mean_gradients,
                                            half *mean_square, half *moment, half *gradients, const size_t size,
                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const double *learning_rate, const double *decay, const double *momentum,
                                            const double *epsilon, double *variable, double *mean_gradients,
                                            double *mean_square, double *moment, double *gradients, const size_t size,
                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const int8_t *learning_rate, const int8_t *decay, const int8_t *momentum,
                                            const int8_t *epsilon, int8_t *variable, int8_t *mean_gradients,
                                            int8_t *mean_square, int8_t *moment, int8_t *gradients, const size_t size,
                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const int16_t *learning_rate, const int16_t *decay, const int16_t *momentum,
                                            const int16_t *epsilon, int16_t *variable, int16_t *mean_gradients,
                                            int16_t *mean_square, int16_t *moment, int16_t *gradients,
                                            const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const int32_t *learning_rate, const int32_t *decay, const int32_t *momentum,
                                            const int32_t *epsilon, int32_t *variable, int32_t *mean_gradients,
                                            int32_t *mean_square, int32_t *moment, int32_t *gradients,
                                            const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const int64_t *learning_rate, const int64_t *decay, const int64_t *momentum,
                                            const int64_t *epsilon, int64_t *variable, int64_t *mean_gradients,
                                            int64_t *mean_square, int64_t *moment, int64_t *gradients,
                                            const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const uint8_t *learning_rate, const uint8_t *decay, const uint8_t *momentum,
                                            const uint8_t *epsilon, uint8_t *variable, uint8_t *mean_gradients,
                                            uint8_t *mean_square, uint8_t *moment, uint8_t *gradients,
                                            const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const uint16_t *learning_rate, const uint16_t *decay,
                                            const uint16_t *momentum, const uint16_t *epsilon, uint16_t *variable,
                                            uint16_t *mean_gradients, uint16_t *mean_square, uint16_t *moment,
                                            uint16_t *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const uint32_t *learning_rate, const uint32_t *decay,
                                            const uint32_t *momentum, const uint32_t *epsilon, uint32_t *variable,
                                            uint32_t *mean_gradients, uint32_t *mean_square, uint32_t *moment,
                                            uint32_t *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const uint64_t *learning_rate, const uint64_t *decay,
                                            const uint64_t *momentum, const uint64_t *epsilon, uint64_t *variable,
                                            uint64_t *mean_gradients, uint64_t *mean_square, uint64_t *moment,
                                            uint64_t *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const Complex<float> *learning_rate, const Complex<float> *decay,
                                            const Complex<float> *momentum, const Complex<float> *epsilon,
                                            Complex<float> *variable, Complex<float> *mean_gradients,
                                            Complex<float> *mean_square, Complex<float> *moment,
                                            Complex<float> *gradients, const size_t size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void RmsPropCenter(const size_t batch_size, const size_t input_elements,
                                            const Complex<double> *learning_rate, const Complex<double> *decay,
                                            const Complex<double> *momentum, const Complex<double> *epsilon,
                                            Complex<double> *variable, Complex<double> *mean_gradients,
                                            Complex<double> *mean_square, Complex<double> *moment,
                                            Complex<double> *gradients, const size_t size, cudaStream_t cuda_stream);
