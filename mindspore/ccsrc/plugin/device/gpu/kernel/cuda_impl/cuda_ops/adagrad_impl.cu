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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adagrad_impl.cuh"
#include "include/cuda_fp16.h"

using Complex64 = Complex<float>;
using Complex128 = Complex<double>;

template <typename T>
__device__ __forceinline__ T SqrtFunc(T input) {
  return sqrt(input);
}

template <>
__device__ __forceinline__ half SqrtFunc(half input) {
  return hsqrt(input);
}

template <>
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

template <>
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

template <typename T, typename S, typename G>
__global__ void ApplyAdagradKernel(const size_t size, const bool update_slots, const S *learning_rate,
                                   const G *gradient, T *variable, T *accumulation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (update_slots) {
      accumulation[i] += gradient[i] * gradient[i];
    }
    variable[i] -= learning_rate[0] * gradient[i] / SqrtFunc(accumulation[i]);
  }
}

template <>
__global__ void ApplyAdagradKernel(const size_t size, const bool update_slots, const float *learning_rate,
                                   const half *gradient, half *variable, half *accumulation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (update_slots) {
      accumulation[i] += gradient[i] * gradient[i];
    }
    variable[i] -= __float2half(learning_rate[0]) * gradient[i] / SqrtFunc(accumulation[i]);
  }
}

template <>
__global__ void ApplyAdagradKernel(const size_t size, const bool update_slots, const half *learning_rate,
                                   const float *gradient, float *variable, float *accumulation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (update_slots) {
      accumulation[i] += gradient[i] * gradient[i];
    }
    variable[i] -= __half2float(learning_rate[0]) * gradient[i] / SqrtFunc(accumulation[i]);
  }
}

template <>
__global__ void ApplyAdagradKernel(const size_t size, const bool update_slots, const double *learning_rate,
                                   const half *gradient, half *variable, half *accumulation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (update_slots) {
      accumulation[i] += gradient[i] * gradient[i];
    }
    variable[i] -= static_cast<half>(learning_rate[0]) * gradient[i] / SqrtFunc(accumulation[i]);
  }
}

template <>
__global__ void ApplyAdagradKernel(const size_t size, const bool update_slots, const double *learning_rate,
                                   const float *gradient, float *variable, float *accumulation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (update_slots) {
      accumulation[i] += gradient[i] * gradient[i];
    }
    variable[i] -= static_cast<float>(learning_rate[0]) * gradient[i] / SqrtFunc(accumulation[i]);
  }
}

template <>
__global__ void ApplyAdagradKernel(const size_t size, const bool update_slots, const double *learning_rate,
                                   const double *gradient, double *variable, double *accumulation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (update_slots) {
      accumulation[i] += gradient[i] * gradient[i];
    }
    variable[i] -= learning_rate[0] * gradient[i] / SqrtFunc(accumulation[i]);
  }
}

template <>
__global__ void ApplyAdagradKernel(const size_t size, const bool update_slots, const half *learning_rate,
                                   const double *gradient, double *variable, double *accumulation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (update_slots) {
      accumulation[i] += gradient[i] * gradient[i];
    }
    variable[i] -= static_cast<double>(learning_rate[0]) * gradient[i] / SqrtFunc(accumulation[i]);
  }
}

template <>
__global__ void ApplyAdagradKernel(const size_t size, const bool update_slots, const float *learning_rate,
                                   const double *gradient, double *variable, double *accumulation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (update_slots) {
      accumulation[i] += gradient[i] * gradient[i];
    }
    variable[i] -= static_cast<double>(learning_rate[0]) * gradient[i] / SqrtFunc(accumulation[i]);
  }
}

template <typename T, typename S, typename G>
cudaError_t ApplyAdagrad(const size_t size, const bool update_slots, const S *learning_rate, const G *gradient,
                         T *variable, T *accumulation, cudaStream_t cuda_stream) {
  ApplyAdagradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, update_slots, learning_rate, gradient,
                                                                        variable, accumulation);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<float, float, float>(const size_t size, const bool update_slots,
                                                                       const float *learning_rate,
                                                                       const float *gradient, float *variable,
                                                                       float *accumulation, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<half, half, half>(const size_t size, const bool update_slots,
                                                                    const half *learning_rate, const half *gradient,
                                                                    half *variable, half *accumulation,
                                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<half, float, half>(const size_t size, const bool update_slots,
                                                                     const float *learning_rate, const half *gradient,
                                                                     half *variable, half *accumulation,
                                                                     cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<float, half, float>(const size_t size, const bool update_slots,
                                                                      const half *learning_rate, const float *gradient,
                                                                      float *variable, float *accumulation,
                                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<half, double, half>(const size_t size, const bool update_slots,
                                                                      const double *learning_rate, const half *gradient,
                                                                      half *variable, half *accumulation,
                                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<float, double, float>(const size_t size, const bool update_slots,
                                                                        const double *learning_rate,
                                                                        const float *gradient, float *variable,
                                                                        float *accumulation, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<double, double, double>(const size_t size, const bool update_slots,
                                                                          const double *learning_rate,
                                                                          const double *gradient, double *variable,
                                                                          double *accumulation,
                                                                          cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<double, half, double>(const size_t size, const bool update_slots,
                                                                        const half *learning_rate,
                                                                        const double *gradient, double *variable,
                                                                        double *accumulation, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<double, float, double>(const size_t size, const bool update_slots,
                                                                         const float *learning_rate,
                                                                         const double *gradient, double *variable,
                                                                         double *accumulation,
                                                                         cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<Complex64, Complex64, Complex64>(
  const size_t size, const bool update_slots, const Complex64 *learning_rate, const Complex64 *gradient,
  Complex64 *variable, Complex64 *accumulation, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagrad<Complex128, Complex128, Complex128>(
  const size_t size, const bool update_slots, const Complex128 *learning_rate, const Complex128 *gradient,
  Complex128 *variable, Complex128 *accumulation, cudaStream_t cuda_stream);
