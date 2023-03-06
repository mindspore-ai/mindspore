/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void Expm1Kernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = expm1f(input[i]);
  }
  return;
}
template <>
__global__ void Expm1Kernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = expm1(input[i]);
  }
  return;
}
template <typename T>
__global__ void Expm1Kernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    Complex<T> complex_one(1, 0);
    output[i] = exp(input[i]) - complex_one;
  }
  return;
}
template <typename T>
__global__ void Log1pKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = log1pf(input[i]);
  }
  return;
}
template <>
__global__ void Log1pKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = log1p(input[i]);
  }
  return;
}
template <typename T>
__global__ void Log1pKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = log(input[i] + Complex<T>(1, 0));
  }
  return;
}
template <typename T>
__global__ void ErfKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = erff(input[i]);
  }
  return;
}
template <>
__global__ void ErfKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = erf(input[i]);
  }
  return;
}
template <typename T>
__global__ void ErfcKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = erfcf(input[i]);
  }
  return;
}
template <>
__global__ void ErfcKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = erfc(input[i]);
  }
  return;
}
template <typename T>
__global__ void InvertKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = ~input[i];
  }
}
template <>
__global__ void InvertKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <>
__global__ void InvertKernel(const float *input, float *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <>
__global__ void InvertKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}

template <typename T>
__global__ void RsqrtKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rsqrtf(input[i]);
  }
  return;
}
template <>
__global__ void RsqrtKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rsqrt(input[i]);
  }
  return;
}
template <>
__global__ void RsqrtKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hrsqrt(input[i]);
  }
  return;
}
template <typename T>
__global__ void RsqrtKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T real = sqrt(input[i]).real();
    T imag = sqrt(input[i]).imag();
    T sum = real * real + imag * imag;
    output[i] = Complex<T>(real / sum, -imag / sum);
  }
  return;
}
template <typename T>
__global__ void SinKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sinf(input[i]);
  }
  return;
}
template <>
__global__ void SinKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sin(input[i]);
  }
  return;
}
template <>
__global__ void SinKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hsin(input[i]);
  }
  return;
}
template <typename T>
__global__ void SinKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sin(input[i]);
  }
  return;
}
template <typename T>
__global__ void SinhKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    output[i] = sinhf(input[i]);
  }
  return;
}
template <>
__global__ void SinhKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    output[i] = sinh(input[i]);
  }
  return;
}
template <>
__global__ void SinhKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    output[i] = half(0.5) * (hexp(input[i]) - hexp(-input[i]));
  }
  return;
}
template <typename T>
__global__ void SinhKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sinh(input[i]);
  }
  return;
}
template <typename T>
__global__ void TanKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = tanf(input[i]);
  }
  return;
}
template <>
__global__ void TanKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = tan(input[i]);
  }
  return;
}
template <>
__global__ void TanKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hsin(input[i]) / hcos(input[i]);
  }
  return;
}
template <typename T>
__global__ void TanKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = tan(input[i]);
  }
  return;
}
template <typename T>
__global__ void AsinKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asinf(input[i]);
  }
  return;
}
template <>
__global__ void AsinKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asin(input[i]);
  }
  return;
}
template <typename T>
__global__ void AsinKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asin(input[i]);
  }
  return;
}
template <typename T>
__global__ void AsinhKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asinhf(input[i]);
  }
  return;
}
template <>
__global__ void AsinhKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asinh(input[i]);
  }
  return;
}
template <typename T>
__global__ void AsinhKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asinh(input[i]);
  }
  return;
}
template <typename T>
__global__ void CosKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = cosf(input[i]);
  }
  return;
}
template <>
__global__ void CosKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = cos(input[i]);
  }
  return;
}
template <>
__global__ void CosKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hcos(input[i]);
  }
  return;
}
template <typename T>
__global__ void CosKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = cos(input[i]);
  }
  return;
}
template <typename T>
__global__ void CoshKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = coshf(input[i]);
  }
  return;
}
template <>
__global__ void CoshKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = cosh(input[i]);
  }
  return;
}
template <>
__global__ void CoshKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = half(0.5) * (hexp(input[i]) + hexp(-input[i]));
  }
  return;
}
template <typename T>
__global__ void CoshKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = cosh(input[i]);
  }
  return;
}
template <typename T>
__global__ void ACosKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acosf(input[i]);
  }
  return;
}
template <>
__global__ void ACosKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acos(input[i]);
  }
  return;
}
template <typename T>
__global__ void ACosKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acos(input[i]);
  }
  return;
}
template <typename T>
__global__ void AcoshKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acoshf(input[i]);
  }
  return;
}
template <>
__global__ void AcoshKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acosh(input[i]);
  }
  return;
}
template <typename T>
__global__ void AcoshKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acosh(input[i]);
  }
  return;
}
template <typename T>
__global__ void AtanhKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atanh(input[i]);
  }
  return;
}
template <typename T>
__global__ void AtanhKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float x = static_cast<float>(input[i]);
    output[i] = static_cast<T>(atanhf(x));
  }
}
template <>
__global__ void AtanhKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atanh(input[i]);
  }
}
template <>
__global__ void AtanhKernel(const float *input, float *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atanhf(input[i]);
  }
}
template <>
__global__ void AtanhKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    half x = input[i];
    const half &one = static_cast<half>(1.);
    // 0.5 * ln((1 + x) / (1 - x))
    output[i] = static_cast<half>(0.5) * hlog((static_cast<half>(1.) + x) / (static_cast<half>(1.) - x));
  }
}
template <typename T>
__global__ void AtanKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atanf(input[i]);
  }
  return;
}
template <>
__global__ void AtanKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atan(input[i]);
  }
  return;
}
template <typename T>
__global__ void AtanKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atan(input[i]);
  }
  return;
}
template <typename T>
__global__ void AbsKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = abs(input[i]);
  }
  return;
}
template <>
__global__ void AbsKernel(const half *input, half *output, const size_t count) {
  half zero = 0.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] < zero ? -input[i] : input[i];
  }
  return;
}
template <>
__global__ void AbsKernel(const bool *input, bool *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <>
__global__ void AbsKernel(const uint8_t *input, uint8_t *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <>
__global__ void AbsKernel(const uint16_t *input, uint16_t *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <>
__global__ void AbsKernel(const uint32_t *input, uint32_t *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <>
__global__ void AbsKernel(const uint64_t *input, uint64_t *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <typename T>
__global__ void AbsKernel(const Complex<T> *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = abs(input[i]);
  }
  return;
}
template <typename T>
__global__ void RealKernel(const Complex<T> *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i].real();
  }
  return;
}
template <typename T>
__global__ void RealKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <typename T>
__global__ void ImagKernel(const Complex<T> *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i].imag();
  }
  return;
}
template <typename T>
__global__ void ImagKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T zero = 0;
    output[i] = zero;
  }
  return;
}
template <typename T>
__global__ void ConjKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = Complex<T>(input[i].real(), -input[i].imag());
  }
  return;
}
template <typename T>
__global__ void ConjKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <typename T>
__global__ void FloorKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = floorf(input[i]);
  }
  return;
}
template <>
__global__ void FloorKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = floor(input[i]);
  }
  return;
}
template <>
__global__ void FloorKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hfloor(input[i]);
  }
  return;
}
template <typename T>
__global__ void TruncKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = truncf(input[i]);
  }
  return;
}
template <>
__global__ void TruncKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = trunc(input[i]);
  }
  return;
}
template <>
__global__ void TruncKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = htrunc(input[i]);
  }
  return;
}
template <typename T>
__global__ void CeilKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = ceilf(input[i]);
  }
  return;
}
template <>
__global__ void CeilKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = ceil(input[i]);
  }
  return;
}
template <>
__global__ void CeilKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hceil(input[i]);
  }
  return;
}
template <typename T>
__global__ void RintKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rintf(input[i]);
  }
  return;
}
template <>
__global__ void RintKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rint(input[i]);
  }
  return;
}
template <>
__global__ void RintKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hrint(input[i]);
  }
  return;
}
template <typename T>
__global__ void RoundKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = nearbyintf(input[i]);
  }
  return;
}
template <>
__global__ void RoundKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = nearbyint(input[i]);
  }
  return;
}
template <typename T>
__global__ void SignKernel(const T *input, T *output, const size_t count) {
  T zero = 0.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T res;
    if (input[i] < zero) {
      res = -1;
    } else if (input[i] > zero) {
      res = 1;
    } else {
      res = 0;
    }
    output[i] = static_cast<T>(res);
  }
  return;
}
template <>
__global__ void SignKernel(const uint8_t *input, uint8_t *output, const size_t count) {
  uint8_t zero = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    uint8_t res;
    if (input[i] > zero) {
      res = 1;
    } else {
      res = 0;
    }
    output[i] = res;
  }
  return;
}
template <>
__global__ void SignKernel(const uint16_t *input, uint16_t *output, const size_t count) {
  uint16_t zero = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    uint16_t res;
    if (input[i] > zero) {
      res = 1;
    } else {
      res = 0;
    }
    output[i] = res;
  }
  return;
}
template <>
__global__ void SignKernel(const uint32_t *input, uint32_t *output, const size_t count) {
  uint32_t zero = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    uint32_t res;
    if (input[i] > zero) {
      res = 1;
    } else {
      res = 0;
    }
    output[i] = res;
  }
  return;
}
template <>
__global__ void SignKernel(const uint64_t *input, uint64_t *output, const size_t count) {
  uint64_t zero = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    uint64_t res;
    if (input[i] > zero) {
      res = 1;
    } else {
      res = 0;
    }
    output[i] = res;
  }
  return;
}
template <typename T>
__global__ void SignKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  T zero = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T real = input[i].real();
    T imag = input[i].imag();
    T sum = sqrt(real * real + imag * imag);
    if (sum != zero) {
      output[i] = Complex<T>(real / sum, imag / sum);
    } else {
      output[i] = 0;
    }
  }
  return;
}
template <typename T>
void Expm1(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  Expm1Kernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Log1p(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  Log1pKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Erf(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ErfKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Erfc(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ErfcKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Invert(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  InvertKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sin(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SinKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sin(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream) {
  SinKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sinh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SinhKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Tan(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  TanKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Cos(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  CosKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Cos(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream) {
  CosKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Cosh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  CoshKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Asin(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AsinKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Asin(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream) {
  AsinKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void ACos(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ACosKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void ACos(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream) {
  ACosKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Atan(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AtanKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Atan(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream) {
  AtanKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Asinh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AsinhKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Asinh(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream) {
  AsinhKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Acosh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AcoshKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Acosh(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream) {
  AcoshKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Atanh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AtanhKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Rsqrt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RsqrtKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Abs(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AbsKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Abs(const Complex<T> *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AbsKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Real(const Complex<T> *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RealKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Real(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RealKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Imag(const Complex<T> *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ImagKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Imag(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ImagKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Conj(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream) {
  ConjKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Conj(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ConjKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Floor(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  FloorKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Trunc(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  TruncKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Ceil(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  CeilKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Rint(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RintKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Round(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RoundKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sign(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SignKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}

// double
template CUDA_LIB_EXPORT void Expm1<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<double>(const double *input, double *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<double>(const double *input, double *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<double>(const double *input, double *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<double>(const double *input, double *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<double>(const double *input, double *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<double>(const double *input, double *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);

// float
template CUDA_LIB_EXPORT void Expm1<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<float>(const float *input, float *output, const size_t count,
                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<float>(const float *input, float *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<float>(const float *input, float *output, const size_t count,
                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<float>(const float *input, float *output, const size_t count,
                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<float>(const float *input, float *output, const size_t count,
                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<float>(const float *input, float *output, const size_t count,
                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);

// half
template CUDA_LIB_EXPORT void Expm1<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<half>(const half *input, half *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);

// int8
template CUDA_LIB_EXPORT void Expm1<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<char>(const char *input, char *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);

// uint8
template CUDA_LIB_EXPORT void Expm1<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<unsigned char>(const unsigned char *input, unsigned char *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);

// int32
template CUDA_LIB_EXPORT void Expm1<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);

// uint32
template CUDA_LIB_EXPORT void Expm1<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);

// int16
template CUDA_LIB_EXPORT void Expm1<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);

// uint16
template CUDA_LIB_EXPORT void Expm1<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);

// int64
template CUDA_LIB_EXPORT void Expm1<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);

// uint64
template CUDA_LIB_EXPORT void Expm1<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);

// complex64
template CUDA_LIB_EXPORT void Abs<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                  const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<float>(const Complex<float> *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<float>(const Complex<float> *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<float>(const Complex<float> *input, Complex<float> *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                  const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                  const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                  const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Expm1<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                    const size_t count, cudaStream_t cuda_stream);

// complex128
template CUDA_LIB_EXPORT void Abs<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<double>(const Complex<double> *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<double>(const Complex<double> *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<double>(const Complex<double> *input, Complex<double> *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                     const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                     const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                     const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                     const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Expm1<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                     const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                     const size_t count, cudaStream_t cuda_stream);

// bool
template CUDA_LIB_EXPORT void Expm1<bool>(const bool *input, bool *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<bool>(const bool *input, bool *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Invert<bool>(const bool *input, bool *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sinh<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Tan<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cosh<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<bool>(const bool *input, bool *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<bool>(const bool *input, bool *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atanh<bool>(const bool *input, bool *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<bool>(const bool *input, bool *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<bool>(const bool *input, bool *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Trunc<bool>(const bool *input, bool *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Ceil<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<bool>(const bool *input, bool *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);

// int16
template CUDA_LIB_EXPORT void Real<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);

// uint16
template CUDA_LIB_EXPORT void Real<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);

// uint32
template CUDA_LIB_EXPORT void Real<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);

// int64
template CUDA_LIB_EXPORT void Real<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);

// uint64
template CUDA_LIB_EXPORT void Real<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
