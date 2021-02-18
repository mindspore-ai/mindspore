/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "unary_op_impl.cuh"
template <typename T>
__global__ void ExponentialKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = exp(input[i]);
  }
  return;
}
template <>
__global__ void ExponentialKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hexp(input[i]);
  }
  return;
}
template <typename T>
__global__ void Expm1Kernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = static_cast<T>(expm1f(static_cast<float>(input[i])));
  }
  return;
}
template <typename T>
__global__ void LogarithmKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = logf(input[i]);
  }
  return;
}
template <>
__global__ void LogarithmKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hlog(input[i]);
  }
  return;
}
template <typename T>
__global__ void Log1pKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = static_cast<T>(log1pf(static_cast<float>(input[i])));
  }
  return;
}
template <typename T>
__global__ void ErfKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = static_cast<T>(erff(static_cast<float>(input[i])));
  }
  return;
}
template <typename T>
__global__ void ErfcKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = static_cast<T>(erfcf(static_cast<float>(input[i])));
  }
  return;
}
template <typename T>
__global__ void NegativeKernel(const T *input, T *output, const size_t count) {
  T neg_one = -1;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = neg_one * input[i];
  }
  return;
}
template <typename T>
__global__ void ReciprocalKernel(const T *input, T *output, const size_t count) {
  T one = 1.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = one / input[i];
  }
  return;
}
template <typename T>
__global__ void SquareKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] * input[i];
  }
  return;
}
template <typename T>
__global__ void SqrtKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sqrt(input[i]);
  }
  return;
}
template <>
__global__ void SqrtKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hsqrt(input[i]);
  }
  return;
}
template <typename T>
__global__ void RsqrtKernel(const T *input, T *output, const size_t count) {
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
__global__ void SinKernel(const T *input, T *output, const size_t count) {
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
__global__ void AsinKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float inputf = static_cast<float>(input[i]);
    T res = static_cast<T>(asinf(inputf));
    output[i] = res;
  }
  return;
}
template <typename T>
__global__ void AsinhKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float inputf = static_cast<float>(input[i]);
    T res = static_cast<T>(asinhf(inputf));
    output[i] = res;
  }
  return;
}
template <typename T>
__global__ void CosKernel(const T *input, T *output, const size_t count) {
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
__global__ void ACosKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float inputf = static_cast<float>(input[i]);
    T res = static_cast<T>(acosf(inputf));
    output[i] = res;
  }
  return;
}
template <typename T>
__global__ void AcoshKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float inputf = static_cast<float>(input[i]);
    T res = static_cast<T>(acoshf(inputf));
    output[i] = res;
  }
  return;
}
template <typename T>
__global__ void AtanKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float inputf = static_cast<float>(input[i]);
    T res = static_cast<T>(atanf(inputf));
    output[i] = res;
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
template <typename T>
__global__ void FloorKernel(const T *input, T *output, const size_t count) {
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
void Exponential(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ExponentialKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Expm1(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  Expm1Kernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Logarithm(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  LogarithmKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
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
void Negative(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  NegativeKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Reciprocal(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ReciprocalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Square(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SquareKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Pow(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  PowKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sqrt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SqrtKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sin(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SinKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Cos(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  CosKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Asin(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AsinKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void ACos(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ACosKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Atan(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AtanKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Asinh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AsinhKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Acosh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AcoshKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
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
void Floor(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  FloorKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}

// double
template void Exponential<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Expm1<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Logarithm<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Log1p<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Erf<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Erfc<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Negative<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Reciprocal<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Square<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Sqrt<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Sin<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Cos<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Asin<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void ACos<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Atan<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Asinh<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Acosh<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Rsqrt<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Abs<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);
template void Floor<double>(const double *input, double *output, const size_t count, cudaStream_t cuda_stream);


// float
template void Exponential<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Expm1<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Logarithm<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Log1p<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Erf<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Erfc<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Negative<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Reciprocal<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Square<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Sqrt<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Sin<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Cos<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Asin<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void ACos<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Atan<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Asinh<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Acosh<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Rsqrt<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Abs<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);
template void Floor<float>(const float *input, float *output, const size_t count, cudaStream_t cuda_stream);

// half
template void Exponential<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Expm1<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Logarithm<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Log1p<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Erf<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Erfc<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Negative<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Reciprocal<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Square<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Sqrt<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Sin<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Cos<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Asin<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void ACos<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Atan<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Asinh<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Acosh<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Rsqrt<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Abs<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template void Floor<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
