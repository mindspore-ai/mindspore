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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_grad_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
__global__ void SqrtGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float input_f = static_cast<float>(input[i]);
    float dout_f = static_cast<float>(dout[i]);
    float res_vmul = dout_f / (2.0 * input_f);
    output[i] = static_cast<T>(res_vmul);
  }
  return;
}

template <>
__global__ void SqrtGradKernel(const Complex<float> *input, const Complex<float> *dout, Complex<float> *output,
                               const size_t count) {
  Complex<float> two = Complex<float>(2.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = dout[i] / (conj(input[i]) * two);
  }
}

template <>
__global__ void SqrtGradKernel(const Complex<double> *input, const Complex<double> *dout, Complex<double> *output,
                               const size_t count) {
  Complex<double> two = Complex<double>(2.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = dout[i] / (conj(input[i]) * two);
  }
}

template <typename T>
__global__ void RsqrtGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float input_f = static_cast<float>(input[i]);
    float dout_f = static_cast<float>(dout[i]);
    float res_vmul = input_f * input_f * input_f;
    res_vmul = -0.5 * res_vmul * dout_f;
    output[i] = static_cast<T>(res_vmul);
  }
  return;
}

template <typename T>
__global__ void AsinGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T one = 1;
    T sqt = sqrtf(one - input[i] * input[i]);
    output[i] = dout[i] / sqt;
  }
  return;
}

template <>
__global__ void AsinGradKernel(const half *input, const half *dout, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    half one = 1;
    half sqt = hsqrt(one - input[i] * input[i]);
    output[i] = dout[i] / sqt;
  }
  return;
}

template <typename T>
__global__ void AsinGradKernel(const Complex<T> *input, const Complex<T> *dout,
                               Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    Complex<T> one = Complex<T> (1);
    Complex<T> sqt = sqrt(one - input[i] * input[i]);
    sqt = Complex<T>(sqt.real(), -sqt.imag());
    output[i] = dout[i] / sqt;
  }
  return;
}

template <>
__global__ void AsinGradKernel(const double *input, const double *dout, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    double one = static_cast<double> (1);
    double sqt = static_cast<double> (sqrt(one - input[i] * input[i]));
    output[i] = dout[i] / sqt;
  }
  return;
}

template <typename T>
__global__ void ACosGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T neg_one = static_cast<T>(-1);
    T one = 1;
    T sqt = sqrtf(one - input[i] * input[i]);
    output[i] = neg_one * dout[i] / sqt;
  }
  return;
}

template <>
__global__ void ACosGradKernel(const half *input, const half *dout, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    half neg_one = -1;
    half one = 1;
    half sqt = hsqrt(one - input[i] * input[i]);
    output[i] = neg_one * dout[i] / sqt;
  }
  return;
}

template <>
__global__ void ACosGradKernel(const double *input, const double *dout, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    double neg_one = static_cast<double>(-1);
    double one = 1;
    double sqt = sqrt(one - input[i] * input[i]);
    output[i] = neg_one * dout[i] / sqt;
  }
  return;
}

template <typename T>
__global__ void ACosGradKernel(const Complex<T> *input, const Complex<T> *dout,
                               Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    Complex<T> neg_one = Complex<T>(-1);
    Complex<T> one = Complex<T>(1);
    Complex<T> sqt = sqrt(one - input[i] * input[i]);
    sqt = Complex<T>(sqt.real(), -sqt.imag());
    output[i] = neg_one * dout[i] / sqt;
  }
  return;
}

template <typename T>
__global__ void AtanGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T one = 1;
    T divisor = one + input[i] * input[i];
    output[i] = dout[i] / divisor;
  }
  return;
}

template <typename T>
__global__ void TanhGradKernel(const T *__restrict__ input, const T *dout, T *output, const size_t count) {
  const T one = static_cast<T>(1);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T divisor = one - input[i] * input[i];
    output[i] = dout[i] * divisor;
  }
  return;
}

template <typename T>
__global__ void AsinhGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float inputf = static_cast<float>(input[i]);
    T coshy = static_cast<T>(coshf(inputf));
    output[i] = dout[i] / coshy;
  }
  return;
}

template <typename T>
__global__ void AsinhGradKernel(const Complex<T> *input, const Complex<T> *dout,
                                Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    Complex<T> inputf = input[i];
    Complex<T> coshy = cosh(inputf);
    coshy = Complex<T>(coshy.real(), -coshy.imag());
    output[i] = dout[i] / coshy;
  }
  return;
}

template <>
__global__ void AsinhGradKernel(const double *input, const double *dout, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    double coshy = cosh(input[i]);
    output[i] = dout[i] / coshy;
  }
  return;
}

template <typename T>
__global__ void AcoshGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float inputf = static_cast<float>(input[i]);
    T sinhy = static_cast<T>(sinhf(inputf));
    output[i] = dout[i] / sinhy;
  }
  return;
}

template <>
__global__ void AcoshGradKernel(const double *input, const double *dout, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    double inputf = static_cast<double>(input[i]);
    double sinhy = static_cast<double>(sinh(inputf));
    output[i] = dout[i] / sinhy;
  }
  return;
}

template <typename T>
__global__ void AcoshGradKernel(const Complex<T> *input, const Complex<T> *dout, Complex<T> *output,
                                const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    Complex<T> inputf = input[i];
    Complex<T> sinhy = sinh(inputf);
    sinhy = Complex<T>(sinhy.real(), -sinhy.imag());
    output[i] = dout[i] / sinhy;
  }
  return;
}

template <typename T>
__global__ void ReciprocalGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    float inputf = static_cast<float>(input[i]);
    float doutf = static_cast<float>(dout[i]);
    float res = -1 * doutf * inputf * inputf;
    output[i] = static_cast<T>(res);
  }
  return;
}

template <>
__global__ void ReciprocalGradKernel(const double *input, const double *dout, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    double neg_one = static_cast<double>(-1);
    output[i] = neg_one * dout[i] * input[i] * input[i];
  }
  return;
}

template <>
__global__ void ReciprocalGradKernel(const Complex<float> *input, const Complex<float> *dout, Complex<float> *output,
                                     const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    Complex<float> neg_one = static_cast<Complex<float>>(-1);
    output[i] = neg_one * dout[i] * input[i] * input[i];
  }
  return;
}

template <>
__global__ void ReciprocalGradKernel(const Complex<double> *input, const Complex<double> *dout, Complex<double> *output,
                                     const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    Complex<double> neg_one = static_cast<Complex<double>>(-1);
    output[i] = neg_one * dout[i] * input[i] * input[i];
  }
  return;
}

template <typename T>
void SqrtGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  SqrtGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void RsqrtGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  RsqrtGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AsinGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  AsinGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AsinGrad(const Complex<T> *input, const Complex<T> *dout, Complex<T> *output, const size_t count,
              cudaStream_t cuda_stream) {
  AsinGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void ACosGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  ACosGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void ACosGrad(const Complex<T> *input, const Complex<T> *dout, Complex<T> *output, const size_t count,
               cudaStream_t cuda_stream) {
  ACosGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AtanGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  AtanGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void TanhGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  TanhGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AsinhGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  AsinhGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AsinhGrad(const Complex<T> *input, const Complex<T> *dout, Complex<T> *output, const size_t count,
               cudaStream_t cuda_stream) {
  AsinhGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AcoshGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  AcoshGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AcoshGrad(const Complex<T> *input, const Complex<T> *dout, Complex<T> *output, const size_t count,
               cudaStream_t cuda_stream) {
  AcoshGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void ReciprocalGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  ReciprocalGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void InvGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  ReciprocalGrad<T>(input, dout, output, count, cuda_stream);
  return;
}

template CUDA_LIB_EXPORT void TanhGrad<Complex<double>>(const Complex<double> *input, const Complex<double> *dout,
                                                        Complex<double> *output, const size_t count,
                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void TanhGrad<Complex<float>>(const Complex<float> *input, const Complex<float> *dout,
                                                       Complex<float> *output, const size_t count,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void SqrtGrad<Complex<double>>(const Complex<double> *input, const Complex<double> *dout,
                                                        Complex<double> *output, const size_t count,
                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void SqrtGrad<Complex<float>>(const Complex<float> *input, const Complex<float> *dout,
                                                       Complex<float> *output, const size_t count,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void SqrtGrad<double>(const double *input, const double *dout, double *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void RsqrtGrad<double>(const double *input, const double *dout, double *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<double>(const double *input, const double *dout, double *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<double>(const double *input, const double *dout, double *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AtanGrad<double>(const double *input, const double *dout, double *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TanhGrad<double>(const double *input, const double *dout, double *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<double>(const double *input, const double *dout, double *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<double>(const double *input, const double *dout, double *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<double>(const double *input, const double *dout, double *output,
                                                     const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<double>(const double *input, const double *dout, double *output,
                                              const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SqrtGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void RsqrtGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AtanGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TanhGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<float>(const float *input, const float *dout, float *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SqrtGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void RsqrtGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AtanGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TanhGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<half>(const half *input, const half *dout, half *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SqrtGrad<char>(const char *input, const char *dout, char *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void RsqrtGrad<char>(const char *input, const char *dout, char *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<char>(const char *input, const char *dout, char *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<char>(const char *input, const char *dout, char *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AtanGrad<char>(const char *input, const char *dout, char *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TanhGrad<char>(const char *input, const char *dout, char *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<char>(const char *input, const char *dout, char *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<char>(const char *input, const char *dout, char *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<char>(const char *input, const char *dout, char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<char>(const char *input, const char *dout, char *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SqrtGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                      unsigned char *output, const size_t count,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void RsqrtGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                       unsigned char *output, const size_t count,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                      unsigned char *output, const size_t count,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                      unsigned char *output, const size_t count,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AtanGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                      unsigned char *output, const size_t count,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TanhGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                      unsigned char *output, const size_t count,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                       unsigned char *output, const size_t count,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                       unsigned char *output, const size_t count,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                            unsigned char *output, const size_t count,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<unsigned char>(const unsigned char *input, const unsigned char *dout,
                                                     unsigned char *output, const size_t count,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SqrtGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void RsqrtGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AtanGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TanhGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                                     const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<int8_t>(const int8_t *input, const int8_t *dout, int8_t *output,
                                              const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SqrtGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void RsqrtGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                                 const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AtanGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TanhGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                                 const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                                 const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                                      const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<int16_t>(const int16_t *input, const int16_t *dout, int16_t *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SqrtGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void RsqrtGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AtanGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TanhGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<int>(const int *input, const int *dout, int *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SqrtGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void RsqrtGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                                 const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AtanGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void TanhGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                                const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                                 const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                                 const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                                      const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<int64_t>(const int64_t *input, const int64_t *dout, int64_t *output,
                                               const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<Complex<float>>(const Complex<float> *input, const Complex<float> *dout,
                                                       Complex<float> *output, const size_t count,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACosGrad<Complex<double>>(const Complex<double> *input, const Complex<double> *dout,
                                                       Complex<double> *output, const size_t count,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<Complex<float>>(const Complex<float> *input, const Complex<float> *dout,
                                                        Complex<float> *output, const size_t count,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AcoshGrad<Complex<double>>(const Complex<double> *input, const Complex<double> *dout,
                                                        Complex<double> *output, const size_t count,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<Complex<float>>(const Complex<float> *input, const Complex<float> *dout,
                                                       Complex<float> *output, const size_t count,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinGrad<Complex<double>>(const Complex<double> *input, const Complex<double> *dout,
                                                       Complex<double> *output, const size_t count,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<Complex<float>>(const Complex<float> *input, const Complex<float> *dout,
                                                        Complex<float> *output, const size_t count,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void AsinhGrad<Complex<double>>(const Complex<double> *input, const Complex<double> *dout,
                                                        Complex<double> *output, const size_t count,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ReciprocalGrad<Complex<float>>(const Complex<float> *input, const Complex<float> *dout,
                                                             Complex<float> *output, const size_t count,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<Complex<float>>(const Complex<float> *input, const Complex<float> *dout,
                                                      Complex<float> *output, const size_t count,
                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ReciprocalGrad<Complex<double>>(const Complex<double> *input, const Complex<double> *dout,
                                                              Complex<double> *output, const size_t count,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void InvGrad<Complex<double>>(const Complex<double> *input, const Complex<double> *dout,
                                                       Complex<double> *output, const size_t count,
                                                       cudaStream_t cuda_stream);
