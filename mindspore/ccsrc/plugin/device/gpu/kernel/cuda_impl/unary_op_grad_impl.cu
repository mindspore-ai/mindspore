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

#include "unary_op_grad_impl.cuh"

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
__global__ void ACosGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T neg_one = -1;
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
__global__ void AsinhGradKernel(const T *input, const T *dout, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float inputf = static_cast<float>(input[i]);
    T coshy = static_cast<T>(coshf(inputf));
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
void ACosGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  ACosGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AtanGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  AtanGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AsinhGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  AsinhGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void AcoshGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  AcoshGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template <typename T>
void ReciprocalGrad(const T *input, const T *dout, T *output, const size_t count, cudaStream_t cuda_stream) {
  ReciprocalGradKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, dout, output, count);
  return;
}

template void SqrtGrad<double>(const double *input, const double *dout, double *output, const size_t count,
                               cudaStream_t cuda_stream);
template void RsqrtGrad<double>(const double *input, const double *dout, double *output, const size_t count,
                                cudaStream_t cuda_stream);
template void AsinGrad<double>(const double *input, const double *dout, double *output, const size_t count,
                               cudaStream_t cuda_stream);
template void ACosGrad<double>(const double *input, const double *dout, double *output, const size_t count,
                               cudaStream_t cuda_stream);
template void AtanGrad<double>(const double *input, const double *dout, double *output, const size_t count,
                               cudaStream_t cuda_stream);
template void AsinhGrad<double>(const double *input, const double *dout, double *output, const size_t count,
                                cudaStream_t cuda_stream);
template void AcoshGrad<double>(const double *input, const double *dout, double *output, const size_t count,
                                cudaStream_t cuda_stream);
template void ReciprocalGrad<double>(const double *input, const double *dout, double *output, const size_t count,
                                     cudaStream_t cuda_stream);

template void SqrtGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                              cudaStream_t cuda_stream);
template void RsqrtGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                               cudaStream_t cuda_stream);
template void AsinGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                              cudaStream_t cuda_stream);
template void ACosGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                              cudaStream_t cuda_stream);
template void AtanGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                              cudaStream_t cuda_stream);
template void AsinhGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                               cudaStream_t cuda_stream);
template void AcoshGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                               cudaStream_t cuda_stream);
template void ReciprocalGrad<float>(const float *input, const float *dout, float *output, const size_t count,
                                    cudaStream_t cuda_stream);
template void SqrtGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                             cudaStream_t cuda_stream);
template void RsqrtGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                              cudaStream_t cuda_stream);
template void AsinGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                             cudaStream_t cuda_stream);
template void ACosGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                             cudaStream_t cuda_stream);
template void AtanGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                             cudaStream_t cuda_stream);
template void AsinhGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                              cudaStream_t cuda_stream);
template void AcoshGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                              cudaStream_t cuda_stream);
template void ReciprocalGrad<half>(const half *input, const half *dout, half *output, const size_t count,
                                   cudaStream_t cuda_stream);
