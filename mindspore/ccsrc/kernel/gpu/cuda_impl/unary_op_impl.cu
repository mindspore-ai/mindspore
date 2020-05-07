/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
__global__ void ExponentialKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = exp(input[i]);
  }
  return;
}
template <>
__global__ void ExponentialKernel(half *input, half *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hexp(input[i]);
  }
  return;
}
template <typename T>
__global__ void LogarithmKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = logf(input[i]);
  }
  return;
}
template <typename T>
__global__ void NegativeKernel(T *input, T *output, size_t count) {
  T neg_one = -1;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = neg_one * input[i];
  }
  return;
}
template <typename T>
__global__ void ReciprocalKernel(T *input, T *output, size_t count) {
  T one = 1.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = one / input[i];
  }
  return;
}
template <typename T>
__global__ void SquareKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] * input[i];
  }
  return;
}
template <typename T>
__global__ void SqrtKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sqrt(input[i]);
  }
  return;
}
template <>
__global__ void SqrtKernel(half *input, half *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hsqrt(input[i]);
  }
  return;
}
template <typename T>
__global__ void RsqrtKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rsqrt(input[i]);
  }
  return;
}
template <>
__global__ void RsqrtKernel(half *input, half *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hrsqrt(input[i]);
  }
  return;
}
template <typename T>
__global__ void ZeroslikeKernel(T *output, size_t count) {
  T zero = 0.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = zero;
  }
  return;
}
template <typename T>
void Exponential(T *input, T *output, size_t count, cudaStream_t cuda_stream) {
  ExponentialKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Logarithm(T *input, T *output, size_t count, cudaStream_t cuda_stream) {
  LogarithmKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Negative(T *input, T *output, size_t count, cudaStream_t cuda_stream) {
  NegativeKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Reciprocal(T *input, T *output, size_t count, cudaStream_t cuda_stream) {
  ReciprocalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Square(T *input, T *output, size_t count, cudaStream_t cuda_stream) {
  SquareKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Pow(T *input, T *output, size_t count, cudaStream_t cuda_stream) {
  PowKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sqrt(T *input, T *output, size_t count, cudaStream_t cuda_stream) {
  SqrtKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Rsqrt(T *input, T *output, size_t count, cudaStream_t cuda_stream) {
  RsqrtKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Zeroslike(T *output, size_t count, cudaStream_t cuda_stream) {
  ZeroslikeKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(output, count);
  return;
}

template void Exponential<float>(float *input, float *output, size_t count, cudaStream_t cuda_stream);
template void Logarithm<float>(float *input, float *output, size_t count, cudaStream_t cuda_stream);
template void Negative<float>(float *input, float *output, size_t count, cudaStream_t cuda_stream);
template void Reciprocal<float>(float *input, float *output, size_t count, cudaStream_t cuda_stream);
template void Square<float>(float *input, float *output, size_t count, cudaStream_t cuda_stream);
template void Sqrt<float>(float *input, float *output, size_t count, cudaStream_t cuda_stream);
template void Rsqrt<float>(float *input, float *output, size_t count, cudaStream_t cuda_stream);
template void Zeroslike<float>(float *output, size_t count, cudaStream_t cuda_stream);
template void Exponential<half>(half *input, half *output, size_t count, cudaStream_t cuda_stream);
template void Logarithm<half>(half *input, half *output, size_t count, cudaStream_t cuda_stream);
template void Negative<half>(half *input, half *output, size_t count, cudaStream_t cuda_stream);
template void Reciprocal<half>(half *input, half *output, size_t count, cudaStream_t cuda_stream);
template void Square<half>(half *input, half *output, size_t count, cudaStream_t cuda_stream);
template void Sqrt<half>(half *input, half *output, size_t count, cudaStream_t cuda_stream);
template void Rsqrt<half>(half *input, half *output, size_t count, cudaStream_t cuda_stream);
template void Zeroslike<half>(half *output, size_t count, cudaStream_t cuda_stream);
