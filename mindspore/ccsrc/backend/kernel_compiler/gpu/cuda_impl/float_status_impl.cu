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

#include "include/cuda_runtime.h"
#include "backend/kernel_compiler/gpu/cuda_impl/float_status_impl.cuh"

template <typename T>
__global__ void IsNan(const size_t size, const T* input, bool* out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isnan(input[pos])) {
      out[pos] = true;
    } else {
      out[pos] = false;
    }
  }
  return;
}
template <>
__global__ void IsNan(const size_t size, const half* input, bool* out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (__hisnan(input[pos])) {
      out[pos] = true;
    } else {
      out[pos] = false;
    }
  }
  return;
}

template <typename T>
__global__ void IsInf(const size_t size, const T* input, bool* out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isinf(input[pos]) != 0) {
      out[pos] = true;
    } else {
      out[pos] = false;
    }
  }
  return;
}
template <>
__global__ void IsInf(const size_t size, const half* input, bool* out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (__hisinf(input[pos]) != 0) {
      out[pos] = true;
    } else {
      out[pos] = false;
    }
  }
  return;
}

template <typename T>
__global__ void IsFinite(const size_t size, const T* input, bool* out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isinf(input[pos]) == 0 && !isnan(input[pos])) {
      out[pos] = true;
    } else {
      out[pos] = false;
    }
  }
  return;
}
template <>
__global__ void IsFinite(const size_t size, const half* input, bool* out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (__hisinf(input[pos]) == 0 && !__hisnan(input[pos])) {
      out[pos] = true;
    } else {
      out[pos] = false;
    }
  }
  return;
}

template <typename T>
__global__ void FloatStatus(const size_t size, const T* input, float* out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isinf(input[pos]) != 0 || isnan(input[pos])) {
      out[0] = 1;
    }
  }
  return;
}
template <>
__global__ void FloatStatus(const size_t size, const half* input, float* out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (__hisinf(input[pos]) != 0 || __hisnan(input[pos])) {
      out[0] = 1;
    }
  }
  return;
}

template <typename T>
void CalFloatStatus(const size_t size, const T* input, float* output, cudaStream_t cuda_stream) {
  FloatStatus<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return;
}
template <typename T>
void CalIsNan(const size_t size, const T* input, bool* output, cudaStream_t cuda_stream) {
  IsNan<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return;
}
template <typename T>
void CalIsInf(const size_t size, const T* input, bool* output, cudaStream_t cuda_stream) {
  IsInf<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return;
}
template <typename T>
void CalIsFinite(const size_t size, const T* input, bool* output, cudaStream_t cuda_stream) {
  IsFinite<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return;
}

template void CalFloatStatus<float>(const size_t size, const float* input, float* output, cudaStream_t cuda_stream);
template void CalFloatStatus<half>(const size_t size, const half* input, float* output, cudaStream_t cuda_stream);
template void CalIsInf<float>(const size_t size, const float* input, bool* output, cudaStream_t cuda_stream);
template void CalIsInf<half>(const size_t size, const half* input, bool* output, cudaStream_t cuda_stream);
template void CalIsNan<float>(const size_t size, const float* input, bool* output, cudaStream_t cuda_stream);
template void CalIsNan<half>(const size_t size, const half* input, bool* output, cudaStream_t cuda_stream);
template void CalIsFinite<float>(const size_t size, const float* input, bool* output, cudaStream_t cuda_stream);
template void CalIsFinite<half>(const size_t size, const half* input, bool* output, cudaStream_t cuda_stream);
