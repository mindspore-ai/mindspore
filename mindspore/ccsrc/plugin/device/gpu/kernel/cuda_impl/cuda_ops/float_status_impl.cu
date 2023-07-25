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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/float_status_impl.cuh"
#include "include/cuda_fp16.h"

#ifndef _MSC_VER
template <>
__device__ __forceinline__ bool isnan<bool>(bool x) {
  return isnan(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isnan<int8_t>(int8_t x) {
  return isnan(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isnan<uint8_t>(uint8_t x) {
  return isnan(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isnan<int16_t>(int16_t x) {
  return isnan(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isnan<uint16_t>(uint16_t x) {
  return isnan(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isnan<int32_t>(int32_t x) {
  return isnan(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isnan<uint32_t>(uint32_t x) {
  return isnan(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isnan<int64_t>(int64_t x) {
  return isnan(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isnan<uint64_t>(uint64_t x) {
  return isnan(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isinf<bool>(bool x) {
  return isinf(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isinf<int8_t>(int8_t x) {
  return isinf(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isinf<uint8_t>(uint8_t x) {
  return isinf(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isinf<int16_t>(int16_t x) {
  return isinf(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isinf<uint16_t>(uint16_t x) {
  return isinf(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isinf<int32_t>(int32_t x) {
  return isinf(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isinf<uint32_t>(uint32_t x) {
  return isinf(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isinf<int64_t>(int64_t x) {
  return isinf(static_cast<double>(x));
}

template <>
__device__ __forceinline__ bool isinf<uint64_t>(uint64_t x) {
  return isinf(static_cast<double>(x));
}
#endif  // _MSC_VER

template <typename T>
__global__ void IsNan(const size_t size, const T *input, bool *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isnan(static_cast<double>(input[pos]))) {
      out[pos] = true;
    } else {
      out[pos] = false;
    }
  }
  return;
}

template <>
__global__ void IsNan(const size_t size, const double *input, bool *out) {
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
__global__ void IsNan(const size_t size, const float *input, bool *out) {
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
__global__ void IsNan(const size_t size, const half *input, bool *out) {
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
__global__ void IsInf(const size_t size, const T *input, bool *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isinf(static_cast<double>(input[pos])) != 0) {
      out[pos] = true;
    } else {
      out[pos] = false;
    }
  }
  return;
}

template <>
__global__ void IsInf(const size_t size, const double *input, bool *out) {
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
__global__ void IsInf(const size_t size, const float *input, bool *out) {
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
__global__ void IsInf(const size_t size, const half *input, bool *out) {
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
__global__ void IsFinite(const size_t size, const T *input, bool *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isinf(static_cast<double>(input[pos])) == 0 && !isnan(static_cast<double>(input[pos]))) {
      out[pos] = true;
    } else {
      out[pos] = false;
    }
  }
  return;
}

template <>
__global__ void IsFinite(const size_t size, const double *input, bool *out) {
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
__global__ void IsFinite(const size_t size, const float *input, bool *out) {
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
__global__ void IsFinite(const size_t size, const half *input, bool *out) {
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
__global__ void FloatStatus(const size_t size, const T *input, float *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isinf(static_cast<double>(input[pos])) != 0 || isnan(static_cast<double>(input[pos]))) {
      out[0] = 1;
    }
  }
  return;
}

template <>
__global__ void FloatStatus(const size_t size, const double *input, float *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isinf(input[pos]) != 0 || isnan(input[pos])) {
      out[0] = 1;
    }
  }
  return;
}

template <>
__global__ void FloatStatus(const size_t size, const float *input, float *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (isinf(input[pos]) != 0 || isnan(input[pos])) {
      out[0] = 1;
    }
  }
  return;
}

template <>
__global__ void FloatStatus(const size_t size, const half *input, float *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (__hisinf(input[pos]) != 0 || __hisnan(input[pos])) {
      out[0] = 1;
    }
  }
  return;
}

template <typename T>
cudaError_t CalFloatStatus(const size_t size, const T *input, float *output, cudaStream_t cuda_stream) {
  FloatStatus<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t CalIsNan(const size_t size, const T *input, bool *output, cudaStream_t cuda_stream) {
  IsNan<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t CalIsInf(const size_t size, const T *input, bool *output, cudaStream_t cuda_stream) {
  IsInf<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t CalIsFinite(const size_t size, const T *input, bool *output, cudaStream_t cuda_stream) {
  IsFinite<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<float>(const size_t size, const float *input, float *output,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<half>(const size_t size, const half *input, float *output,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<double>(const size_t size, const double *input, float *output,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<bool>(const size_t size, const bool *input, float *output,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<int8_t>(const size_t size, const int8_t *input, float *output,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<int16_t>(const size_t size, const int16_t *input, float *output,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<int32_t>(const size_t size, const int32_t *input, float *output,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<int64_t>(const size_t size, const int64_t *input, float *output,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<uint8_t>(const size_t size, const uint8_t *input, float *output,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<uint16_t>(const size_t size, const uint16_t *input, float *output,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<uint32_t>(const size_t size, const uint32_t *input, float *output,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalFloatStatus<uint64_t>(const size_t size, const uint64_t *input, float *output,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<float>(const size_t size, const float *input, bool *output,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<half>(const size_t size, const half *input, bool *output,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<double>(const size_t size, const double *input, bool *output,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<bool>(const size_t size, const bool *input, bool *output,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<int8_t>(const size_t size, const int8_t *input, bool *output,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<int16_t>(const size_t size, const int16_t *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<int32_t>(const size_t size, const int32_t *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<int64_t>(const size_t size, const int64_t *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<uint8_t>(const size_t size, const uint8_t *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<uint16_t>(const size_t size, const uint16_t *input, bool *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<uint32_t>(const size_t size, const uint32_t *input, bool *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsInf<uint64_t>(const size_t size, const uint64_t *input, bool *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<float>(const size_t size, const float *input, bool *output,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<half>(const size_t size, const half *input, bool *output,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<double>(const size_t size, const double *input, bool *output,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<bool>(const size_t size, const bool *input, bool *output,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<int8_t>(const size_t size, const int8_t *input, bool *output,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<int16_t>(const size_t size, const int16_t *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<int32_t>(const size_t size, const int32_t *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<int64_t>(const size_t size, const int64_t *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<uint8_t>(const size_t size, const uint8_t *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<uint16_t>(const size_t size, const uint16_t *input, bool *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<uint32_t>(const size_t size, const uint32_t *input, bool *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsNan<uint64_t>(const size_t size, const uint64_t *input, bool *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<float>(const size_t size, const float *input, bool *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<half>(const size_t size, const half *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<double>(const size_t size, const double *input, bool *output,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<bool>(const size_t size, const bool *input, bool *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<int8_t>(const size_t size, const int8_t *input, bool *output,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<int16_t>(const size_t size, const int16_t *input, bool *output,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<int32_t>(const size_t size, const int32_t *input, bool *output,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<int64_t>(const size_t size, const int64_t *input, bool *output,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<uint8_t>(const size_t size, const uint8_t *input, bool *output,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<uint16_t>(const size_t size, const uint16_t *input, bool *output,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<uint32_t>(const size_t size, const uint32_t *input, bool *output,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalIsFinite<uint64_t>(const size_t size, const uint64_t *input, bool *output,
                                                           cudaStream_t cuda_stream);
