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

#include <vector>
#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cast_impl.cuh"
#include "include/cuda_fp16.h"

// Generic cast
template <typename S, typename T>
__device__ __forceinline__ void CastBase(const S *input_addr, T *output_addr) {
  *output_addr = static_cast<T>((*input_addr));
}

// half --> integer
__device__ __forceinline__ void CastBase(const half *input_addr, uint64_t *output_addr) {
  *output_addr = __half2ull_rz((*input_addr));
}

__device__ __forceinline__ void CastBase(const half *input_addr, int64_t *output_addr) {
  *output_addr = __half2ll_rz((*input_addr));
}

__device__ __forceinline__ void CastBase(const half *input_addr, uint32_t *output_addr) {
  *output_addr = __half2uint_rz((*input_addr));
}

__device__ __forceinline__ void CastBase(const half *input_addr, int32_t *output_addr) {
  *output_addr = __half2int_rz((*input_addr));
}

__device__ __forceinline__ void CastBase(const half *input_addr, uint16_t *output_addr) {
  *output_addr = __half2ushort_rz((*input_addr));
}

__device__ __forceinline__ void CastBase(const half *input_addr, int16_t *output_addr) {
  *output_addr = __half2short_rz((*input_addr));
}

__device__ __forceinline__ void CastBase(const half *input_addr, uint8_t *output_addr) {
  *output_addr = static_cast<uint8_t>(__half2ushort_rz((*input_addr)));
}

__device__ __forceinline__ void CastBase(const half *input_addr, int8_t *output_addr) {
  *output_addr = static_cast<int8_t>(__half2short_rz((*input_addr)));
}

// integer --> half
__device__ __forceinline__ void CastBase(const uint64_t *input_addr, half *output_addr) {
  *output_addr = __ull2half_rn((*input_addr));
}

__device__ __forceinline__ void CastBase(const int64_t *input_addr, half *output_addr) {
  *output_addr = __ll2half_rn((*input_addr));
}

__device__ __forceinline__ void CastBase(const uint32_t *input_addr, half *output_addr) {
  *output_addr = __uint2half_rn((*input_addr));
}

__device__ __forceinline__ void CastBase(const int32_t *input_addr, half *output_addr) {
  *output_addr = __int2half_rn((*input_addr));
}

__device__ __forceinline__ void CastBase(const uint16_t *input_addr, half *output_addr) {
  *output_addr = __ushort2half_rn((*input_addr));
}

__device__ __forceinline__ void CastBase(const int16_t *input_addr, half *output_addr) {
  *output_addr = __short2half_rn((*input_addr));
}

__device__ __forceinline__ void CastBase(const uint8_t *input_addr, half *output_addr) {
  *output_addr = __ushort2half_rn(static_cast<uint16_t>(*input_addr));
}

__device__ __forceinline__ void CastBase(const int8_t *input_addr, half *output_addr) {
  *output_addr = __short2half_rn(static_cast<int16_t>(*input_addr));
}

// Cast
template <typename S, typename T>
__global__ void CastKernel(const int input_size, const S *input_addr, T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size; pos += blockDim.x * gridDim.x) {
    CastBase(input_addr + pos, output_addr + pos);
  }
}

template <typename S, typename T>
void Cast(const int input_size, const S *input_addr, T *output_addr, cudaStream_t stream, uint32_t device_id) {
  CastKernel<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, stream>>>(input_size, input_addr,
                                                                                         output_addr);
}

template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int8_t *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int16_t *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int32_t *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const int64_t *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint8_t *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint16_t *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint32_t *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const uint64_t *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, half *output_addr, cudaStream_t stream,
                                   uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, bool *output_addr, cudaStream_t stream,
                                   uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const half *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const float *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const double *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, half *output_addr, cudaStream_t stream,
                                   uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, bool *output_addr, cudaStream_t stream,
                                   uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const bool *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<float> *input_addr, Complex<double> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);

template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, int8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, int16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, int32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, int64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, uint8_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, uint16_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, uint32_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, uint64_t *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, float *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, double *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, half *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, bool *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
template CUDA_LIB_EXPORT void Cast(const int input_size, const Complex<double> *input_addr, Complex<float> *output_addr,
                                   cudaStream_t stream, uint32_t device_id);
