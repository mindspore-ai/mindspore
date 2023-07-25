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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elementswise_pub_impl.cuh"

template <typename S, typename T>
struct CastFunctor {
  CastFunctor() {}
  __device__ __forceinline__ T operator()(S x) const { return static_cast<T>(x); }
};

template <>
struct CastFunctor<Complex<float>, bool> {
  CastFunctor() {}
  __device__ __forceinline__ bool operator()(Complex<float> x) const { return static_cast<bool>(x.real()); }
};

template <>
struct CastFunctor<Complex<double>, bool> {
  CastFunctor() {}
  __device__ __forceinline__ bool operator()(Complex<double> x) const { return static_cast<bool>(x.real()); }
};

template <>
struct CastFunctor<half, uint64_t> {
  CastFunctor() {}
  __device__ __forceinline__ uint64_t operator()(half x) const { return __half2ull_rz(x); }
};

template <>
struct CastFunctor<half, int64_t> {
  CastFunctor() {}
  __device__ __forceinline__ int64_t operator()(half x) const { return __half2ll_rz(x); }
};

template <>
struct CastFunctor<half, uint32_t> {
  CastFunctor() {}
  __device__ __forceinline__ uint32_t operator()(half x) const { return __half2uint_rz(x); }
};

template <>
struct CastFunctor<half, int32_t> {
  CastFunctor() {}
  __device__ __forceinline__ int32_t operator()(half x) const { return __half2int_rz(x); }
};

template <>
struct CastFunctor<half, uint16_t> {
  CastFunctor() {}
  __device__ __forceinline__ uint16_t operator()(half x) const { return __half2ushort_rz(x); }
};

template <>
struct CastFunctor<half, int16_t> {
  CastFunctor() {}
  __device__ __forceinline__ int16_t operator()(half x) const { return __half2short_rz(x); }
};

template <>
struct CastFunctor<half, uint8_t> {
  CastFunctor() {}
  __device__ __forceinline__ uint8_t operator()(half x) const { return static_cast<uint8_t>(__half2ushort_rz(x)); }
};

template <>
struct CastFunctor<half, int8_t> {
  CastFunctor() {}
  __device__ __forceinline__ int8_t operator()(half x) const { return static_cast<int8_t>(__half2short_rz(x)); }
};

template <>
struct CastFunctor<uint64_t, half> {
  CastFunctor() {}
  __device__ __forceinline__ half operator()(uint64_t x) const { return __ull2half_rn(x); }
};

template <>
struct CastFunctor<int64_t, half> {
  CastFunctor() {}
  __device__ __forceinline__ half operator()(int64_t x) const { return __ll2half_rn(x); }
};

template <>
struct CastFunctor<uint32_t, half> {
  CastFunctor() {}
  __device__ __forceinline__ half operator()(uint32_t x) const { return __uint2half_rn(x); }
};

template <>
struct CastFunctor<int32_t, half> {
  CastFunctor() {}
  __device__ __forceinline__ half operator()(int32_t x) const { return __int2half_rn(x); }
};

template <>
struct CastFunctor<uint16_t, half> {
  CastFunctor() {}
  __device__ __forceinline__ half operator()(uint16_t x) const { return __ushort2half_rn(x); }
};

template <>
struct CastFunctor<int16_t, half> {
  CastFunctor() {}
  __device__ __forceinline__ half operator()(int16_t x) const { return __short2half_rn(x); }
};

template <>
struct CastFunctor<uint8_t, half> {
  CastFunctor() {}
  __device__ __forceinline__ half operator()(uint8_t x) const { return __ushort2half_rn(x); }
};

template <>
struct CastFunctor<int8_t, half> {
  CastFunctor() {}
  __device__ __forceinline__ half operator()(int8_t x) const { return __short2half_rn(x); }
};

template <typename S, typename T>
cudaError_t Cast(const int input_size, const S *input, T *output, cudaStream_t cuda_stream) {
  CastFunctor<S, T> functor;
  cuda::elementwise::Unary(functor, (uint)(input_size), output, input, cuda_stream);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int8_t *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int16_t *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int32_t *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const int64_t *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint8_t *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint16_t *input_addr,
                                          Complex<double> *output_addr, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint32_t *input_addr,
                                          Complex<double> *output_addr, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const uint64_t *input_addr,
                                          Complex<double> *output_addr, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const half *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const float *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const double *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, Complex<float> *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const bool *input_addr, Complex<double> *output_addr,
                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, uint16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, uint32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, uint64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<float> *input_addr,
                                          Complex<double> *output_addr, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr, int8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr, int16_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr, int32_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr, int64_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr, uint8_t *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr,
                                          uint16_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr,
                                          uint32_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr,
                                          uint64_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr, float *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr, double *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr, half *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr, bool *output_addr,
                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Cast(const int input_size, const Complex<double> *input_addr,
                                          Complex<float> *output_addr, cudaStream_t stream);
