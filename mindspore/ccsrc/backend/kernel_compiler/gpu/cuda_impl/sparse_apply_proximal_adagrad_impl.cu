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

#include "backend/kernel_compiler/gpu/cuda_impl/sparse_apply_proximal_adagrad_impl.cuh"

template <typename T>
__device__ __forceinline__ bool CompareFunc(T x, T y) {
  return x > y;
}

template <>
__device__ __forceinline__ bool CompareFunc(half x, half y) {
  return __half2float(x) > __half2float(y);
}

template <typename T>
__device__ __forceinline__ T RsqrtFunc(T x) {
  return __frsqrt_rn(x);
}

template <>
__device__ __forceinline__ half RsqrtFunc(half x) {
  return hrsqrt(x);
}

template <typename T>
__device__ __forceinline__ T AbsFunc(T x) {
  return abs(x);
}

template <>
__device__ __forceinline__ half AbsFunc(half x) {
  return __float2half(abs(__half2float(x)));
}

template <typename T>
__device__ __forceinline__ T Sgn(T x) {
  return static_cast<T>(x != 0 ? (x > 0 ? 1 : -1) : 0);
}

template <>
__device__ __forceinline__ half Sgn(half x) {
  return __float2half(__half2float(x) != 0 ? (__half2float(x) > 0 ? 1 : -1) : 0);
}

template <typename T>
__global__ void SparseApplyProximalAdagradUpdate(const size_t size, const size_t indices_size, const T *learning_rate,
                                                 const T *l1_regularization, const T *l2_regularization,
                                                 const T *gradient, const int *indices, T *variable, T *accumulation,
                                                 T *variable_out, T *accumulation_out) {
  const int inner_size = static_cast<int>(size / indices_size);
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < static_cast<int>(size); pos += gridDim.x * blockDim.x) {
    const int index = pos / inner_size;
    const int inner_pos = pos % inner_size;
    const int grad_pos = pos;
    const int cur_pos = indices[index] * inner_size + inner_pos;
    accumulation[cur_pos] += gradient[grad_pos] * gradient[grad_pos];
    const T scratch1 = learning_rate[0] * RsqrtFunc(accumulation[cur_pos]);
    T prox_v = variable[cur_pos];
    prox_v -= gradient[grad_pos] * scratch1;
    const T scratch2 = AbsFunc(prox_v) - scratch1 * l1_regularization[0];
    const T scratch3 = CompareFunc(scratch2, static_cast<T>(0.0)) ? scratch2 : static_cast<T>(0.0);
    variable[cur_pos] = CompareFunc(l1_regularization[0], static_cast<T>(0.0)) ? Sgn(prox_v) * scratch3 : prox_v;
    variable[cur_pos] = variable[cur_pos] / (static_cast<T>(1.0) + l2_regularization[0] * scratch1);
    accumulation_out[cur_pos] = accumulation[cur_pos];
    variable_out[cur_pos] = variable[cur_pos];
  }
}

template <typename T>
void CalSparseApplyProximalAdagrad(const size_t size, const size_t indices_size, const T *learning_rate,
                                   const T *l1_regularization, const T *l2_regularization, const T *gradient,
                                   const int *indices, T *variable, T *accumulation, T *variable_out,
                                   T *accumulation_out, cudaStream_t cuda_stream) {
  SparseApplyProximalAdagradUpdate<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, indices_size, learning_rate, l1_regularization, l2_regularization, gradient, indices, variable, accumulation,
    variable_out, accumulation_out);
}

template void CalSparseApplyProximalAdagrad<float>(const size_t size, const size_t indices_size,
                                                   const float *learning_rate, const float *l1_regularization,
                                                   const float *l2_regularization, const float *gradient,
                                                   const int *indices, float *variable, float *accumulation,
                                                   float *variable_out, float *accumulation_out,
                                                   cudaStream_t cuda_stream);
template void CalSparseApplyProximalAdagrad<half>(const size_t size, const size_t indices_size,
                                                  const half *learning_rate, const half *l1_regularization,
                                                  const half *l2_regularization, const half *gradient,
                                                  const int *indices, half *variable, half *accumulation,
                                                  half *variable_out, half *accumulation_out, cudaStream_t cuda_stream);
