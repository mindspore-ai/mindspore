/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/erfinv_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

constexpr uint elements_per_thread = 4;
constexpr uint threads_per_block = 256;
constexpr uint elements_per_block = elements_per_thread * threads_per_block;

template <typename T>
struct VectorizedTrait {  // Only use of raw pointer with no offset.
  static const uint VecSize = 4;
};

template <>
struct VectorizedTrait<half> {
  static const uint VecSize = 2;
};

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) AlignVec {
  T data[VecSize];
};

template <typename Func, typename T>
__device__ __forceinline__ void VectorizedCall(Func func, const T *in, T *out) {
  constexpr uint vec_size = VectorizedTrait<T>::VecSize;
  constexpr uint elements_per_loop = elements_per_thread / vec_size;
  using VecT = AlignVec<T, vec_size>;

  uint tid = threadIdx.x;
  auto vec_in = reinterpret_cast<const VecT *>(in);
  auto vec_out = reinterpret_cast<VecT *>(out);

#pragma unroll
  for (uint i = 0; i < elements_per_loop; i++) {
    uint index = tid + i * threads_per_block;
    VecT cache = vec_in[index];
#pragma unroll
    for (uint j = 0; j < vec_size; j++) {
      cache.data[j] = func(cache.data[j]);
    }
    vec_out[index] = cache;
  }
}

template <typename Func, typename T>
__device__ __forceinline__ void NormalCall(Func func, const T *in, T *out, uint remaining) {
  uint loop = UP_DIV(remaining, elements_per_thread);
  for (uint i = threadIdx.x; i < loop; i += blockDim.x) {
#pragma unroll
    for (uint j = 0; j < elements_per_thread; j++) {
      uint index = i * elements_per_thread + j;
      if (index >= remaining) {
        return;
      }
      out[index] = func(in[index]);
    }
  }
}

template <typename Func, typename T>
__global__ void VectorizedFor(Func func, const T *in, T *out, uint num_of_elements) {
  uint offset = elements_per_block * blockIdx.x;
  uint remaining = num_of_elements - offset;

  if (blockIdx.x + 1 == gridDim.x && remaining != elements_per_block) {
    NormalCall(func, in + offset, out + offset, remaining);
  } else {
    VectorizedCall(func, in + offset, out + offset);
  }
}

template <typename T>
struct ErfinvFunctor {
  __device__ __forceinline__ T operator()(T x) const { return erfinv(x); }
};

template <>
struct ErfinvFunctor<half> {
  __device__ __forceinline__ half operator()(half x) const {
    return __float2half(erfinvf(__half2float(x)));
  }
};

template <>
struct ErfinvFunctor<double> {
  __device__ __forceinline__ double operator()(double x) const {
    return erfinvf(x);
  }
};


template <typename T>
void Erfinv(size_t input_size, const T *input, T *output,
                   const uint32_t &device_id, cudaStream_t cuda_stream) {
  ErfinvFunctor<T> functor{};
  auto block_x = threads_per_block;
  auto grid_x = UP_DIV(static_cast<uint>(input_size), elements_per_block);
  dim3 block{block_x};
  dim3 grid{grid_x};
  VectorizedFor<<<grid, block, 0, cuda_stream>>>(functor, input, output, static_cast<uint>(input_size));
}

template CUDA_LIB_EXPORT void Erfinv<float>(size_t input_size, const float *input,
                                                    float *output, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void Erfinv<half>(size_t input_size, const half *input,
                                                   half *output, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void Erfinv<double>(size_t input_size, const double *input,
                                                  double *output, const uint32_t &device_id,
                                                  cudaStream_t cuda_stream);
