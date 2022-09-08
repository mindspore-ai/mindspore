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

#include <vector>
#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex_abs_impls.cuh"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "include/cuda_fp16.h"

constexpr uint elements_per_thread = 4;
constexpr uint threads_per_block = 256;
constexpr uint elements_per_block = elements_per_thread * threads_per_block;


template <typename T, typename S>
struct VectorizedTrait {
  static const uint VecSizeT = 4;
  static const uint VecSizeS = 4;
};

template <typename T, int VecSizeT>
struct alignas(sizeof(T) * VecSizeT) AlignVecIn {
  T datain[VecSizeT];
};

template <typename S, int VecSizeS>
struct alignas(sizeof(S) * VecSizeS) AlignVecOut {
  S dataout[VecSizeS];
};

template <typename Func, typename T, typename S>
__device__ __forceinline__ void VectorizedCall(Func func, const T *in, S *out) {
  constexpr uint vec_size_t = VectorizedTrait<T, S>::VecSizeT;
  constexpr uint vec_size_s = VectorizedTrait<T, S>::VecSizeS;
  constexpr uint elements_per_loop = elements_per_thread / vec_size_t;
  using VecT = AlignVecIn<T, vec_size_t>;
  using VecS = AlignVecOut<S, vec_size_s>;

  uint tid = threadIdx.x;
  auto vec_in = reinterpret_cast<const VecT *>(in);
  auto vec_out = reinterpret_cast<VecS *>(out);
  for (uint i = 0; i < elements_per_loop; i++) {
    uint index = tid + i * threads_per_block;
    VecT cache_in = vec_in[index];
    VecS cache_out = vec_out[index];
    for (uint j = 0; j < vec_size_t; j++) {
      cache_out.dataout[j] = func(cache_in.datain[j]);
    }
    vec_out[index] = cache_out;
  }
}

template <typename Func, typename T, typename S>
__device__ __forceinline__ void NormalCall(Func func, const T *in, S *out, uint remaining) {
  uint loop = UP_DIV(remaining, elements_per_thread);

  for (uint i = threadIdx.x; i < loop; i += blockDim.x) {
    for (uint j = 0; j < elements_per_thread; j++) {
      uint index = i * elements_per_thread + j;
      if (index >= remaining) {
        return;
      }
      out[index] = func(in[index]);
    }
  }
}

template <typename Func, typename T, typename S>
__global__ void VectorizedFor(Func func, const T *in, S *out, uint num_of_elements) {
  uint offset = elements_per_block * blockIdx.x;
  uint remaining = num_of_elements - offset;
  if (blockIdx.x + 1 == gridDim.x && remaining != elements_per_block) {
    NormalCall(func, in + offset, out + offset, remaining);
  } else {
    VectorizedCall(func, in + offset, out + offset);
  }
}

template <typename T, typename S>
struct ComplexAbsFunctor {
  __device__ __forceinline__ S operator()(T x) const {
    S r = x.real();
    S i = x.imag();
    return sqrt(r * r + i * i);
  }
};

template <typename T, typename S>
void ComplexAbs(const size_t input_elements, const T *x0, S *y, const uint32_t &device_id, cudaStream_t stream) {
  ComplexAbsFunctor<T, S> functor{};
  auto block_x = threads_per_block;
  auto grid_x = UP_DIV(static_cast<uint>(input_elements), elements_per_block);
  dim3 block{block_x};
  dim3 grid{grid_x};
  VectorizedFor<<<grid, block, 0, stream>>>(functor, x0, y, static_cast<uint>(input_elements));
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
template CUDA_LIB_EXPORT void ComplexAbs<Complex<float>, float>(const size_t nums, const Complex<float> *x0,
                                                                float *y, const uint32_t &device_id,
                                                                cudaStream_t stream);
template CUDA_LIB_EXPORT void ComplexAbs<Complex<double>, double>(const size_t nums, const Complex<double> *x0,
                                                                  double *y, const uint32_t &device_id,
                                                                  cudaStream_t stream);

