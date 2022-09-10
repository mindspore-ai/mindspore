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

#include <math.h>
#include <stdint.h>
#include <complex.h>
#include "polar_impl.cuh"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementswise_op_impl.cuh"
constexpr uint kThreadsPerBlock = cuda::elementwise::kThreadsPerBlock;

template <typename R>
using Complex = mindspore::utils::Complex<R>;

template <typename T, typename S>
struct PolarFunctor {
  __device__ __forceinline__ S operator()(const T abs, const T angle) const {
    S output = 0;
    output.real(abs * std::cos(angle));
    output.imag(abs * std::sin(angle));
    return output;
  }
};

template <typename Func, uint vec_size, typename T, typename S>
__device__ __forceinline__ void NormalCall(Func func, const T *abs_addr, const T *angle_addr, S *output, uint offset,
                                           uint remaining) {
  uint loop = UP_DIV(remaining, vec_size);
  for (uint i = threadIdx.x; i < loop; i += blockDim.x) {
#pragma unroll
    for (uint j = 0; j < vec_size; j++) {
      uint index = i * vec_size + j;
      if (index >= remaining) {
        return;
      }
      index += offset;
      output[index] = func(abs_addr[index], angle_addr[index]);
    }
  }
}

template <typename Func, uint vec_size, typename T, typename S>
__device__ __forceinline__ void VectorizedCall(Func func, const T *abs_addr, const T *angle_addr, S *output,
                                               uint offset) {
  uint tid = threadIdx.x;

  using VecT = cuda::elementwise::AlignVec<T, vec_size>;
  using VecS = cuda::elementwise::AlignVec<S, vec_size>;

  auto vec_abs = reinterpret_cast<const VecT *>(abs_addr + offset);
  auto vec_angle = reinterpret_cast<const VecT *>(angle_addr + offset);
  auto vec_output = reinterpret_cast<VecS *>(output + offset);
  VecT abs = vec_abs[tid];
  VecT angle = vec_angle[tid];
  VecS out{0};

#pragma unroll
  for (uint j = 0; j < vec_size; j++) {
    out.elements_[j] = func(abs.elements_[j], angle.elements_[j]);
  }
  vec_output[tid] = out;
}

template <typename Func, uint vec_size, typename T, typename S>
__global__ void PolarVectorized(Func func, const T *abs_addr, const T *angle_addr, S *output, uint num_of_elements) {
  uint elements_per_block = kThreadsPerBlock * vec_size;
  for (uint offset = elements_per_block * blockIdx.x; offset < num_of_elements;
       offset += elements_per_block * gridDim.x) {
    uint remaining = num_of_elements - offset;
    if (remaining < elements_per_block) {
      NormalCall<Func, vec_size, T>(func, abs_addr, angle_addr, output, offset, remaining);
    } else {
      VectorizedCall<Func, vec_size, T>(func, abs_addr, angle_addr, output, offset);
    }
  }
}

template <typename T, typename S>
void CalPolar(const size_t size, const T *abs, const T *angle, S *output, const uint32_t &device_id,
              cudaStream_t cuda_stream) {
  constexpr uint vec_size = cuda::elementwise::VecSize<T>();
  const auto block_x = uint(kThreadsPerBlock);
  const uint elements_per_block = kThreadsPerBlock * vec_size;
  const auto grid_x = uint(UP_DIV(size, elements_per_block));
  dim3 block{block_x};
  dim3 grid{grid_x};
  PolarFunctor<T, S> functor{};
  PolarVectorized<PolarFunctor<T, S>, vec_size, T, S>
    <<<grid, block, 0, cuda_stream>>>(functor, abs, angle, output, size);
  return;
}


template
CUDA_LIB_EXPORT void CalPolar<float, Complex<float>>(const size_t size, const float *abs, const float *angle,
                                                     Complex<float> *output, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalPolar<double, Complex<double>>(const size_t size, const double *abs, const double *angle,
                                                       Complex<double> *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
