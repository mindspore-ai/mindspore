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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SOFTMAX_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SOFTMAX_IMPL_CUH_
#include <cuda_fp16.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T, bool is_cuda>
struct AccumulateType {};
template <>
struct AccumulateType<half, true> {
  using type = float;
};
template <>
struct AccumulateType<float, true> {
  using type = float;
};
template <>
struct AccumulateType<double, true> {
  using type = float;
};

template <typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

template <typename T, typename AccumT, typename OutT, bool is_log_softmax>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
      : max_input(max_input), sum(is_log_softmax == true ? std::log(sum) : sum) {}
  __device__ __forceinline__ OutT operator()(T input) const {
    return is_log_softmax == true ? static_cast<OutT>((AccumT)input - max_input - sum)
                                  : static_cast<OutT>(std::exp((AccumT)input - max_input) / sum);
  }
  const AccumT max_input;
  const AccumT sum;
};

// aligned vector generates vectorized load/store on CUDA
template <typename scalar_t>
struct alignas(sizeof(scalar_t) * sizeof(float4) / sizeof(scalar_t)) aligned_vector {
  scalar_t val[sizeof(float4) / sizeof(scalar_t)];
};

template <typename T, bool is_log_softmax>
cudaError_t Softmax(T *input_, T *output_, size_t dim_size_, size_t outer_size_, size_t inner_size_, size_t device_id,
                    cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SOFTMAX_IMPL_CUH_
