/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <iostream>
#include <vector>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bce_with_logits_loss_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elementswise_pub_impl.cuh"

struct StrideInfo {
  size_t inp_stride[9];
  size_t out_stride[9];
  size_t index_stride[9];
};

template <typename T>
struct FillWithoutBroadcastFunctor {
  FillWithoutBroadcastFunctor() {}
  __device__ __forceinline__ void operator()(T *dst, const T *src) const { dst[0] = src[0]; }
};

StrideInfo CalBceStride(const std::vector<int64_t> inp_shape, const std::vector<int64_t> out_shape) {
  StrideInfo strides;
  size_t out_shape_size = out_shape.size();
  size_t cur_inp_shape = 0;
  if (out_shape_size == 0) {
    strides.inp_stride[0] = 1;
    strides.inp_stride[1] = 1;
    strides.out_stride[0] = 1;
    strides.out_stride[1] = 1;
    strides.index_stride[0] = 0;
    return strides;
  }
  strides.inp_stride[out_shape_size] = 1;
  strides.out_stride[out_shape_size] = 1;
  for (int idx = out_shape_size - 1; idx >= 0; --idx) {
    strides.inp_stride[idx] = strides.inp_stride[idx + 1] * inp_shape[idx];
    strides.out_stride[idx] = strides.out_stride[idx + 1] * out_shape[idx];
    cur_inp_shape = strides.inp_stride[idx] / strides.inp_stride[idx + 1];
    strides.index_stride[idx] = (cur_inp_shape == 1) ? 0 : 1;
  }
  return strides;
}

template <typename T>
__global__ void FillAndBroadcast(const size_t size, const size_t shape_size, const StrideInfo strides, const T *src,
                                 T *dst) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    size_t tmp_pos = pos;
    size_t cur_idx = 0;
    size_t input_pos = 0;
    for (int idx = 0; idx < shape_size; ++idx) {
      cur_idx = tmp_pos / strides.out_stride[idx + 1];
      input_pos += cur_idx * strides.inp_stride[idx + 1] * strides.index_stride[idx];
      tmp_pos -= cur_idx * strides.out_stride[idx + 1];
    }
    dst[pos] = src[input_pos];
  }
  return;
}

template <typename T>
struct BCEWithLogitsLossMainFunctor {
  BCEWithLogitsLossMainFunctor() {}
  __device__ __forceinline__ void operator()(T *output, const T *predict, const T *target,
                                             const T *shape_broadcasted) const {
    T max_value = -predict[0];
    max_value = max_value > static_cast<T>(0) ? max_value : static_cast<T>(0);
    const T log_weight = (shape_broadcasted[0] - static_cast<T>(1)) * target[0] + static_cast<T>(1);
    output[0] = (static_cast<T>(1) - target[0]) * predict[0] +
                log_weight * (log(exp(-max_value) + exp(-predict[0] - max_value)) + max_value);
  }
};

template <>
struct BCEWithLogitsLossMainFunctor<half> {
  BCEWithLogitsLossMainFunctor() {}
  __device__ __forceinline__ void operator()(half *output, const half *predict, const half *target,
                                             const half *shape_broadcasted) const {
    half max_value = -predict[0];
    max_value = max_value > static_cast<half>(0) ? max_value : static_cast<half>(0);
    const half log_weight = (shape_broadcasted[0] - static_cast<half>(1)) * target[0] + static_cast<half>(1);
    output[0] = (static_cast<half>(1) - target[0]) * predict[0] +
                log_weight * (hlog(hexp(-max_value) + hexp(-predict[0] - max_value)) + max_value);
  }
};

template <typename T>
struct MulFunctor {
  MulFunctor() {}
  __device__ __forceinline__ void operator()(T *rhs, const T *lhs) const { rhs[0] *= lhs[0]; }
};

template <typename T>
cudaError_t CalBCEWithLogitsLoss(const size_t input_size, const T *predict, const T *target,
                                 const std::vector<int64_t> &input_shape, const size_t shape_size, const T *weight,
                                 const std::vector<int64_t> &weight_shape, const bool weight_need_broadcast,
                                 const T *pos_weight, const std::vector<int64_t> &pos_weight_shape,
                                 const bool pos_weight_need_broadcast, T *shape_broadcasted, T *output,
                                 cudaStream_t cuda_stream) {
  if (pos_weight_need_broadcast) {
    StrideInfo strides = CalBceStride(pos_weight_shape, input_shape);
    FillAndBroadcast<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, shape_size, strides,
                                                                              pos_weight, shape_broadcasted);
  } else {
    FillWithoutBroadcastFunctor<T> functor;
    cuda::elementwise::EltWiseCudaOpsFunc(functor, (uint)(input_size), shape_broadcasted, pos_weight, cuda_stream);
  }
  BCEWithLogitsLossMainFunctor<T> loss_functor;
  cuda::elementwise::EltWiseCudaOpsFunc(loss_functor, (uint)(input_size), output, predict, target, shape_broadcasted,
                                        cuda_stream);
  if (weight_need_broadcast) {
    StrideInfo strides = CalBceStride(weight_shape, input_shape);
    FillAndBroadcast<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, shape_size, strides, weight,
                                                                              shape_broadcasted);
  } else {
    FillWithoutBroadcastFunctor<T> functor;
    cuda::elementwise::EltWiseCudaOpsFunc(functor, (uint)(input_size), shape_broadcasted, weight, cuda_stream);
  }
  MulFunctor<T> functor;
  cuda::elementwise::EltWiseCudaOpsFunc(functor, (uint)(input_size), output, shape_broadcasted, cuda_stream);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBCEWithLogitsLoss<half>(
  const size_t input_size, const half *predict, const half *target, const std::vector<int64_t> &input_shape,
  const size_t shape_size, const half *weight, const std::vector<int64_t> &weight_shape,
  const bool weight_need_broadcast, const half *pos_weight, const std::vector<int64_t> &pos_weight_shape,
  const bool pos_weight_need_broadcast, half *shape_broadcasted, half *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBCEWithLogitsLoss<float>(
  const size_t input_size, const float *predict, const float *target, const std::vector<int64_t> &input_shape,
  const size_t shape_size, const float *weight, const std::vector<int64_t> &weight_shape,
  const bool weight_need_broadcast, const float *pos_weight, const std::vector<int64_t> &pos_weight_shape,
  const bool pos_weight_need_broadcast, float *shape_broadcasted, float *output, cudaStream_t cuda_stream);
