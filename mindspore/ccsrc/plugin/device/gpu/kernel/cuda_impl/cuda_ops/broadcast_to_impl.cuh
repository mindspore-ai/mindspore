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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BROADCAST_TO_OPT_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BROADCAST_TO_OPT_IMPL_CUH_
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

struct UnaryBroadcastStrideInfo {
  size_t input_stride[8];
  size_t output_stride[8];
};
UnaryBroadcastStrideInfo UnaryBroadcastCalStride(const size_t dim_size, const std::vector<int64_t> &inp_shape,
                                                 const std::vector<int64_t> &out_shape);

struct BinaryBroadcastStrideInfo {
  size_t in0_stride[8];
  size_t in1_stride[8];
  size_t out_stride[8];
};
BinaryBroadcastStrideInfo BinaryBroadcastCalStride(const size_t dim_size, const std::vector<int64_t> &in0_shape,
                                                   const std::vector<int64_t> &in1_shape,
                                                   const std::vector<int64_t> &out_shape);

struct TrinaryBroadcastStrideInfo {
  size_t in0_stride[8];
  size_t in1_stride[8];
  size_t in2_stride[8];
  size_t out_stride[8];
};
TrinaryBroadcastStrideInfo TrinaryBroadcastCalStride(const size_t dim_size, const std::vector<int64_t> &in0_shape,
                                                     const std::vector<int64_t> &in1_shape,
                                                     const std::vector<int64_t> &in2_shape,
                                                     const std::vector<int64_t> &out_shape);

template <typename T>
__global__ void BroadcastToCpyCuda(size_t dim_size, size_t output_num, UnaryBroadcastStrideInfo strides, T *input,
                                   T *output);

template <typename T>
CUDA_LIB_EXPORT cudaError_t BroadcastTo(const std::vector<int64_t> &inp_shape, const std::vector<int64_t> &out_shape,
                                        T *input, T *output, size_t device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BROADCAST_TO_OPT_IMPL_CUH_
