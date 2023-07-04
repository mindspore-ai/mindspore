/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_REDUCE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_REDUCE_IMPL_CUH_
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

typedef enum {
  ReduceSum = 0,
  ReduceProd = 1,
  ReduceMax = 2,
  ReduceMin = 3,
  ReduceMean = 4,
  ReduceAny = 5,
  ReduceAll = 6,
} ReduceType_t;

template <typename T>
CUDA_LIB_EXPORT cudaError_t ArrayReduce(T *input, const std::vector<size_t> &input_reshape,
                                        const bool reduce_first_axis, ReduceType_t type, T *temp, T *output,
                                        cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t ArrayReduceComplex(T *input, const std::vector<size_t> &input_reshape,
                                               const bool reduce_first_axis, ReduceType_t type, T *temp, T *output,
                                               cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_REDUCE_IMPL_CUH_
