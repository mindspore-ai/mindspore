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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNIQUE_CONSECUTIVE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNIQUE_CONSECUTIVE_IMPL_CUH_

#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t CalUniqueConsecutive(const T *input, int num_elements,
                                                 const std::vector<int64_t> &input_shape, bool is_axis_none,
                                                 int64_t axis, S *input_index, S *sorted_index, S *range_data,
                                                 T *indices_data, size_t *dev_input_shape, size_t *dev_input_axis,
                                                 T *output, S *index, S *count, cudaStream_t cuda_stream,
                                                 std::vector<std::vector<int>> *res);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UNIQUE_CONSECUTIVE_IMPL_CUH_
