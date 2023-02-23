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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GATHER_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GATHER_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t Gather(const T *input, const S *index, T *output, const size_t dim_before_axis,
                                   const size_t dim_at_axis_input, const size_t dim_at_axis_output,
                                   const size_t dim_after_axis, cudaStream_t stream, uint32_t device_id);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GATHER_CUH_
