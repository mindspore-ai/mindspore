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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADDCDIV_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADDCDIV_IMPL_CUH_
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "include/cuda_fp16.h"

template <typename T, typename VT>
CUDA_LIB_EXPORT cudaError_t CalAddcdiv(const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims,
                                       const std::vector<int64_t> &x2_dims, const std::vector<int64_t> &value_dims,
                                       const std::vector<int64_t> &output_dims, const T *input_data, const T *x1,
                                       const T *x2, const VT *value, T *output, const uint32_t &device_id,
                                       cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADDCDIV_IMPL_CUH_
