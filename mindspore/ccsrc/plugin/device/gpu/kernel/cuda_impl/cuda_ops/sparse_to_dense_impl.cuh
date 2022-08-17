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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_TO_DENSE_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_TO_DENSE_CUH_
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
template <typename T>
void CallSetDefaultValue(T default_value, const int64_t output_elements, T *output, const uint32_t &device_id,
                         cudaStream_t cuda_stream);

template <typename T, typename Index>
void CallSparseToDense(const Index *indices, const T *vals, const int num_elems, const int num_vals,
                       const Index *output_shape, const int ndims, T *output, const uint32_t &device_id,
                       cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_TO_DENSE_CUH_
