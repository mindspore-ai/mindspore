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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SCATTER_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SCATTER_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

enum ScatterType {
  SCATTER_MUL = 0,
  SCATTER_DIV,
  SCATTER_INVALID_TYPE = 255
};

template <typename T, typename S>
CUDA_LIB_EXPORT void Scatter(enum ScatterType func_type, S size_limit, const size_t &inner_size,
                                  const size_t &indices_size, const S *indices, const T *updates, T *input,
                                  const uint32_t &device_id, cudaStream_t cuda_stream);

#endif
