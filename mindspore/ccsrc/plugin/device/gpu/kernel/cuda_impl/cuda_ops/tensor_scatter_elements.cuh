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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TENSOR_SCATTER_ELEMETNS_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TENSOR_SCATTER_ELEMETNS_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

enum TensorScatterElementsReductionType { REDUCTION_ASSIGNMENT = 0, REDUCTION_ADD, REDCUTION_INVALID_TYPE = 255 };

template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t TensorScatterElements(const enum TensorScatterElementsReductionType reduction_type,
                                                  const int input_dims, const int indices_size, const S *indices,
                                                  const T *updates, T *output, const int64_t axis,
                                                  const int64_t input_axis_size, const size_t *indices_stride,
                                                  const size_t *output_stride, const uint32_t &device_id,
                                                  cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TENSOR_SCATTER_ELEMETNS_CUH_
