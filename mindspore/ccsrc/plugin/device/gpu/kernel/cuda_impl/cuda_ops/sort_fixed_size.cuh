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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SORT_FIXED_SIZE_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SORT_FIXED_SIZE_CUH_

#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tensor_layout_helper.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

constexpr int kFixedSizeLevel1 = 4096;
constexpr int kFixedSizeLevel2 = 2048;
constexpr int kFixedSizeLevel3 = 1024;
constexpr int kFixedSizeLevel4 = 128;
constexpr int kFixedSizeLevel5 = 32;
constexpr int kFixedSizeLevel1ItemPreThread = 32;
constexpr int kFixedSizeLevel2ItemPreThread = 32;
constexpr int kFixedSizeLevel3ItemPreThread = 32;
constexpr int kFixedSizeLevel4ItemPreThread = 4;
constexpr int kFixedSizeLevel5ItemPreThread = 2;
constexpr int kFixedSizeSortKeyDimsLast = -1;
constexpr int kFixedSizeSortKeyDimsSecond = 2;
constexpr int kFixedSizeSortKeyDimsLastSecond = -2;

template <int sort_size, int items_per_thread, typename K, typename V>
CUDA_LIB_EXPORT cudaError_t SortFixedSize(const int key_dims, const TensorLayoutHelper &key_info, K *key_data,
                                          int64_t key_slices, int64_t key_slice_size, int64_t key_slice_stride,
                                          const TensorLayoutHelper &value_info, V *value_data,
                                          int64_t value_slice_stride, bool descending, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SORT_FIXED_SIZE_CUH_
