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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PAD_V3_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PAD_V3_IMPL_CUH_
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T>
CUDA_LIB_EXPORT void CalCircularPad3d(const size_t size, const T *input, const int64_t old_depth,
                                      const int64_t old_height, const int64_t old_width, const int64_t padded_depth,
                                      const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
                                      const int64_t pad_top, const int64_t pad_left, const int64_t pad_back,
                                      const int64_t pad_down, const int64_t pad_right, T *output,
                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalConstantPad3d(const size_t size, const T *input, const int64_t num, const int64_t channels,
                                      const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                                      const int64_t padded_depth, const int64_t padded_height,
                                      const int64_t padded_width, const int64_t pad_head, const int64_t pad_top,
                                      const int64_t pad_left, const T *pad_value, T *output, const uint32_t &device_id,
                                      cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalReflectPad3d(const size_t size, const T *input, const int64_t num, const int64_t channels,
                                     const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                                     const int64_t padded_depth, const int64_t padded_height,
                                     const int64_t padded_width, const int64_t pad_head, const int64_t pad_top,
                                     const int64_t pad_left, T *output, const uint32_t &device_id,
                                     cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalEdgePad3d(const size_t size, const T *input, const int64_t num, const int64_t channels,
                                  const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                                  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width,
                                  const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, T *output,
                                  const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalConstantPadGrad3d(const size_t size, const T *dy, const int64_t num, const int64_t channels,
                                          const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                                          const int64_t padded_depth, const int64_t padded_height,
                                          const int64_t padded_width, const int64_t pad_head, const int64_t pad_top,
                                          const int64_t pad_left, T *dx, const uint32_t &device_id,
                                          cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalReflectPadGrad3d(const size_t size, T *input, const int64_t num, const int64_t channels,
                                         const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                                         const int64_t padded_depth, const int64_t padded_height,
                                         const int64_t padded_width, const int64_t pad_head, const int64_t pad_top,
                                         const int64_t pad_left, T *output, const uint32_t &device_id,
                                         cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalEdgePadGrad3d(const size_t size, T *input, const int64_t num, const int64_t channels,
                                      const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                                      const int64_t padded_depth, const int64_t padded_height,
                                      const int64_t padded_width, const int64_t pad_head, const int64_t pad_top,
                                      const int64_t pad_left, T *output, const uint32_t &device_id,
                                      cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalCircularPadGrad3d(const size_t size, const T *input, const int64_t old_depth,
                                          const int64_t old_height, const int64_t old_width, const int64_t padded_depth,
                                          const int64_t padded_height, const int64_t padded_width,
                                          const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
                                          const int64_t pad_back, const int64_t pad_down, const int64_t pad_right,
                                          T *output, const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PAD_V3_IMPL_CUH_
