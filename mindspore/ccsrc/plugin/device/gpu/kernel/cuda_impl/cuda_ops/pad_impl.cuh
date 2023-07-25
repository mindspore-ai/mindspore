/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PAD_IMPL_CUH_
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

struct PadInfo {
  int shape[8] = {0};
  int strides[8] = {0};
  int paddings[16] = {0};
};

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPad(const size_t size, const T *input, const int num, const int channels,
                                   const int old_height, const int old_width, const int padded_height,
                                   const int padded_width, const int pad_top, const int pad_left, float pad_value,
                                   T *output, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPadGrad(const size_t size, const T *dy, const int num, const int channels,
                                       const int old_height, const int old_width, const int padded_height,
                                       const int padded_width, const int pad_top, const int pad_left, T *dx,
                                       cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPadNHWC(const size_t size, const T *input, const int num, const int old_height,
                                       const int old_width, const int channels, const int padded_height,
                                       const int padded_width, const int pad_top, const int pad_left, float pad_value,
                                       T *output, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPadGradNHWC(const size_t size, const T *input, const int num, const int old_height,
                                           const int old_width, const int channels, const int padded_height,
                                           const int padded_width, const int pad_top, const int pad_left, T *output,
                                           cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPadGeneral(const T *input, T *output, const PadInfo &info, const int input_size,
                                          const size_t input_rank, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPad3d(const size_t size, const T *input, const int num, const int channels,
                                     const int old_depth, const int old_height, const int old_width,
                                     const int padded_depth, const int padded_height, const int padded_width,
                                     const int pad_head, const int pad_top, const int pad_left, const float pad_value,
                                     T *output, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPadGrad3d(const size_t size, const T *dy, const int num, const int channels,
                                         const int old_depth, const int old_height, const int old_width,
                                         const int padded_depth, const int padded_height, const int padded_width,
                                         const int pad_head, const int pad_top, const int pad_left, T *dx,
                                         cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPadNDHWC(const size_t size, const T *input, const int num, const int old_depth,
                                        const int old_height, const int old_width, const int channels,
                                        const int padded_depth, const int padded_height, const int padded_width,
                                        const int pad_head, const int pad_top, const int pad_left,
                                        const float pad_value, T *output, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPadGradNDHWC(const size_t size, const T *dy, const int num, const int old_depth,
                                            const int old_height, const int old_width, const int channels,
                                            const int padded_depth, const int padded_height, const int padded_width,
                                            const int pad_head, const int pad_top, const int pad_left, T *dx,
                                            cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PAD_IMPL_CUH_
