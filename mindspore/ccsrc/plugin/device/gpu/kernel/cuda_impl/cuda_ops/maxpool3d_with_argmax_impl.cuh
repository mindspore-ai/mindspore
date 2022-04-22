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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MAXPOOL3D_WITH_ARGMAX_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MAXPOOL3D_WITH_ARGMAX_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T, typename S>
CUDA_LIB_EXPORT void CalMaxPool3DWithArgmax(const T *input, const int64_t n, const int64_t c, const int64_t d,
                                            const int64_t h, const int64_t w, const int64_t ksize_d,
                                            const int64_t ksize_h, const int64_t ksize_w, const int64_t stride_d,
                                            const int64_t stride_h, const int64_t stride_w, const int64_t pad_front,
                                            const int64_t pad_top, const int64_t pad_left, const int64_t dilation_d,
                                            const int64_t dilation_h, const int64_t dilation_w, const int64_t out_d,
                                            const int64_t out_h, const int64_t out_w, T *output, S *index,
                                            const uint32_t device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MAXPOOL3D_WITH_ARGMAX_IMPL_CUH_
