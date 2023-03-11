/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MAXPOOL_WITH_ARGMAX_V2_IMPL_CUH
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MAXPOOL_WITH_ARGMAX_V2_IMPL_CUH

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T, typename S>
CUDA_LIB_EXPORT void CalMaxPoolWithArgmaxV2(const T *input, const int n, const int c, const int h, const int w,
                                            const int ksize_h, const int ksize_w, const int strides_h,
                                            const int strides_w, const int pads_h, const int pads_w,
                                            const int dilation_h, const int dilation_w, const int out_h,
                                            const int out_w, T *output, S *index, const uint32_t &device_id,
                                            cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MAXPOOL_WITH_ARGMAX_V2_IMPL_CUH
