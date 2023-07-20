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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CONVOLUTION_COMMON_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CONVOLUTION_COMMON_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

enum class ConvolutionOpType {
  kConv2dDepthWiseForwardNCHW = 3,
  kConv2dDepthWiseForwardNHWC = 4,
  kConv2dDepthWiseInputGradNCHW = 5,
  kConv2dDepthWiseInputGradNHWC = 6,
  kConv2dDepthWiseFilterGradNCHW = 7,
  kConv2dDepthWiseFilterGradNHWC = 8,
  kInvalid = INT_MAX
};

struct ConvolutionCudaArgs {
  size_t output_size{0};
  int batch_size{0};
  int in_height{0};
  int in_width{0};
  int in_channel{0};
  int out_channel{0};
  int filter_height{0};
  int filter_width{0};
  int pad_height{0};
  int pad_width{0};
  int pad_top{0};
  int pad_left{0};
  int out_height{0};
  int out_width{0};
  int group{0};
  int stride_height{0};
  int stride_width{0};
  int dilation_height{0};
  int dilation_width{0};
};

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CONVOLUTION_COMMON_CUH_
