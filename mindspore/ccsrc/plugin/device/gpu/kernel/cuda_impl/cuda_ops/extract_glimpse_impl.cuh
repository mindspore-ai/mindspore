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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_EXTRACT_GLIMPSE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_EXTRACT_GLIMPSE_IMPL_CUH_
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

enum ExtractGlimpsenoiseMode { ZERO = 0, GAUSSIAN, UNIFORM };

static std::map<std::string, ExtractGlimpsenoiseMode> kExtractGlimpsenoiseMap{
  {"zero", ExtractGlimpsenoiseMode::ZERO},
  {"gaussian", ExtractGlimpsenoiseMode::GAUSSIAN},
  {"uniform", ExtractGlimpsenoiseMode::UNIFORM}};

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalExtractGlimpse(const size_t output_size, const size_t batch_cnt_, const size_t channels_,
                                              const size_t image_height_, const size_t image_width_,
                                              const ExtractGlimpsenoiseMode noise, const bool centered,
                                              const bool normalized, const bool uniform_noise, const T *inputs,
                                              const int *size, const T *offsets, T *output, cudaStream_t cuda_stream_);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_EXTRACT_GLIMPSE_IMPL_CUH_
