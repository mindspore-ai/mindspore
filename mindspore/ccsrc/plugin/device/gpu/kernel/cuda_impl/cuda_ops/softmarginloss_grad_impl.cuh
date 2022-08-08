/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SOFTMARGINLOSS_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SOFTMARGINLOSS_GRAD_IMPL_CUH_
#include <map>
#include <string>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

enum class ReductionMode { kNone, kMean, kSum };

static std::map<std::string, ReductionMode> kReductionModeMap{
  {"none", ReductionMode::kNone}, {"mean", ReductionMode::kMean}, {"sum", ReductionMode::kSum}};

template <typename T>
CUDA_LIB_EXPORT void SoftMarginLossGrad(const T *prediction, const T *target, const T *dout, const size_t input_size,
                                        const T norm, const ReductionMode &reduction, T *gradient,
                                        const uint32_t &device_id, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SOFTMARGINLOSS_GRAD_IMPL_CUH_
