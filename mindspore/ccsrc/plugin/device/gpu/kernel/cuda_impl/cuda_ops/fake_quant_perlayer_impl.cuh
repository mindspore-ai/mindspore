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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FAKE_QUANT_PERLAYER_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FAKE_QUANT_PERLAYER_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

CUDA_LIB_EXPORT void CalNudgePerLayer(float *input_min, float *input_max, const float quant_min, const float quant_max,
                                      float *nudge_min, float *nudge_max, float *scale, const bool symmetric,
                                      cudaStream_t cuda_stream);

CUDA_LIB_EXPORT void CalFakeQuantPerLayer(const float *input, float *output, const int size, const float *nudge_min,
                                          const float *nudge_max, const float *scale, cudaStream_t cuda_stream);

CUDA_LIB_EXPORT void CalFakeQuantPerLayerGrad(const float *input, const float *gradient, float *output, const int size,
                                              const float *nudge_min, const float *nudge_max,
                                              cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FAKE_QUANT_PERLAYER_IMPL_CUH_
