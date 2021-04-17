/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_FAKE_LEARNED_SCALE_QUANT_PERCHANNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_FAKE_LEARNED_SCALE_QUANT_PERCHANNEL_H_

#include "runtime/device/gpu/cuda_common.h"

void CalLSQNudgePerChannel(const float *input, const int size, float *input_alpha, float *input_quant_max,
                           float *input_div_alpha, float *input_quant, const bool neg_trunc, const int channel_num,
                           cudaStream_t cuda_stream);

void CalFakeLearnedScaleQuantPerChannel(float *output, const int size, float *input_alpha, float *input_quant,
                                        const int channel_num, cudaStream_t cuda_stream);

void CalFakeLearnedScaleQuantPerChannelGrad(float *grad_input, float *grad_alpha, const float *gradient, const int size,
                                            const float *input_div_alpha, const float *input_quant,
                                            const bool neg_trunc, const int channel_num, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_FAKE_LEARNED_SCALE_QUANT_PERCHANNEL_H_
