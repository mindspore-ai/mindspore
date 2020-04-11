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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_FAKEQUANTIZE_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_FAKEQUANTIZE_H_

void CalNudgePerChannel(const float* input_min, const float* input_max, const float quant_min, const float quant_max,
                        float* nudge_min, float* nudge_max, float* scale, const int channel_num,
                        cudaStream_t cuda_stream);

void CalFakeQuantizePerChannel(const float* input, float* output, const int total_num, const int channel_num,
                               const float* nudge_min, const float* nudge_max, const float* scale, bool symmetric,
                               cudaStream_t cuda_stream);

void CalMinMaxPerChannel(float* input, float* input_min, float* input_max, const int total_num, const int channel_num,
                         const float ema_decay, const bool ema, cudaStream_t cuda_stream);

void CalFakeQuantizePerChannelGrad(const float* input, const float* gradient, float* output, const int total_num,
                                   const int channel_num, const float* nudge_min, const float* nudge_max,
                                   cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_FAKEQUANTIZE_H_
