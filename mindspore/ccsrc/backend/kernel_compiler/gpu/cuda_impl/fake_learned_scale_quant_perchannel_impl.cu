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

#include "fake_learned_scale_quant_perchannel_impl.cuh"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/pair.h>
#include <algorithm>
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void FakeLearnedScaleQuantPerChannel(float *output, const int size, float *input_alpha,
                                                float *input_quant, const int channel_num) {
  int channel_idx = 0;
  int per_channel_num = size / channel_num;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    channel_idx = floor(static_cast<double>(i) / static_cast<double>(per_channel_num));
    // dequantize
    output[i] = input_quant[i] * input_alpha[channel_idx];
  }
  return;
}

__global__ void FakeLearnedScaleQuantPerChannelGrad(float *grad_input, float *grad_alpha, const float *gradient,
                                                    const int size, const float *input_div_alpha,
                                                    const float *input_quant, const bool neg_trunc,
                                                    const int channel_num) {
  int channel_idx = 0;
  int per_channel_num = size / channel_num;
  float lower_bound = -1.0 * !neg_trunc;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    float grad_alpha_temp = 0.f;
    channel_idx = floor(static_cast<double>(i) / static_cast<double>(per_channel_num));
    if (input_div_alpha[i] > 1.0) {
      grad_alpha_temp = gradient[i];
      grad_input[i] = 0;
    } else if (input_div_alpha[i] < lower_bound) {
      grad_alpha_temp = -gradient[i];
      grad_input[i] = 0;
    } else {
      grad_input[i] = gradient[i];
      grad_alpha_temp = (gradient[i] * (input_quant[i] -  input_div_alpha[i]));
    }
    MsAtomicAdd(grad_alpha + channel_idx, grad_alpha_temp);
  }
  return;
}

__global__ void LSQNudgePerChannel(const float *input, const int size, float *input_alpha, float *input_quant_max,
                                   float *input_div_alpha, float *input_quant, const bool neg_trunc,
                                   const int channel_num) {
  float input_x;
  int channel_idx = 0;
  int per_channel_num = size / channel_num;
  float lower_bound = -1.0 * !neg_trunc;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    channel_idx = floor(static_cast<double>(i) / static_cast<double>(per_channel_num));
    input_x = input[i] / input_alpha[channel_idx];
    input_div_alpha[i] = input_x;
    input_x = max(input_x, lower_bound);
    input_x = min(input_x, 1.0);

    // quantize
    input_quant[i] = floor(input_x * input_quant_max[0] + 0.5f) / input_quant_max[0];
  }
  return;
}

void CalFakeLearnedScaleQuantPerChannel(float *output, const int size, float *input_alpha, float *input_quant,
                                        const int channel_num, cudaStream_t cuda_stream) {
  FakeLearnedScaleQuantPerChannel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(output, size, input_alpha,
                                                                                     input_quant, channel_num);
  return;
}

void CalFakeLearnedScaleQuantPerChannelGrad(float *grad_input, float *grad_alpha, const float *gradient, const int size,
                                            const float *input_div_alpha, const float *input_quant,
                                            const bool neg_trunc, const int channel_num, cudaStream_t cuda_stream) {
  FakeLearnedScaleQuantPerChannelGrad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(grad_input,
                                                                                         grad_alpha,
                                                                                         gradient,
                                                                                         size,
                                                                                         input_div_alpha,
                                                                                         input_quant,
                                                                                         neg_trunc,
                                                                                         channel_num);
  return;
}

void CalLSQNudgePerChannel(const float *input, const int size, float *input_alpha, float *input_quant_max,
                           float *input_div_alpha, float *input_quant, const bool neg_trunc, const int channel_num,
                           cudaStream_t cuda_stream) {
  LSQNudgePerChannel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input, size, input_alpha, input_quant_max,
                                                                        input_div_alpha, input_quant, neg_trunc,
                                                                        channel_num);
  return;
}
