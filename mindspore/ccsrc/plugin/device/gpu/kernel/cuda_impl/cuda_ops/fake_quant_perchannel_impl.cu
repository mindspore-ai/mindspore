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

#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/pair.h>
#include "fake_quant_perchannel_impl.cuh"

/**
 * Find the nudge min, max and scale value as output.
 * @param input_min array
 * @param input_max array
 * @param quant_min 1 << bit -1
 * @param quant_max 0
 * @param nudge_min array
 * @param nudge_max array
 * @param scale array
 * @param channel_num
 * @return
 */
__global__ void NudgeMinMaxPerChannel(float *input_min, float *input_max, const float quant_min, const float quant_max,
                                      float *nudge_min, float *nudge_max, float *scale, int channel_num,
                                      const bool symmetric) {
  float zp_from_min = 0.f;
  float nudge_zp = 0.f;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < channel_num; i += blockDim.x * gridDim.x) {
    float max_data = input_max[i];
    float min_data = input_min[i];
    if (symmetric) {
      max_data = abs(input_min[i]) < input_max[i] ? input_max[i] : -input_min[i];
      min_data = abs(input_min[i]) < input_max[i] ? -input_max[i] : input_min[i];
    }
    if ((quant_max - quant_min) == 0 || (max_data - min_data) == 0) {
      scale[i] = 0.f;
      zp_from_min = 0.f;
    } else {
      scale[i] = (max_data - min_data) / (quant_max - quant_min);
      zp_from_min = quant_min - min_data / scale[i];
    }

    if (zp_from_min <= quant_min) {
      nudge_zp = quant_min;
    } else if (zp_from_min >= quant_max) {
      nudge_zp = quant_max;
    } else {
      nudge_zp = round(zp_from_min);
    }

    nudge_min[i] = (quant_min - nudge_zp) * (scale[i]);
    nudge_max[i] = (quant_max - nudge_zp) * (scale[i]);
  }
}

cudaError_t CalNudgePerChannel(float *input_min, float *input_max, const float quant_min, const float quant_max,
                               float *nudge_min, float *nudge_max, float *scale, const int channel_num,
                               const bool symmetric, cudaStream_t cuda_stream) {
  NudgeMinMaxPerChannel<<<GET_BLOCKS(channel_num), GET_THREADS, 0, cuda_stream>>>(
    input_min, input_max, quant_min, quant_max, nudge_min, nudge_max, scale, channel_num, symmetric);
  return GetCudaStatus();
}

/**
 * Calculate fake quant output according by nudge min, nudge max, nudge scale.
 * @param input - array
 * @param output - array
 * @param total_size - int, purpose for cal the per channel number in filters
 * @param channel_size - int, purpose for cal the per channel number in filters
 * @param nudge_min - array
 * @param nudge_max - array
 * @param scale - array
 * @return
 */
__global__ void FakeQuantPerChannel(const float *input, float *output, const int total_size, const int channel_size,
                                    const float *nudge_min, const float *nudge_max, const float *scale) {
  float input_x = 0.f;
  int nudge_input = 0;
  int channel_idx = 0;
  int per_channel_num = total_size / channel_size;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
    input_x = input[i];
    channel_idx = floor(static_cast<double>(i) / static_cast<double>(per_channel_num));
    // clamp input x
    if (input_x < nudge_min[channel_idx]) {
      input_x = nudge_min[channel_idx];
    }
    if (input_x > nudge_max[channel_idx]) {
      input_x = nudge_max[channel_idx];
    }
    // clamp shift
    nudge_input = floor((input_x - nudge_min[channel_idx]) / scale[channel_idx] + 0.5f);

    // quantize
    output[i] = nudge_input * scale[channel_idx] + nudge_min[channel_idx];
  }
}

cudaError_t CalFakeQuantPerChannel(const float *input, float *output, const int total_size, const int channel_size,
                                   const float *nudge_min, const float *nudge_max, const float *scale,
                                   cudaStream_t cuda_stream) {
  FakeQuantPerChannel<<<GET_BLOCKS(total_size), GET_THREADS, 0, cuda_stream>>>(input, output, total_size, channel_size,
                                                                               nudge_min, nudge_max, scale);
  return GetCudaStatus();
}

__global__ void FakeQuantPerChannelGrad(const float *input, const float *gradient, float *output, const int total_size,
                                        const int channel_size, const float *nudge_min, const float *nudge_max) {
  int channel_idx = 0;
  int per_channel_num = total_size / channel_size;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
    channel_idx = floor(static_cast<double>(i) / static_cast<double>(per_channel_num));
    if (input[i] < nudge_min[channel_idx] || input[i] > nudge_max[channel_idx]) {
      output[i] = 0;
    } else {
      output[i] = gradient[i];
    }
  }
}

cudaError_t CalFakeQuantPerChannelGrad(const float *input, const float *gradient, float *output, const int total_num,
                                       const int channel_num, const float *nudge_min, const float *nudge_max,
                                       cudaStream_t cuda_stream) {
  FakeQuantPerChannelGrad<<<GET_BLOCKS(channel_num), GET_THREADS, 0, cuda_stream>>>(input, gradient, output, total_num,
                                                                                    channel_num, nudge_min, nudge_max);
  return GetCudaStatus();
}
