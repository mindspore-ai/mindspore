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
#include <thrust/pair.h>
#include "device/gpu/cuda_common.h"
#include "fake_quant_impl.cuh"

__global__ void FakeQuantize(const float* input, float* output, const int size, const float* nudge_min,
                             const float* nudge_max, const float* scale, bool symmetric) {
  float input_x = 0.f;
  int nudge_input = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    input_x = input[i];
    // clamp input x
    if (input_x < nudge_min[0]) {
      input_x = nudge_min[0];
    }
    if (input_x > nudge_max[0]) {
      input_x = nudge_max[0];
    }
    // clamp shift
    nudge_input = floor((input_x - nudge_min[0]) / scale[0] + 0.5f);

    // quantize
    output[i] = nudge_input * scale[0] + nudge_min[0];
  }
  return;
}

__global__ void FakeQuantizeGrad(const float* input, const float* gradient, float* output, const int size,
                                 const float* nudge_min, const float* nudge_max) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    if (input[i] < nudge_min[0] || input[i] > nudge_max[0]) {
      output[i] = 0;
    } else {
      output[i] = gradient[i];
    }
  }
  return;
}

__global__ void NudgeMinMax(const float* input_min, const float* input_max, const float quant_min,
                            const float quant_max, float* nudge_min, float* nudge_max, float* scale) {
  float zp_from_min = 0.f;
  if ((quant_max - quant_min) == 0 || (*input_max - *input_min) == 0) {
    *scale = 0.f;
    zp_from_min = 0.f;
  } else {
    *scale = (*input_max - *input_min) / (quant_max - quant_min);
    zp_from_min = quant_min - *input_min / *scale;
  }

  float nudge_zp = 0.f;
  if (zp_from_min <= quant_min) {
    nudge_zp = quant_min;
  } else if (zp_from_min >= quant_max) {
    nudge_zp = quant_max;
  } else {
    nudge_zp = round(zp_from_min);
  }

  *nudge_min = (quant_min - nudge_zp) * (*scale);
  *nudge_max = (quant_max - nudge_zp) * (*scale);
  return;
}

__global__ void UpdateInputMinMaxWithEMA(float* input_min, float* input_max, const float min, const float max,
                                         const float decay) {
  *input_min = decay * (min) + (1 - decay) * (*input_min);
  *input_min = *input_min > 0 ? 0 : *input_min;
  *input_max = decay * (max) + (1 - decay) * (*input_max);
  *input_max = *input_max < 0 ? 0 : *input_max;
  return;
}

__global__ void UpdateInputMinMax(float* input_min, float* input_max, const float min, const float max) {
  *input_min = min;
  *input_max = max;
}

void CalFakeQuantize(const float* input, float* output, const int size, const float* nudge_min, const float* nudge_max,
                     const float* scale, bool symmetric, cudaStream_t cuda_stream) {
  FakeQuantize<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input, output, size, nudge_min, nudge_max, scale,
                                                                  symmetric);
  return;
}

void CalFakeQuantizeGrad(const float* input, const float* gradient, float* output, const int size,
                         const float* nudge_min, const float* nudge_max, cudaStream_t cuda_stream) {
  FakeQuantizeGrad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input, gradient, output, size, nudge_min,
                                                                      nudge_max);
  return;
}

void CalNudge(const float* input_min, const float* input_max, const float quant_min, const float quant_max,
              float* nudge_min, float* nudge_max, float* scale, cudaStream_t cuda_stream) {
  NudgeMinMax<<<1, 1>>>(input_min, input_max, quant_min, quant_max, nudge_min, nudge_max, scale);
  return;
}

void CalMinMax(float* input, float* input_min, float* input_max, const int size, const float ema_decay, const bool ema,
               cudaStream_t cuda_stream) {
  float minel = 0.f;
  float maxel = 0.f;
  thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>> tuple;
  tuple = thrust::minmax_element(thrust::device_pointer_cast(input), thrust::device_pointer_cast(input) + size);
  minel = tuple.first[0];
  maxel = tuple.second[0];

  if (ema) {
    UpdateInputMinMaxWithEMA<<<1, 1>>>(input_min, input_max, minel, maxel, ema_decay);
  } else {
    UpdateInputMinMax<<<1, 1>>>(input_min, input_max, minel, maxel);
  }
  return;
}

