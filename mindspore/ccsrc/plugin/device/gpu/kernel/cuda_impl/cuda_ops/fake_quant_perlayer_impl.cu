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
#include "fake_quant_perlayer_impl.cuh"

__global__ void FakeQuantPerLayer(const float *input, float *output, const int size, const float *nudge_min,
                                  const float *nudge_max, const float *scale) {
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
    nudge_input = round((input_x - nudge_min[0]) / scale[0]);

    // quantize
    output[i] = nudge_input * scale[0] + nudge_min[0];
  }
  return;
}

__global__ void FakeQuantPerLayerGrad(const float *input, const float *gradient, float *output, const int size,
                                      const float *nudge_min, const float *nudge_max) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    if (input[i] < nudge_min[0] || input[i] > nudge_max[0]) {
      output[i] = 0;
    } else {
      output[i] = gradient[i];
    }
  }
  return;
}

__global__ void NudgeMinMaxPerLayer(float *input_min, float *input_max, const float quant_min, const float quant_max,
                                    float *nudge_min, float *nudge_max, float *scale, const bool symmetric) {
  float zp_from_min = 0.f;
  scale[0] = 0.f;
  nudge_max[0] = 0.f;
  nudge_min[0] = 0.f;

  float max_data = input_max[0];
  float min_data = input_min[0];
  if (symmetric) {
    max_data = abs(input_min[0]) < input_max[0] ? input_max[0] : -input_min[0];
    min_data = abs(input_min[0]) < input_max[0] ? -input_max[0] : input_min[0];
  }

  if ((quant_max - quant_min) == 0 || (max_data - min_data) == 0) {
    scale[0] = 0.f;
    zp_from_min = 0.f;
  } else {
    scale[0] = (max_data - min_data) / (quant_max - quant_min);
    zp_from_min = quant_min - min_data / scale[0];
  }

  float nudge_zp = 0.f;
  if (zp_from_min <= quant_min) {
    nudge_zp = quant_min;
  } else if (zp_from_min >= quant_max) {
    nudge_zp = quant_max;
  } else {
    nudge_zp = round(zp_from_min);
  }

  nudge_min[0] = (quant_min - nudge_zp) * (scale[0]);
  nudge_max[0] = (quant_max - nudge_zp) * (scale[0]);
  return;
}

cudaError_t CalFakeQuantPerLayer(const float *input, float *output, const int size, const float *nudge_min,
                                 const float *nudge_max, const float *scale, cudaStream_t cuda_stream) {
  FakeQuantPerLayer<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input, output, size, nudge_min, nudge_max,
                                                                       scale);
  return GetCudaStatus();
}

cudaError_t CalFakeQuantPerLayerGrad(const float *input, const float *gradient, float *output, const int size,
                                     const float *nudge_min, const float *nudge_max, cudaStream_t cuda_stream) {
  FakeQuantPerLayerGrad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input, gradient, output, size, nudge_min,
                                                                           nudge_max);
  return GetCudaStatus();
}

cudaError_t CalNudgePerLayer(float *input_min, float *input_max, const float quant_min, const float quant_max,
                             float *nudge_min, float *nudge_max, float *scale, const bool symmetric,
                             cudaStream_t cuda_stream) {
  NudgeMinMaxPerLayer<<<1, 1, 0, cuda_stream>>>(input_min, input_max, quant_min, quant_max, nudge_min, nudge_max, scale,
                                                symmetric);
  return GetCudaStatus();
}
