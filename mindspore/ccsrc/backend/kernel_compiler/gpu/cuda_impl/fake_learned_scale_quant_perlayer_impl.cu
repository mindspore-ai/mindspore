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

#include "fake_learned_scale_quant_perlayer_impl.cuh"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <algorithm>
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void FakeLearnedScaleQuantPerLayer(float *output, const int size, float *input_alpha,
                                              float *input_quant) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    // dequantize
    output[i] = input_quant[i] * input_alpha[0];
  }
  return;
}

__global__ void FakeLearnedScaleQuantPerLayerGrad(float *grad_input, float *grad_alpha, const float *gradient,
                                                  const int size, const float *input_div_alpha,
                                                  const float *input_quant, const bool neg_trunc) {
  float grad_alpha_temp = 0.f;
  float lower_bound = -1.0 * !neg_trunc;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    if (input_div_alpha[i] > 1.0) {
      grad_alpha_temp += gradient[i];
      grad_input[i] = 0;
    } else if (input_div_alpha[i] < lower_bound) {
      grad_alpha_temp -= gradient[i];
      grad_input[i] = 0;
    } else {
      grad_input[i] = gradient[i];
      grad_alpha_temp += (gradient[i] * (input_quant[i] -  input_div_alpha[i]));
    }
  }
  MsAtomicAdd(grad_alpha, grad_alpha_temp);
  return;
}

__global__ void LSQNudgePerLayer(const float *input, const int size, float *input_alpha, float *input_quant_max,
                                 float *input_div_alpha, float *input_quant, const bool neg_trunc) {
  float input_x;
  float lower_bound = -1.0 * !neg_trunc;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    input_x = input[i] / input_alpha[0];
    input_div_alpha[i] = input_x;
    input_x = max(input_x, lower_bound);
    input_x = min(input_x, 1.0);

    // quantize
    input_quant[i] = floor(input_x * input_quant_max[0] + 0.5f) / input_quant_max[0];
  }
  return;
}

void CalFakeLearnedScaleQuantPerLayer(float *output, const int size, float *input_alpha, float *input_quant,
                                      cudaStream_t cuda_stream) {
  FakeLearnedScaleQuantPerLayer<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(output, size, input_alpha,
                                                                                   input_quant);
  return;
}

void CalFakeLearnedScaleQuantPerLayerGrad(float *grad_input, float *grad_alpha, const float *gradient, const int size,
                                          const float *input_div_alpha, const float *input_quant, const bool neg_trunc,
                                          cudaStream_t cuda_stream) {
  FakeLearnedScaleQuantPerLayerGrad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(grad_input,
                                                                                       grad_alpha,
                                                                                       gradient,
                                                                                       size,
                                                                                       input_div_alpha,
                                                                                       input_quant,
                                                                                       neg_trunc);
  return;
}

void CalLSQNudgePerLayer(const float *input, const int size, float *input_alpha, float *input_quant_max,
                         float *input_div_alpha, float *input_quant, const bool neg_trunc, cudaStream_t cuda_stream) {
  LSQNudgePerLayer<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input, size, input_alpha, input_quant_max,
                                                                      input_div_alpha, input_quant, neg_trunc);
  return;
}
