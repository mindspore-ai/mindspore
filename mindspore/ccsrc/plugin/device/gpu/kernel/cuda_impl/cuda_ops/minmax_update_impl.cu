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
#include "minmax_update_impl.cuh"

__global__ void UpdateInputMinMaxPerLayerWithEMA(const float *input_min, const float *input_max, float *output_min,
                                                 float *output_max, const float min, const float max,
                                                 const float decay) {
  output_min[0] = decay * (min) + (1 - decay) * (input_min[0]);
  output_min[0] = output_min[0] > 0 ? 0 : output_min[0];
  output_max[0] = decay * (max) + (1 - decay) * (input_max[0]);
  output_max[0] = output_max[0] < 0 ? 0 : output_max[0];
  return;
}

__global__ void UpdateInputMinMaxPerLayer(float *output_min, float *output_max, const float min, const float max) {
  output_min[0] = min > 0 ? 0 : min;
  output_max[0] = max < 0 ? 0 : max;
  return;
}

__global__ void UpdateInputMinMaxPerChannel(float *input, float *input_min, float *input_max, float *output_min,
                                            float *output_max, int channels, int per_channel_nums, bool ema,
                                            float ema_decay) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < channels; i += blockDim.x * gridDim.x) {
    thrust::pair<float *, float *> sum =
      thrust::minmax_element(thrust::device, input + i * per_channel_nums, input + per_channel_nums * (i + 1));
    if (ema) {
      output_min[i] = ema_decay * sum.first[0] + (1 - ema_decay) * input_min[i];
      output_max[i] = ema_decay * sum.second[0] + (1 - ema_decay) * input_max[i];
    } else {
      output_min[i] = sum.first[0];
      output_max[i] = sum.second[0];
    }
    output_min[i] = output_min[i] > 0 ? 0 : output_min[i];
    output_max[i] = output_max[i] < 0 ? 0 : output_max[i];
  }
  return;
}

cudaError_t CalMinMaxPerChannel(float *input, float *input_min, float *input_max, float *output_min, float *output_max,
                                const int total_num, const int channel_num, const float ema_decay, const bool ema,
                                cudaStream_t cuda_stream) {
  int per_channel_num = total_num / channel_num;
  UpdateInputMinMaxPerChannel<<<GET_BLOCKS(channel_num), GET_THREADS, 0, cuda_stream>>>(
    input, input_min, input_max, output_min, output_max, channel_num, per_channel_num, ema, ema_decay);
  return GetCudaStatus();
}

cudaError_t CalMinMaxPerLayer(float *input, float *input_min, float *input_max, float *output_min, float *output_max,
                              const int total_num, const float ema_decay, const bool ema, cudaStream_t cuda_stream) {
  float minel = 0.f;
  float maxel = 0.f;
  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>> tuple;
  tuple =
    thrust::minmax_element(policy, thrust::device_pointer_cast(input), thrust::device_pointer_cast(input) + total_num);
  minel = tuple.first[0];
  maxel = tuple.second[0];

  if (ema) {
    UpdateInputMinMaxPerLayerWithEMA<<<1, 1, 0, cuda_stream>>>(input_min, input_max, output_min, output_max, minel,
                                                               maxel, ema_decay);
  } else {
    UpdateInputMinMaxPerLayer<<<1, 1, 0, cuda_stream>>>(output_min, output_max, minel, maxel);
  }
  return GetCudaStatus();
}
