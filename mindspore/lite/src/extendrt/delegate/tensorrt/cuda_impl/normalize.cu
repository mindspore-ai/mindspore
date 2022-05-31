/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/cuda_impl/normalize.cuh"
#include <stdio.h>
#include <math.h>
#include "src/extendrt/delegate/tensorrt/cuda_impl/cuda_helper.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/utils.cuh"

template <typename T>
__global__ void NormalizeKernel(const T *input, const T *gamma, const T *beta, T *output, size_t n, float epsilion,
                                int dim_before_axis) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int block_loop = (dim_before_axis - 1) / gridDim.x + 1;
  const int element_cnt = dim_before_axis * n;

  __shared__ float s_mean[2048];
  __shared__ float s_variance[2048];
  float sum = 0.0f;
  float variance = 0.0f;

  for (int block = 0; block < block_loop; block++) {
    float local_sum = 0.0f;
    int mean_index = bid + block * gridDim.x;
    int num_index = bid * n + block * gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x) {
      if (num_index + i >= element_cnt) {
        break;
      }
      local_sum += static_cast<float>(input[num_index + i]);
    }
    sum = blockReduceSum(local_sum);
    if (tid == 0) {
      s_mean[mean_index] = sum / n;
    }
  }
  __syncthreads();

  for (int block = 0; block < block_loop; block++) {
    float local_var_sum = 0.0f;
    int var_index = bid + block * gridDim.x;
    int num_index = bid * n + block * gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x) {
      if (num_index + i >= element_cnt) {
        break;
      }
      float diff = static_cast<float>(input[num_index + i]) - s_mean[var_index];
      local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);
    if (tid == 0) {
      s_variance[var_index] = rsqrtf(variance / n + epsilion);
    }
  }
  __syncthreads();
  for (int block = 0; block < block_loop; block++) {
    int var_index = bid + block * gridDim.x;
    int num_index = bid * n + block * gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x) {
      if (num_index + i >= element_cnt) {
        break;
      }
      float beta_val = (beta == nullptr) ? 0.0f : static_cast<float>(beta[i]);
      output[num_index + i] =
        static_cast<T>(((static_cast<float>(input[num_index + i]) - s_mean[var_index]) * s_variance[var_index]) *
                         static_cast<float>(gamma[i]) +
                       beta_val);
    }
  }
}

template <typename T>
void Normalize(const T *input, const T *gamma, const T *beta, T *output, size_t dim_at_axis, float epsilion,
               int element_cnt, cudaStream_t stream) {
  int thread_num = GET_THREADS_CAL(dim_at_axis);
  int block_num = GET_BLOCKS_CAL(element_cnt, thread_num);
  int dim_before_axis = element_cnt / dim_at_axis;
  NormalizeKernel<<<block_num, thread_num, 0, stream>>>(input, gamma, beta, output, dim_at_axis, epsilion,
                                                        dim_before_axis);
  return;
}

template void Normalize(const float *input, const float *gamma, const float *beta, float *output, size_t dim_at_axis,
                        float epsilion, int element_cnt, cudaStream_t stream);
