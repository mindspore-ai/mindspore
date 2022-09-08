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

#include "plugin/device/gpu/kernel/cuda_impl/rl/discounted_return_impl.cuh"
#include <cuda_fp16.h>
#include <algorithm>

template <typename T>
__global__ void DiscountedReturnKernel(const int timestep, const int num_env, const int num_element, const float gamma,
                                       const T *reward, const bool *done, const T *last_value, T *discouted_return) {
  int elements_per_timestep = num_env * num_element;
  int idx_in_timestep = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_in_timestep >= num_env * num_element) {
    return;
  }

  T last_value_local = last_value[idx_in_timestep];
  int env_index = idx_in_timestep / num_element;
  for (int t = timestep - 1; t >= 0; t--) {
    int curr_timestep_index_offset = t * elements_per_timestep + idx_in_timestep;

    last_value_local = reward[curr_timestep_index_offset] +
                       static_cast<T>((1 - done[t * num_env + env_index]) * gamma) * last_value_local;
    discouted_return[curr_timestep_index_offset] = last_value_local;
  }
}

template <typename T>
void DiscountedReturn(const int &timestep, const int &num_env, const int &num_element,
                      const float &gamma, const T *reward, const bool *done, const T *last_value,
                      T *discouted_return, cudaStream_t stream) {
  // Every block process M element, 256 is a common tile size.
  const int element_per_step = num_env * num_element;
  const int element_per_block = std::min(256, element_per_step);
  const int grid_dim = (element_per_step + element_per_block - 1) / element_per_block;

  DiscountedReturnKernel<<<grid_dim, element_per_block, 0, stream>>>(timestep, num_env, num_element, gamma, reward,
                                                                     done, last_value, discouted_return);
}

template CUDA_LIB_EXPORT void DiscountedReturn(const int &timestep, const int &num_env, const int &num_element,
                                               const float &gamma, const float *reward, const bool *done,
                                               const float *last_value, float *discouted_return, cudaStream_t stream);
template CUDA_LIB_EXPORT void DiscountedReturn(const int &timestep, const int &num_env, const int &num_element,
                                               const float &gamma, const half *reward, const bool *done,
                                               const half *last_value, half *discouted_return, cudaStream_t stream);
