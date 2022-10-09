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

#include "where_impl.cuh"

template <typename T>
__global__ void WhereKernel(const int *cond, const T *input_x, const T *input_y, T *output, int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = cond[pos] ? input_x[pos] : input_y[pos];
  }
}

template <typename T>
void Where(const int *cond, const T *input_x, const T *input_y, T *output, int element_cnt, const uint32_t &device_id,
           cudaStream_t stream) {
  WhereKernel<<<CUDA_BLOCKS(device_id, element_cnt), CUDA_THREADS(device_id), 0, stream>>>(cond, input_x, input_y,
                                                                                           output, element_cnt);
}

template void Where(const int *cond, const float *input_x, const float *input_y, float *output, int element_cnt,
                    const uint32_t &device_id, cudaStream_t stream);

template void Where(const int *cond, const int *input_x, const int *input_y, int *output, int element_cnt,
                    const uint32_t &device_id, cudaStream_t stream);
