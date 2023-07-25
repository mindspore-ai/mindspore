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
#include "dynamic_stitch_impl.cuh"

__global__ void StitchKernel(const int *index_addr, const unsigned char *data_addr, unsigned char *output_addr,
                             const size_t index_num, const size_t data_size, int *max_index_dev) {
  for (size_t i = 0; i < index_num; i++) {
    int index = index_addr[i];
    if (max_index_dev[0] < index) {
      max_index_dev[0] = index;
    }
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < data_size; j += gridDim.x * blockDim.x) {
      output_addr[index * data_size + j] = data_addr[i * data_size + j];
    }
  }
}

cudaError_t CallStitch(const int *index_addr, const unsigned char *data_addr, unsigned char *output_addr,
                       const size_t index_num, const size_t data_size, int *max_index_dev, cudaStream_t cuda_stream) {
  StitchKernel<<<GET_BLOCKS(data_size), GET_THREADS, 0, cuda_stream>>>(index_addr, data_addr, output_addr, index_num,
                                                                       data_size, max_index_dev);
  return GetCudaStatus();
}
