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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMP_PRIORITY_REPLAY_BUFFER_IMPL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMP_PRIORITY_REPLAY_BUFFER_IMPL_H_

#include <curand_kernel.h>

struct SumTree {
  float sum;
  float min;
};

void SumTreeInit(SumTree *tree, float *max_priority, const size_t &capacity, cudaStream_t stream);
void InitRandState(const size_t &batch_size, const uint64_t &seed, curandState *state, cudaStream_t stream);
void SumTreePush(SumTree *tree, const float &alpha, const size_t &idx, const size_t &capacity, float *priority,
                 float *max_priority, cudaStream_t stream);
void SumTreeSample(SumTree *tree, curandState *state, const size_t &capacity, float *beta, const size_t &batch_size,
                   size_t *indices, float *weights, cudaStream_t stream);
void SumTreeUpdate(SumTree *tree, const size_t &capacity, const float &alpha, float *max_priority, size_t *indices,
                   float *priorities, const size_t &batch_size, cudaStream_t stream);
void FifoSlice(const uint8_t *input, const size_t *indice, uint8_t *output, size_t batch_size, size_t column,
               cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMP_PRIORITY_REPLAY_BUFFER_IMPL_H_
