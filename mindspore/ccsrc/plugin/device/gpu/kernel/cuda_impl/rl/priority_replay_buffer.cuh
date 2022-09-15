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
#include <thrust/detail/minmax.h>
#include <algorithm>
#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

// Use template to support tree type generalize since inheritance is not supported by CUDA.
struct SumMinTree {
  float sum;
  float min;

  __host__ __device__ __forceinline__ void init() {
    sum = 0;
    min = std::numeric_limits<float>::max();
  }

  __host__ __device__ __forceinline__ void set(float priority) {
    sum = priority;
    min = priority;
  }

  __host__ __device__ __forceinline__ void reduce(SumMinTree *lhs, SumMinTree *rhs, SumMinTree *parent) {
    parent->sum = lhs->sum + rhs->sum;
    parent->min = thrust::min(lhs->min, rhs->min);
  }
};

struct SumMinMaxTree {
  float sum;
  float min;
  float max;

  __host__ __device__ __forceinline__ void init() {
    sum = 0;
    min = std::numeric_limits<float>::max();
    max = std::numeric_limits<float>::min();
  }

  __host__ __device__ __forceinline__ void set(float priority) {
    sum = priority;
    min = priority;
    max = priority;
  }

  __host__ __device__ __forceinline__ void reduce(SumMinMaxTree *lhs, SumMinMaxTree *rhs, SumMinMaxTree *parent) {
    parent->sum = lhs->sum + rhs->sum;
    parent->min = thrust::min(lhs->min, rhs->min);
    parent->max = thrust::max(lhs->min, rhs->min);
  }
};

// Init Random state
void CUDA_LIB_EXPORT InitRandState(const size_t &size, const uint64_t &seed, curandState *state, cudaStream_t stream);

// Returns the slice of the input corresponding to the elements of `indice` on the first axis.
void CUDA_LIB_EXPORT FifoSlice(const uint8_t *input, const size_t *indice, uint8_t *output, size_t size, size_t column,
                               cudaStream_t stream);

// Init Segment Tree: Fill all item with the specified value
template <typename T>
CUDA_LIB_EXPORT void SumTreeInit(T *tree, float *max_priority, const size_t &capacity, cudaStream_t stream);

// Push one item to the tree. Set item with max_priority if the priority not provided.
template <typename T>
CUDA_LIB_EXPORT void SumTreePush(T *tree, const float &alpha, const size_t &idx, const size_t &capacity,
                                 float *priority, float *max_priority, cudaStream_t stream);

// Sample a batch item. Return indices and correction weights.
template <typename T>
CUDA_LIB_EXPORT void SumTreeSample(T *tree, curandState *state, const size_t &capacity, float *beta,
                                   const size_t &batch_size, size_t *indices, float *weights, cudaStream_t stream);

CUDA_LIB_EXPORT void SumTreeGetGlobalIdx(size_t batch_size, size_t *indices, size_t total_num, size_t capacity,
                                         cudaStream_t stream);

// Update item priority.
template <typename T>
CUDA_LIB_EXPORT void SumTreeUpdate(T *tree, const size_t &capacity, const size_t &last_idx, const float &alpha,
                                   float *max_priority, size_t *indices, float *priorities, const size_t &batch_size,
                                   cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMP_PRIORITY_REPLAY_BUFFER_IMPL_H_
