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

#include "plugin/device/gpu/kernel/cuda_impl/rl/priority_replay_buffer.cuh"
#include <limits>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

constexpr float kInitPriority = 1.0;
constexpr size_t kRootIdx = 1;
constexpr size_t kNumSubNode = 2;
constexpr size_t kMaxThreadPerBlock = 128;

__global__ void SumTreeInitKernel(SumTree *tree, float *max_priority, size_t size) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    tree[i].sum = 0;
    tree[i].min = std::numeric_limits<float>::max();

    if (i == 0) {
      *max_priority = kInitPriority;
    }
  }
}

__global__ void InitRandStateKernel(uint64_t seed, curandState *state) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, tid, 0, &(state[tid]));
}

__forceinline__ __device__ void SumTreeInsert(SumTree *tree, size_t idx, float priority) {
  tree[idx].sum = priority;
  tree[idx].min = priority;

  size_t parent = idx >> 1;
  while (parent >= kRootIdx) {
    size_t left_child = parent << 1;
    // TODO(WilfChen): Different thead may rewrite same node parallel since indices maybe repeat.
    tree[parent].sum = tree[left_child].sum + tree[left_child + 1].sum;
    tree[parent].min = min(tree[left_child].min, tree[left_child + 1].min);
    parent >>= 1;
  }
}

__global__ void SumTreePushKernel(SumTree *tree, float alpha, size_t idx, float *priority, float *max_priority) {
  float prio;
  if (!priority) {
    prio = powf(*max_priority, alpha);
  } else {
    *max_priority = max(*max_priority, *priority);
    prio = powf(*priority, alpha);
  }

  SumTreeInsert(tree, idx, prio);
}

__forceinline__ __device__ size_t GetPrefixSumIdx(SumTree *tree, size_t capacity, float prefix_sum) {
  size_t idx = kRootIdx;
  while (idx < capacity) {
    const float &left_priority = tree[kNumSubNode * idx].sum;
    if (prefix_sum <= left_priority) {
      idx = kNumSubNode * idx;
    } else {
      prefix_sum -= left_priority;
      idx = kNumSubNode * idx + 1;
    }
  }
  return idx - capacity;
}

__global__ void SumTreeSampleKernel(SumTree *tree, curandState *state, size_t capacity, float *beta, size_t batch_size,
                                    size_t *indices, float *weights) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += gridDim.x * blockDim.x) {
    size_t segment_len = tree[kRootIdx].sum / batch_size;
    float prefix_sum = (curand_uniform(&state[i]) + i) * segment_len;
    size_t idx = GetPrefixSumIdx(tree, capacity, prefix_sum);
    indices[i] = idx;
    weights[i] = powf((tree[idx + capacity].sum / tree[kRootIdx].min), -beta[0]);
  }
}

__global__ void SumTreeUpdateKernel(SumTree *tree, size_t capacity, float alpha, float *max_priority, size_t *indices,
                                    float *priorities, size_t batch_size) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += gridDim.x * blockDim.x) {
    size_t idx = indices[i];
    float priority = powf(priorities[i], alpha);
    MsAtomicMax(max_priority, priority);

    idx += capacity;
    SumTreeInsert(tree, idx, priority);
  }
}

void SumTreeInit(SumTree *tree, float *max_priority, const size_t &capacity, cudaStream_t stream) {
  size_t size = capacity * kNumSubNode;
  size_t block = std::min(size, kMaxThreadPerBlock);
  size_t grid = (size + block - 1) / block;
  SumTreeInitKernel<<<grid, block, 0, stream>>>(tree, max_priority, size);
}

void InitRandState(const size_t &batch_size, const uint64_t &seed, curandState *state, cudaStream_t stream) {
  size_t block = std::min(batch_size, kMaxThreadPerBlock);
  size_t grid = (batch_size + block - 1) / block;
  InitRandStateKernel<<<grid, block, 0, stream>>>(seed, state);
}

void SumTreePush(SumTree *tree, const float &alpha, const size_t &idx, const size_t &capacity, float *priority,
                 float *max_priority, cudaStream_t stream) {
  size_t idx_in_tree = idx + capacity;
  SumTreePushKernel<<<1, 1, 0, stream>>>(tree, alpha, idx_in_tree, priority, max_priority);
}

void SumTreeSample(SumTree *tree, curandState *state, const size_t &capacity, float *beta, const size_t &batch_size,
                   size_t *indices, float *weights, cudaStream_t stream) {
  size_t block = std::min(batch_size, kMaxThreadPerBlock);
  size_t grid = (batch_size + block - 1) / block;
  SumTreeSampleKernel<<<grid, block, 0, stream>>>(tree, state, capacity, beta, batch_size, indices, weights);
}

void SumTreeUpdate(SumTree *tree, const size_t &capacity, const float &alpha, float *max_priority, size_t *indices,
                   float *priorities, const size_t &batch_size, cudaStream_t stream) {
  size_t block = std::min(batch_size, kMaxThreadPerBlock);
  size_t grid = (batch_size + block - 1) / block;
  SumTreeUpdateKernel<<<grid, block, 0, stream>>>(tree, capacity, alpha, max_priority, indices, priorities, batch_size);
}

__global__ void FifoSliceKernel(const uint8_t *input, const size_t *indice, uint8_t *output, size_t batch_size,
                                size_t column) {
  for (size_t num = blockIdx.x * blockDim.x + threadIdx.x; num < batch_size * column; num += blockDim.x * gridDim.x) {
    size_t i = num / column;
    size_t j = num % column;
    size_t read_idex = indice[i] * column + j;
    output[num] = input[read_idex];
  }
}

void FifoSlice(const uint8_t *input, const size_t *indice, uint8_t *output, size_t batch_size, size_t column,
               cudaStream_t stream) {
  size_t num = batch_size * column;
  size_t block = std::min(num, kMaxThreadPerBlock);
  size_t grid = (num + block - 1) / block;
  FifoSliceKernel<<<grid, block, 0, stream>>>(input, indice, output, batch_size, column);
  return;
}
