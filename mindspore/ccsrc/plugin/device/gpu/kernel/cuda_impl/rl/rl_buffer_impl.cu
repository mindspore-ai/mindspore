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

#include "plugin/device/gpu/kernel/cuda_impl/rl/rl_buffer_impl.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__global__ void BufferAppendKernel(const int64_t capacity, const size_t size, const int *index, const int exp_batch,
                                   unsigned char *buffer, const unsigned char *exp) {
  size_t index_ = index[0];
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (i >= (size / exp_batch) * (capacity - index[0])) {
      index_ = i - (size / exp_batch) * (capacity - index[0]);  // The exp_batch >= 1, guaranteed by op prim.
    } else {
      index_ = i + index[0] * size / exp_batch;
    }
    buffer[index_] = exp[i];
  }
}

__global__ void IncreaseCountKernel(const int64_t capacity, const int exp_batch, int *count, int *head, int *index) {
  int index_ = 0;
  if (count[0] <= capacity - 1 && head[0] == 0) {
    index_ = count[0];
    count[0] += exp_batch;
    if (count[0] > capacity) {
      count[0] = capacity;
      head[0] = (exp_batch + count[0] - capacity) % capacity;
    }
  } else {
    index_ = head[0];
    if (head[0] == count[0])
      head[0] = 0;
    else
      head[0] = (exp_batch + head[0]) % capacity;
  }
  index[0] = index_;
}

__global__ void ReMappingIndexKernel(const int *count, const int *head, const int *origin_index, int *index) {
  index[0] = origin_index[0];
  if (index[0] < 0) {
    index[0] += count[0];
  }
  if (!(index[0] >= 0 && index[0] < count[0])) {
    printf("[ERROR] The index %d is out of range:[%d, %d).", origin_index[0], -1 * count[0], count[0]);
    index[0] = -1;
    return;
  }
  int t = count[0] - head[0];
  if (index[0] < t) {
    index[0] += head[0];
  } else {
    index[0] -= t;
  }
}

__global__ void BufferGetItemKernel(const size_t size, const int *index, const size_t one_exp_len,
                                    const unsigned char *buffer, unsigned char *out) {
  if (index[0] == -1) {
    return;
  }
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    out[i] = buffer[i + index[0] * one_exp_len];
  }
}

__global__ void CheckBatchSizeKernel(const int *count, const int *head, const size_t batch_size,
                                     const int64_t capacity) {
  if ((head[0] > 0 && int64_t(batch_size) > capacity) || (head[0] == 0 && batch_size > size_t(count[0]))) {
    printf("[ERROR] The batch size %d is larger than total buffer size %d", static_cast<int>(batch_size),
           (capacity > static_cast<int64_t>(count[0]) ? static_cast<int>(count[0]) : static_cast<int>(capacity)));
  }
}

__global__ void BufferSampleKernel(const size_t size, const size_t one_element, const unsigned int *index,
                                   const unsigned char *buffer, unsigned char *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    out[i] = buffer[index[i / one_element] * one_element + i % one_element];
  }
}

__global__ void SetupKernel(const int seed, curandState *state, const int size) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    curand_init(seed, i, 0, &state[i]);
  }
}

__global__ void SrandUInt(const int size, curandState *globalState, unsigned int *value, unsigned int *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    out[i] = curand(&globalState[i]);
    value[i] = i;
  }
}

template <typename T>
__global__ void SrandUniformInt(const int size, curandState *globalState, const int upBound, T *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    // curand_uniform return a pseudorandom floats uniformly distributed between 0.0 and 1.0, where 1.0 is
    // included and 0.0 is excluded. So decrease the upBound by 1 to avoid out of range.
    out[i] = static_cast<T>(curand_uniform(&globalState[i]) * (upBound - 1));
  }
}

void BufferAppend(const int64_t capacity, const size_t size, const int *index, const int exp_batch,
                  unsigned char *buffer, const unsigned char *exp, cudaStream_t cuda_stream) {
  BufferAppendKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(capacity, size, index, exp_batch, buffer, exp);
}

void IncreaseCount(const int64_t capacity, const int exp_batch, int *count, int *head, int *index,
                   cudaStream_t cuda_stream) {
  IncreaseCountKernel<<<1, 1, 0, cuda_stream>>>(capacity, exp_batch, count, head, index);
}

void ReMappingIndex(const int *count, const int *head, const int *origin_index, int *index, cudaStream_t cuda_stream) {
  ReMappingIndexKernel<<<1, 1, 0, cuda_stream>>>(count, head, origin_index, index);
}

void BufferGetItem(const size_t size, const int *index, const size_t one_exp_len, const unsigned char *buffer,
                   unsigned char *out, cudaStream_t cuda_stream) {
  BufferGetItemKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, index, one_exp_len, buffer, out);
}

void CheckBatchSize(const int *count, const int *head, const size_t batch_size, const int64_t capacity,
                    cudaStream_t cuda_stream) {
  CheckBatchSizeKernel<<<1, 1, 0, cuda_stream>>>(count, head, batch_size, capacity);
}

void BufferSample(const size_t size, const size_t one_element, const unsigned int *index, const unsigned char *buffer,
                  unsigned char *out, cudaStream_t cuda_stream) {
  BufferSampleKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, one_element, index, buffer, out);
}

void RandInit(const int size, const int seed, curandState *state, cudaStream_t stream) {
  SetupKernel<<<(size + 255) / 256, 256, 0, stream>>>(seed, state, size);
}

void RandomGen(const int size, curandState *globalState, unsigned int *value, unsigned int *key, cudaStream_t stream) {
  // 1 Generate two list, value for random int num, key for sequence form [0, size).
  SrandUInt<<<(size + 255) / 256, 256, 0, stream>>>(size, globalState, value, key);
  auto policy = thrust::cuda::par.on(stream);
  thrust::device_ptr<unsigned int> dev_data_ptr(value);
  thrust::device_ptr<unsigned int> dev_key_ptr(key);
  // 2 Sort the key and get the sorted indexes.
  thrust::sort_by_key(policy, dev_key_ptr, dev_key_ptr + size, dev_data_ptr);
}

template <typename T>
void RandomGenUniform(const int size, curandState *globalState, const int up_bound, T *indexes, cudaStream_t stream) {
  SrandUniformInt<<<(size + 255) / 256, 256, 0, stream>>>(size, globalState, up_bound, indexes);
}

template CUDA_LIB_EXPORT
void RandomGenUniform<unsigned int>(const int size, curandState *globalState, const int up_bound,
                                    unsigned int *indexes, cudaStream_t stream);
template CUDA_LIB_EXPORT
void RandomGenUniform<size_t>(const int size, curandState *globalState, const int up_bound, size_t *indexes,
                              cudaStream_t stream);
