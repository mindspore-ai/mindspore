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

#include "backend/kernel_compiler/gpu/cuda_impl/unsorted_segment_min.cuh"
#include <limits>

template <typename T>
__device__ __forceinline__ void max_val_init(T *init_val) {
  *init_val = std::numeric_limits<T>::max();
}
// Handle fp16 differently for assignment
template <>
__device__ __forceinline__ void max_val_init(half *init_val) {
  *init_val = __int2half_rd(65504);  // Max value for Half
}

template <typename T>
__global__ void UnsortedSegmentMin(const T *input, const int *segment_ids, const int64_t num_segments,
                                   size_t outer_size, size_t inner_size, T init_K, T *output) {
  max_val_init(&init_K);
  for (int t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < KWARPSIZE * num_segments * inner_size;
       t_idx += blockDim.x * gridDim.x) {
    int segment_id = t_idx / KWARPSIZE / inner_size;
    int inner_id = t_idx / KWARPSIZE % inner_size;
    int lane_id = threadIdx.x % KWARPSIZE;
    T threadK = init_K;

    for (int i = lane_id; i < outer_size; i += KWARPSIZE) {
      if (segment_ids[i] != segment_id) continue;
      T other_K = input[i * inner_size + inner_id];
      if (threadK > other_K) {
        threadK = other_K;
      }
    }
    __syncwarp();
    for (int offset = KWARPSIZE / 2; offset > 0; offset /= 2) {
      T other_K = __shfl_down_sync(0xffffffff, threadK, offset);
      if (threadK > other_K) {
        threadK = other_K;
      }
    }
    __syncwarp();

    if (lane_id == 0) {
      output[segment_id * inner_size + inner_id] = threadK;
    }
    __syncthreads();
  }
}

template <typename T>
void CalUnsortedSegmentMin(const T *input, const int *segment_ids, const int64_t num_segments, size_t outer_size,
                           size_t inner_size, T *output, cudaStream_t stream) {
  int size = (inner_size * KWARPSIZE * num_segments);
  T init_K = std::numeric_limits<T>::lowest();  // only init here - overwritten later
  UnsortedSegmentMin<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, segment_ids, num_segments, outer_size,
                                                                   inner_size, init_K, output);
  return;
}

template void CalUnsortedSegmentMin<float>(const float *input, const int *segment_ids, const int64_t num_segments,
                                           size_t outer_size, size_t inner_size, float *output, cudaStream_t stream);
template void CalUnsortedSegmentMin<half>(const half *input, const int *segment_ids, const int64_t num_segments,
                                          size_t outer_size, size_t inner_size, half *output, cudaStream_t stream);
template void CalUnsortedSegmentMin<int>(const int *input, const int *segment_ids, const int64_t num_segments,
                                         size_t outer_size, size_t inner_size, int *output, cudaStream_t stream);
