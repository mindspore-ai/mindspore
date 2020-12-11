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

#include "backend/kernel_compiler/gpu/cuda_impl/unsorted_segment_max.cuh"
#include <limits>

template <typename T, typename S>
__global__ void UnsortedSegmentMax(const T *input, const S *segment_ids, const int64_t num_segments, size_t outer_size,
                                   size_t inner_size, bool fp16_flag, T init_K, T *output) {
  if (fp16_flag) {
    init_K = __int2half_rd(-65504);  // min value representable by float16
  }

  for (size_t t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < KWARPSIZE * num_segments * inner_size;
       t_idx += blockDim.x * gridDim.x) {
    size_t segment_id = t_idx / KWARPSIZE / inner_size;
    size_t inner_id = t_idx / KWARPSIZE % inner_size;
    size_t lane_id = threadIdx.x % KWARPSIZE;
    T threadK = init_K;

    for (size_t i = lane_id; i < outer_size; i += KWARPSIZE) {
      if (segment_ids[i] != segment_id) continue;
      T other_K = input[i * inner_size + inner_id];
      if (threadK < other_K) {
        threadK = other_K;
      }
    }
    __syncwarp();

    for (size_t offset = KWARPSIZE / 2; offset > 0; offset /= 2) {
      T other_K = __shfl_down_sync(0xffffffff, threadK, offset);
      if (threadK < other_K) {
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

template <typename T, typename S>
void CalUnsortedSegmentMax(const T *input, const S *segment_ids, const int64_t num_segments, size_t outer_size,
                           size_t inner_size, T *output, cudaStream_t stream) {
  size_t size = (inner_size * KWARPSIZE * num_segments);
  bool fp16_flag = false;
  // handle fp16 min value
  if (std::is_same<T, half>::value) {
    fp16_flag = true;
  }
  T init_K = std::numeric_limits<T>::lowest();
  UnsortedSegmentMax<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, segment_ids, num_segments, outer_size,
                                                                   inner_size, fp16_flag, init_K, output);
  return;
}

template void CalUnsortedSegmentMax<float, int>(const float *input, const int *segment_ids, const int64_t num_segments,
                                                size_t outer_size, size_t inner_size, float *output,
                                                cudaStream_t stream);
template void CalUnsortedSegmentMax<float, int64_t>(const float *input, const int64_t *segment_ids,
                                                    const int64_t num_segments, size_t outer_size, size_t inner_size,
                                                    float *output, cudaStream_t stream);
template void CalUnsortedSegmentMax<half, int>(const half *input, const int *segment_ids, const int64_t num_segments,
                                               size_t outer_size, size_t inner_size, half *output, cudaStream_t stream);
template void CalUnsortedSegmentMax<half, int64_t>(const half *input, const int64_t *segment_ids,
                                                   const int64_t num_segments, size_t outer_size, size_t inner_size,
                                                   half *output, cudaStream_t stream);
template void CalUnsortedSegmentMax<int, int>(const int *input, const int *segment_ids, const int64_t num_segments,
                                              size_t outer_size, size_t inner_size, int *output, cudaStream_t stream);
template void CalUnsortedSegmentMax<int, int64_t>(const int *input, const int64_t *segment_ids,
                                                  const int64_t num_segments, size_t outer_size, size_t inner_size,
                                                  int *output, cudaStream_t stream);
