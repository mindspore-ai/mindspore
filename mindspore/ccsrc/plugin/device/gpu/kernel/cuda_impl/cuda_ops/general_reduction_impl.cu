/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <limits>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/general_reduction_impl.cuh"

const int kWarpSize = 32;
const int kBlockSize = 512;
const int kWarpGroup = 4;
const int kNumWarps = kBlockSize / kWarpSize;   // 16
const int kGroupSize = kWarpGroup * kWarpSize;  // 128

// Mode selection constant
const int kMaxThreadLoop = 4;
const int kMaxWarpLoop = kWarpSize * 3;    // 32 * 3 = 96
const int kMaxGroupLoop = kGroupSize * 3;  // 128 * 3 =
                                           // 384

template <typename T, typename S>
struct Cmp {
  __device__ static inline bool lt(T a, T b, S i, S j) { return (a < b) || ((a == b) && ((i < 0 || j < i) && j >= 0)); }
  __device__ static inline bool gt(T a, T b, S i, S j) { return (a > b) || ((a == b) && ((i < 0 || j < i)) && j >= 0); }
};

template <typename T>
inline __device__ void ConditionAssign(bool is_assign, T *x, const T &y) {
  (*x) = is_assign ? y : (*x);
}

template <typename T, typename S>
__global__ void ThreadReduction(bool small, size_t outer_size, size_t bound, size_t inner_size, const T *input,
                                T *output, S *output_index, T init_K) {
  const S init_V = static_cast<S>(-1);

  for (int t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < outer_size * inner_size;
       t_idx += blockDim.x * gridDim.x) {
    int outer_id = t_idx / inner_size;
    int inner_id = t_idx % inner_size;

    T threadK = init_K;
    S threadV = init_V;

    for (int i = 0; i < bound; i++) {
      T other_K = input[outer_id * bound * inner_size + i * inner_size + inner_id];
      S other_V = i;
      bool is_winner =
        small ? Cmp<T, S>::gt(threadK, other_K, threadV, other_V) : Cmp<T, S>::lt(threadK, other_K, threadV, other_V);
      ConditionAssign(is_winner, &threadK, other_K);
      ConditionAssign(is_winner, &threadV, other_V);
    }

    output[outer_id * inner_size + inner_id] = threadK;
    output_index[outer_id * inner_size + inner_id] = threadV;
  }
}

template <typename T, typename S>
__global__ void WarpReduction(bool small, size_t outer_size, size_t bound, size_t inner_size, const T *input, T *output,
                              S *output_index, T init_K) {
  const S init_V = static_cast<S>(-1);

  for (int t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < kWarpSize * outer_size * inner_size;
       t_idx += blockDim.x * gridDim.x) {
    int outer_id = t_idx / kWarpSize / inner_size;
    int inner_id = t_idx / kWarpSize % inner_size;

    int laneId = threadIdx.x % kWarpSize;

    T threadK = init_K;
    S threadV = init_V;

    for (int i = laneId; i < bound; i += kWarpSize) {
      T other_K = input[outer_id * bound * inner_size + i * inner_size + inner_id];
      S other_V = i;
      bool is_winner =
        small ? Cmp<T, S>::gt(threadK, other_K, threadV, other_V) : Cmp<T, S>::lt(threadK, other_K, threadV, other_V);
      ConditionAssign(is_winner, &threadK, other_K);
      ConditionAssign(is_winner, &threadV, other_V);
    }
    __syncwarp();

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      T other_K = __shfl_down_sync(0xffffffff, threadK, offset);
      S other_V = __shfl_down_sync(0xffffffff, threadV, offset);

      bool is_winner =
        small ? Cmp<T, S>::gt(threadK, other_K, threadV, other_V) : Cmp<T, S>::lt(threadK, other_K, threadV, other_V);
      ConditionAssign(is_winner, &threadK, other_K);
      ConditionAssign(is_winner, &threadV, other_V);
    }

    __syncwarp();

    if (laneId == 0) {
      output[outer_id * inner_size + inner_id] = threadK;
      output_index[outer_id * inner_size + inner_id] = threadV;
    }
    __syncthreads();
  }
}

template <typename T, typename S>
__global__ void Warp4Reduction(bool small, size_t outer_size, size_t bound, size_t inner_size, const T *input,
                               T *output, S *output_index, T init_K) {
  __shared__ T shared_K[kNumWarps];
  __shared__ S shared_V[kNumWarps];
  const S init_V = static_cast<S>(-1);

  for (int t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < kGroupSize * outer_size * inner_size;
       t_idx += blockDim.x * gridDim.x) {
    int outer_id = t_idx / kGroupSize / inner_size;
    int inner_id = t_idx / kGroupSize % inner_size;

    int groupId = threadIdx.x / kGroupSize;
    int tgId = threadIdx.x % kGroupSize;
    int warpId = threadIdx.x / kWarpSize;
    int laneId = threadIdx.x % kWarpSize;

    T threadK = init_K;
    S threadV = init_V;

    if (laneId == 0) {
      shared_K[warpId] = init_K;
      shared_V[warpId] = init_V;
    }
    __syncthreads();

    for (int i = tgId; i < bound; i += kGroupSize) {
      T other_K = input[outer_id * bound * inner_size + i * inner_size + inner_id];
      S other_V = i;
      bool is_winner =
        small ? Cmp<T, S>::gt(threadK, other_K, threadV, other_V) : Cmp<T, S>::lt(threadK, other_K, threadV, other_V);
      ConditionAssign(is_winner, &threadK, other_K);
      ConditionAssign(is_winner, &threadV, other_V);
    }
    __syncwarp();

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      T other_K = __shfl_down_sync(0xffffffff, threadK, offset);
      S other_V = __shfl_down_sync(0xffffffff, threadV, offset);

      bool is_winner =
        small ? Cmp<T, S>::gt(threadK, other_K, threadV, other_V) : Cmp<T, S>::lt(threadK, other_K, threadV, other_V);
      ConditionAssign(is_winner, &threadK, other_K);
      ConditionAssign(is_winner, &threadV, other_V);
    }

    __syncwarp();

    if (laneId == 0) {
      shared_K[warpId] = threadK;
      shared_V[warpId] = threadV;
    }
    __syncthreads();

    if (tgId < 2) {
      bool is_winner =
        small ? Cmp<T, S>::gt(shared_K[(groupId * kWarpGroup) + tgId], shared_K[(groupId * kWarpGroup) + tgId + 2],
                              shared_V[(groupId * kWarpGroup) + tgId], shared_V[(groupId * kWarpGroup) + tgId + 2])
              : Cmp<T, S>::lt(shared_K[(groupId * kWarpGroup) + tgId], shared_K[(groupId * kWarpGroup) + tgId + 2],
                              shared_V[(groupId * kWarpGroup) + tgId], shared_V[(groupId * kWarpGroup) + tgId + 2]);
      ConditionAssign(is_winner, (shared_K + (groupId * kWarpGroup) + tgId),
                      (shared_K[(groupId * kWarpGroup) + tgId + 2]));
      ConditionAssign(is_winner, (shared_V + (groupId * kWarpGroup) + tgId),
                      (shared_V[(groupId * kWarpGroup) + tgId + 2]));
    }
    __syncwarp();

    if (tgId == 0) {
      bool is_winner =
        small ? Cmp<T, S>::gt(shared_K[(groupId * kWarpGroup) + tgId], shared_K[(groupId * kWarpGroup) + tgId + 1],
                              shared_V[(groupId * kWarpGroup) + tgId], shared_V[(groupId * kWarpGroup) + tgId + 1])
              : Cmp<T, S>::lt(shared_K[(groupId * kWarpGroup) + tgId], shared_K[(groupId * kWarpGroup) + tgId + 1],
                              shared_V[(groupId * kWarpGroup) + tgId], shared_V[(groupId * kWarpGroup) + tgId + 1]);
      ConditionAssign(is_winner, (shared_K + (groupId * kWarpGroup) + tgId),
                      (shared_K[(groupId * kWarpGroup) + tgId + 1]));
      ConditionAssign(is_winner, (shared_V + (groupId * kWarpGroup) + tgId),
                      (shared_V[(groupId * kWarpGroup) + tgId + 1]));

      // The first thread of each group write output
      output[outer_id * inner_size + inner_id] = shared_K[groupId * kWarpGroup];
      output_index[outer_id * inner_size + inner_id] = shared_V[groupId * kWarpGroup];
    }
    __syncthreads();
  }
}

template <typename T, typename S>
__global__ void BlockReduction(bool small, size_t outer_size, size_t bound, size_t inner_size, const T *input,
                               T *output, S *output_index, T init_K) {
  __shared__ T shared_K[kNumWarps];
  __shared__ S shared_V[kNumWarps];
  const S init_V = static_cast<S>(-1);

  for (int t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < kBlockSize * outer_size * inner_size;
       t_idx += blockDim.x * gridDim.x) {
    int outer_id = t_idx / kBlockSize / inner_size;
    int inner_id = t_idx / kBlockSize % inner_size;

    int tgId = threadIdx.x % kBlockSize;
    int warpId = threadIdx.x / kWarpSize;
    int laneId = threadIdx.x % kWarpSize;

    T threadK = init_K;
    S threadV = init_V;

    if (laneId == 0) {
      shared_K[warpId] = init_K;
      shared_V[warpId] = init_V;
    }
    __syncthreads();

    for (int i = tgId; i < bound; i += kBlockSize) {
      T other_K = input[outer_id * bound * inner_size + i * inner_size + inner_id];
      S other_V = i;
      bool is_winner =
        small ? Cmp<T, S>::gt(threadK, other_K, threadV, other_V) : Cmp<T, S>::lt(threadK, other_K, threadV, other_V);
      ConditionAssign(is_winner, &threadK, other_K);
      ConditionAssign(is_winner, &threadV, other_V);
    }
    __syncwarp();

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      T other_K = __shfl_down_sync(0xffffffff, threadK, offset);
      S other_V = __shfl_down_sync(0xffffffff, threadV, offset);

      bool is_winner =
        small ? Cmp<T, S>::gt(threadK, other_K, threadV, other_V) : Cmp<T, S>::lt(threadK, other_K, threadV, other_V);
      ConditionAssign(is_winner, &threadK, other_K);
      ConditionAssign(is_winner, &threadV, other_V);
    }

    __syncwarp();

    if (laneId == 0) {
      shared_K[warpId] = threadK;
      shared_V[warpId] = threadV;
    }
    __syncthreads();

    // Shared memory reduction
    // There are 16 items in shared memory, can be reduced within one warp.
    if (warpId == 0) {
      threadK = laneId < kNumWarps ? shared_K[laneId] : init_K;
      threadV = laneId < kNumWarps ? shared_V[laneId] : init_V;
    }
    __syncwarp();

    if (warpId == 0) {
      for (int offset = kWarpSize / 4; offset > 0; offset /= 2) {
        T other_K = __shfl_down_sync(0xffffffff, threadK, offset);
        S other_V = __shfl_down_sync(0xffffffff, threadV, offset);

        bool is_winner =
          small ? Cmp<T, S>::gt(threadK, other_K, threadV, other_V) : Cmp<T, S>::lt(threadK, other_K, threadV, other_V);
        ConditionAssign(is_winner, &threadK, other_K);
        ConditionAssign(is_winner, &threadV, other_V);
      }
    }
    __syncwarp();

    if (warpId == 0 && laneId == 0) {
      output[outer_id * inner_size + inner_id] = threadK;
      output_index[outer_id * inner_size + inner_id] = threadV;
    }
  }
}

template <typename T, typename S>
void GeneralReductionImpl(bool small, size_t outer_size, size_t bound, size_t inner_size, const T *input, T *output,
                      S *output_index, T init_K, cudaStream_t stream) {
  int block_num_limit = outer_size * inner_size;
  if (bound <= kMaxThreadLoop) {
    ThreadReduction<T, S><<<GET_BLOCKS(block_num_limit * kBlockSize), kBlockSize, 0, stream>>>(
      small, outer_size, bound, inner_size, input, output, output_index, init_K);
  } else if (bound <= kMaxWarpLoop) {
    WarpReduction<T, S><<<GET_BLOCKS(block_num_limit * kBlockSize), kBlockSize, 0, stream>>>(
      small, outer_size, bound, inner_size, input, output, output_index, init_K);
  } else if (bound <= kMaxGroupLoop) {
    Warp4Reduction<T, S><<<GET_BLOCKS(block_num_limit * kBlockSize), kBlockSize, 0, stream>>>(
      small, outer_size, bound, inner_size, input, output, output_index, init_K);
  } else {
    BlockReduction<T, S><<<GET_BLOCKS(block_num_limit * kBlockSize), kBlockSize, 0, stream>>>(
      small, outer_size, bound, inner_size, input, output, output_index, init_K);
  }
}

template <typename T, typename S>
cudaError_t CalGeneralReduction(bool small, const T *input, const size_t bound, const size_t outerSize,
                         const size_t innerSize, S *output_index, T *output, cudaStream_t stream) {
  T init_K = small ? std::numeric_limits<T>::max() : std::numeric_limits<T>::lowest();
  GeneralReductionImpl(small, outerSize, bound, innerSize, input, output, output_index, init_K, stream);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template <typename S>
cudaError_t CalGeneralReduction(bool small, const half *input, const size_t bound, const size_t outerSize,
                        const size_t innerSize, S *output_index, half *output, cudaStream_t stream) {
  half init_K = small ? static_cast<half>(65504) : static_cast<half>(-65504);
  GeneralReductionImpl(small, outerSize, bound, innerSize, input, output, output_index, init_K, stream);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const int8_t *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  int8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const int64_t *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  int64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const uint8_t *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  uint8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const uint64_t *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  uint64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const int16_t *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  int16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const int32_t *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  int32_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const uint16_t *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  uint16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const uint32_t *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  uint32_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const double *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  double *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const float *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGeneralReduction(bool small, const half *input, const size_t bound_,
                                                  const size_t outerSize_, const size_t innerSize_, int *index,
                                                  half *output, cudaStream_t cuda_stream);
