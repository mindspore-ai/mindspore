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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/topk_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/topk_lib.cuh"
#include <limits>
#include <algorithm>
#include "include/cuda_fp16.h"

const int kMaxQueue = 128;

#define TOPK_HELPER(BLOCK, NUM_WARP_Q, NUM_THREAD_Q, IS_DESCEND)                                                   \
  do {                                                                                                             \
    TopKBlock<T, S, NUM_WARP_Q, NUM_THREAD_Q, BLOCK, IS_DESCEND>                                                   \
      <<<block_num_limit, BLOCK, 0, stream>>>(outer_size, inner_size, input, output, output_index, k_cut, init_K); \
  } while (0)

#define LEFT_INSERT_THREAD_QUEUE(_k, _v)                                                                            \
  do {                                                                                                              \
    if (is_descend ? CmpKV<T, S>::gt(_k, _v, (*ceil_K), (*ceil_V)) : CmpKV<T, S>::lt(_k, _v, (*ceil_K), (*ceil_V))) \
      break;                                                                                                        \
    if (is_descend ? CmpKV<T, S>::gt(_k, _v, warp_K_top, warp_V_top)                                                \
                   : CmpKV<T, S>::lt(_k, _v, warp_K_top, warp_V_top)) {                                             \
      {                                                                                                             \
        _Pragma("unroll") for (int i = thread_queue - 1; i > 0; --i) {                                              \
          threadK[i] = threadK[i - 1];                                                                              \
          threadV[i] = threadV[i - 1];                                                                              \
        }                                                                                                           \
      }                                                                                                             \
      threadK[0] = _k;                                                                                              \
      threadV[0] = _v;                                                                                              \
      ++num_vals;                                                                                                   \
    }                                                                                                               \
  } while (0)

template <typename T, typename S, int warp_queue, int thread_queue, int threads_per_block, bool is_descend>
inline __device__ void TopKInBuffer(T *shared_K, S *shared_V, int *watermark, T *ceil_K, S *ceil_V, int laneId) {
  constexpr int kNumWarps = threads_per_block / kWarpSize;  // kNumWarps is 1024/32=32

  // If kNumWarps != kWarpSize, need to adjust this code as we are using lanes to aggregate kNumWArps
  // if kWarpSize > kNumWarps, each warp now has kWarpSize/kNumWarps threads when we only need kNumWarp
  constexpr int kWarpQueuePerLane = warp_queue * kNumWarps / kWarpSize;
  constexpr int kLanesPerWarp = kWarpSize / kNumWarps;

  T last_K = shared_K[laneId * kWarpQueuePerLane + kWarpQueuePerLane - 1];
  S last_V = shared_V[laneId * kWarpQueuePerLane + kWarpQueuePerLane - 1];

  __syncwarp();

  // Find KCut:
  // - The last element of each warp is the lowest in that warp
  // --- If we have multiple lanes per warp look at last lane per warp
  // - k_cut will the higheset of last elements of each warp
  for (int offset = kNumWarps / 2; offset > 0; offset /= 2) {
    T other_K = __shfl_down_sync(0xffffffff, last_K, offset * kLanesPerWarp);
    S other_V = __shfl_down_sync(0xffffffff, last_V, offset * kLanesPerWarp);

    bool is_greater = CmpKV<T, S>::gt(other_K, other_V, last_K, last_V);
    ConditionalAssign(is_greater, &last_K, other_K);
    ConditionalAssign(is_greater, &last_V, other_V);
  }
  __syncwarp();

  // want to fetch last_K from last lane of first warp
  if (laneId == kLanesPerWarp - 1) {
    *ceil_K = last_K;
    *ceil_V = last_V;
  }
  __syncwarp();

  // calculate index cut by last_K.  Do this per thread/lane instead of per warp
  int L = 0;
  int R = kWarpQueuePerLane;
  while (L < R) {
    int m = (L + R) / 2;
    CmpKV<T, S>::gt(shared_K[laneId * kWarpQueuePerLane + m],
                    shared_V[laneId * kWarpQueuePerLane + m], (*ceil_K), (*ceil_V))
      ? L = m + 1
      : R = m;
  }
  __syncwarp();

  // R is calculated per thread --> sum over all threads and not just all warps
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    R += __shfl_down_sync(0xffffffff, R, offset);
  }

  __syncwarp();

  if (laneId == 0) {
    watermark[0] = R;
  }
  __syncwarp();
}

template <typename T, typename S, int warp_queue, int thread_queue, int threads_per_block, bool is_descend>
inline __device__ void TopKStep(const int &outer_size, const int &inner_size, const T *input, T *output,
                                S *output_index, S k_cut, const T &init_K, const int &outer_id, T *shared_K,
                                S *shared_V, int *watermark, T *threadK, S *threadV, T *ceil_K, S *ceil_V, S *k_prime) {
  constexpr int kNumWarps = threads_per_block / kWarpSize;
  constexpr S init_V = static_cast<S>(-1);

  T *warp_K;
  S *warp_V;

  T warp_K_top = init_K;
  S warp_V_top = init_V;
  int k_minus_1 = (k_cut <= kMaxQueue ? k_cut - 1 : kMaxQueue - 1);
  int num_vals = 0;
  int limit = (inner_size / kWarpSize) * kWarpSize;

  _Pragma("unroll") for (int i = 0; i < thread_queue; ++i) {
    threadK[i] = init_K;
    threadV[i] = init_V;
  }

  int laneId = GetLaneId();
  int warpId = threadIdx.x / kWarpSize;  // 0,1,2 or 3

  warp_K = shared_K + warpId * warp_queue;
  warp_V = shared_V + warpId * warp_queue;

  for (int i = laneId; i < warp_queue; i += kWarpSize) {
    warp_K[i] = init_K;
    warp_V[i] = init_V;
  }

  __syncwarp();

  int i = threadIdx.x;
  for (; i < limit; i += threads_per_block) {
    LEFT_INSERT_THREAD_QUEUE((input[outer_id * inner_size + i]), (outer_id * inner_size + i));

    bool needSort = (num_vals == thread_queue);
    needSort = __any_sync(0xffffffff, needSort);
    if (!needSort) continue;

    MergeWarpQueue<T, S, warp_queue, thread_queue, is_descend>(threadK, threadV, warp_K, warp_V);

    num_vals = 0;
    _Pragma("unroll") for (int i = 0; i < thread_queue; ++i) {
      threadK[i] = init_K;
      threadV[i] = init_V;
    }
    warp_K_top = warp_K[k_minus_1];
    warp_V_top = warp_V[k_minus_1];
    __syncwarp();
  }

  if (i < inner_size) {
    LEFT_INSERT_THREAD_QUEUE((input[outer_id * inner_size + i]), (outer_id * inner_size + i));
  }

  MergeWarpQueue<T, S, warp_queue, thread_queue, is_descend>(threadK, threadV, warp_K, warp_V);
  __syncthreads();

  if (k_cut > kMaxQueue && warpId == 0) {
    TopKInBuffer<T, S, warp_queue, thread_queue, threads_per_block, is_descend>(shared_K, shared_V, watermark, ceil_K,
                                                                                ceil_V, laneId);
  }
  __syncthreads();

  // Wide sort doesn't sort properly if kNumWarp != kWarpSize, so pass kWarpsize
  SortBlockWide<kWarpSize, threads_per_block, T, S, warp_queue, is_descend>(shared_K, shared_V);
  __syncthreads();

  S k_step = (*k_prime) + watermark[0] <= k_cut ? watermark[0] : k_cut - (*k_prime);
  for (int i = threadIdx.x; i < k_step; i += blockDim.x) {
    output[outer_id * k_cut + (*k_prime) + i] = shared_K[i];
    output_index[outer_id * k_cut + (*k_prime) + i] = shared_V[i] % inner_size;
  }
  *k_prime += k_step;
  __syncthreads();
}

template <typename T, typename S, int warp_queue, int thread_queue, int threads_per_block, bool is_descend>
__global__ void TopKBlock(int outer_size, int inner_size, const T *input, T *output, S *output_index, S k_cut,
                          const T init_K) {
  constexpr int kNumWarps = threads_per_block / kWarpSize;

  __shared__ T shared_K[kNumWarps * warp_queue];
  __shared__ S shared_V[kNumWarps * warp_queue];
  __shared__ int watermark[1];
  __shared__ T ceil_K;
  __shared__ S ceil_V;

  T threadK[thread_queue];  // NOLINT
  S threadV[thread_queue];  // NOLINT

  for (int t_idx = blockIdx.x * blockDim.x + threadIdx.x; t_idx < blockDim.x * outer_size;
       t_idx += blockDim.x * gridDim.x) {
    S k_prime = 0;
    int outer_id = t_idx / blockDim.x;
    ceil_K = -init_K;
    ceil_V = -1;
    watermark[0] = k_cut;
    do {
      TopKStep<T, S, warp_queue, thread_queue, threads_per_block, is_descend>(
        outer_size, inner_size, input, output, output_index, k_cut, init_K, outer_id, shared_K, shared_V, watermark,
        threadK, threadV, &ceil_K, &ceil_V, &k_prime);
    } while (k_prime < k_cut);
  }
}

template <typename T, typename S>
void FastTopK(const int outer_size, const int inner_size, const T *input, S k_cut, T *output, S *output_index,
              const T init_K, cudaStream_t stream) {
  int block_num_limit = outer_size < 128 ? outer_size : 128;
  if (k_cut > inner_size) k_cut = inner_size;

  if (k_cut <= 32) {
    // num-threads-of-block, warp-queue-size, thread-queue-size
    TOPK_HELPER(256, 32, 2, true);
  } else if (k_cut <= 64) {
    TOPK_HELPER(256, 64, 3, true);
  } else if (k_cut <= 128) {
    TOPK_HELPER(256, 128, 3, true);
  } else {
    // cuda 11.6 has lower # threads.  Set lower number for all platforms for consistency
    TOPK_HELPER(256, 128, 3, true);
  }
}

template CUDA_LIB_EXPORT void FastTopK(const int outer_size, const int inner_size, const half *input, int k_cut,
                                       half *output, int *output_index, const half init_K, cudaStream_t stream);
template CUDA_LIB_EXPORT void FastTopK(const int outer_size, const int inner_size, const float *input, int k_cut,
                                       float *output, int *output_index, const float init_K, cudaStream_t stream);
