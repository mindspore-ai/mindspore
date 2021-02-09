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

#include "backend/kernel_compiler/gpu/cuda_impl/topk_lib.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/random_choice_with_mask_impl.cuh"

// Kernel started from here
#define L2_RCWM_HELPER(BLOCK, NUM_WARP_Q, NUM_THREAD_Q, IS_DESCEND)                      \
  do {                                                                                   \
    L2Rcwm<T, S, K, NUM_WARP_Q, NUM_THREAD_Q, BLOCK, IS_DESCEND>                         \
      <<<1, BLOCK, 0, stream>>>(seedc, input_size, input, output_mask, output_index, k); \
  } while (0)

#define LEFT_INSERT_THREAD_QUEUE(_k, _v)                                        \
  do {                                                                          \
    if (is_descend ? Cmp<T>::gt(_k, warp_K_top) : Cmp<T>::lt(_k, warp_K_top)) { \
      {                                                                         \
        _Pragma("unroll") for (int i = thread_queue - 1; i > 0; --i) {          \
          threadK[i] = threadK[i - 1];                                          \
          threadV[i] = threadV[i - 1];                                          \
        }                                                                       \
      }                                                                         \
      threadK[0] = _k;                                                          \
      threadV[0] = _v;                                                          \
      ++num_vals;                                                               \
    }                                                                           \
  } while (0)

template <typename T, typename S, typename K, int warp_queue, int thread_queue, int threads_per_block, bool is_descend>
__global__ void L2Rcwm(int seedc, int input_size, const K *input, K *output_mask, S *output_index, int k) {
  constexpr int kNumWarps = threads_per_block / kWarpSize;
  constexpr T init_K = static_cast<T>(-2.0);
  constexpr S init_V = static_cast<S>(0);

  __shared__ T shared_K[kNumWarps * warp_queue];
  __shared__ S shared_V[kNumWarps * warp_queue];

  curandState devState;
  curand_init(seedc, threadIdx.x, 0, &devState);

  T threadK[thread_queue];  // NOLINT
  S threadV[thread_queue];  // NOLINT

  T *warp_K;
  S *warp_V;

  T warp_K_top = init_K;
  int k_minus_1 = k - 1;
  int num_vals = 0;
  int limit = (input_size / kWarpSize) * kWarpSize;
  int i = threadIdx.x;

  // init begin
  _Pragma("unroll") for (int i = 0; i < thread_queue; ++i) {
    threadK[i] = init_K;
    threadV[i] = init_V;
  }

  int laneId = GetLaneId();
  int warpId = threadIdx.x / kWarpSize;  // 0,1,2 or 3

  // warp shared memory start address
  warp_K = shared_K + warpId * warp_queue;
  warp_V = shared_V + warpId * warp_queue;

  for (int i = laneId; i < warp_queue; i += kWarpSize) {
    warp_K[i] = init_K;
    warp_V[i] = init_V;
  }

  // sync till all threads init done
  __syncwarp();

  // insert begin
  for (; i < limit; i += threads_per_block) {
    T rand_num = input[i] ? __uint2float_rn(curand(&devState)) : init_K;
    LEFT_INSERT_THREAD_QUEUE(rand_num, i);

    // CHECK_AND_MERGE_THREAD_QUEUE() begin
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
    __syncwarp();
  }

  if (i < input_size) {
    T rand_num = input[i] ? __uint2float_rn(curand(&devState)) : init_K;
    LEFT_INSERT_THREAD_QUEUE(rand_num, i);
  }

  // reduce begin
  MergeWarpQueue<T, S, warp_queue, thread_queue, is_descend>(threadK, threadV, warp_K, warp_V);
  __syncthreads();
  SortBlockWide<kNumWarps, threads_per_block, T, S, warp_queue, is_descend>(shared_K, shared_V);

  // ship data from shared memory to output buffer
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    output_mask[i] = shared_K[i] > static_cast<T>(-1.0) ? true : false;
    output_index[i] = shared_V[i];
  }
}

template <typename T, typename S, typename K>
void RCWMScaleK(int seedc, int input_size, K *input, int k, S *output_index, K *output_mask, cudaStream_t stream) {
  if (k <= 32) {
    // num-threads-of-block, warp-queue-size, thread-queue-size
    L2_RCWM_HELPER(256, 32, 2, true);
  } else if (k <= 64) {
    L2_RCWM_HELPER(256, 64, 3, true);
  } else if (k <= 128) {
    L2_RCWM_HELPER(256, 128, 3, true);
  } else if (k <= 256) {
    L2_RCWM_HELPER(256, 256, 4, true);
  } else if (k <= 512) {
    L2_RCWM_HELPER(256, 512, 8, true);
  } else if (k <= 1024) {
    L2_RCWM_HELPER(128, 1024, 8, true);
  } else if (k <= 2048) {
    L2_RCWM_HELPER(64, 2048, 8, true);
  }
}

template <typename T, typename S, typename K>
void CalRandomChoiceWithMaskSmall(int input_size, int seedc, int count, K *input, S *output_index, K *output_mask,
                               cudaStream_t stream) {
  RCWMScaleK<T, S, K>(seedc, input_size, input, count, output_index, output_mask, stream);
}

template void CalRandomChoiceWithMaskSmall<float, int, bool>(int input_size, int seedc, int count, bool *input,
                                                          int *output_index, bool *output_mask, cudaStream_t stream);
