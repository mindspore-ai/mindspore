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

#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/no_repeat_ngram_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ void max_val_init(T *init_val) {
  *init_val = std::numeric_limits<T>::max();
}
// Handle fp16 differently for assignment
template <>
__device__ __forceinline__ void max_val_init(half *init_val) {
  *init_val = __int2half_rd(65504);  // Max value for Half
}

template <typename StateType, typename LogProbType>
__global__ void reassign_probability(StateType* __restrict__ tokens,
                                    LogProbType* __restrict__ lprobs,
                                    LogProbType* __restrict__ output,
                                    int batch_mul_beam_size,
                                    int vocab_size,
                                    int no_repeat_ngram_size,
                                     int total_blocks) {
  extern __shared__ int32_t shared_mem[];
  LogProbType pad_value = 0.0;
  max_val_init(&pad_value);
  for (size_t batch_index = blockIdx.x; batch_index < total_blocks; batch_index += gridDim.x) {
    // This requires the thread id
    auto position_id_in_one_batch = threadIdx.x;
    auto indexed_token = batch_index * batch_mul_beam_size + position_id_in_one_batch;
    auto last_ngram_tokens = blockDim.x - 1;
    auto lprob_start = batch_index * vocab_size;
    shared_mem[position_id_in_one_batch] = tokens[indexed_token];
    if (position_id_in_one_batch == last_ngram_tokens) {
      for (int i = 0; i < no_repeat_ngram_size; ++i) {
        if (position_id_in_one_batch + i < batch_mul_beam_size) {
          shared_mem[position_id_in_one_batch + i ] = tokens[indexed_token + i];
        }
      }
    }
    __syncthreads();
    bool should_modify = true;
    for (int i = 0; i < no_repeat_ngram_size - 1; ++i) {
      if (shared_mem[position_id_in_one_batch + i] != shared_mem[last_ngram_tokens + i + 1]) {
        should_modify = false;
        break;
      }
    }
    if (should_modify) {
      // reset probability
      auto id = shared_mem[position_id_in_one_batch + no_repeat_ngram_size - 1];
      output[lprob_start + id] = -pad_value;
    }
  }
}


template <typename StateType, typename LogProbType>
__global__ void reassign_probability_no_shared(StateType* __restrict__ tokens,
                                               LogProbType* __restrict__ lprobs,
                                               LogProbType* __restrict__ output,
                                               int batch_mul_beam_size,
                                               int vocab_size,
                                               int no_repeat_ngram_size,
                                               int total_blocks,
                                               int total_threads) {
  extern __shared__ int32_t shared_mem[];
  LogProbType pad_value = 0.0;
  max_val_init(&pad_value);
  for (size_t batch_index = blockIdx.x; batch_index < total_blocks; batch_index += gridDim.x) {
    for (size_t thread_index = threadIdx.x; thread_index < total_threads; thread_index += blockDim.x) {
      auto offsets = batch_index * batch_mul_beam_size;
      auto indexed_token = offsets + thread_index;
      auto last_ngram_tokens = offsets + total_threads - 1;
      auto lprob_start = batch_index * vocab_size;
      bool should_modify = true;
      for (int i = 0; i < no_repeat_ngram_size - 1; ++i) {
        if (tokens[indexed_token + i] != tokens[last_ngram_tokens + i + 1]) {
          should_modify = false;
          break;
        }
      }
      if (should_modify) {
        // reset probability
        auto id = tokens[indexed_token + no_repeat_ngram_size - 1];
        output[lprob_start + id] = -pad_value;
      }
    }
  }
}

template <typename StateType, typename LogProbType>
void CalculateNoRepeatNGram(const StateType *tokens,
                            LogProbType *lprobs,
                            LogProbType *output,
                            int batch_mul_beam_size,
                            int no_repeat_ngram_size,
                            const uint32_t &device_id,
                            int vocab_size,
                            int blocks,
                            int shared_mem_size,
                            cudaStream_t cuda_stream) {
  int threads = batch_mul_beam_size - no_repeat_ngram_size + 2 - 1;
  if (threads <= 0) return;
  auto cuda_threads = CUDA_THREADS(device_id);
  if (cuda_threads >= threads) {
      reassign_probability<<<CUDA_BLOCKS(device_id, blocks), threads, shared_mem_size, cuda_stream>>>(
              tokens, lprobs, output, batch_mul_beam_size, vocab_size, no_repeat_ngram_size, blocks);
  } else {
       reassign_probability_no_shared<<<CUDA_BLOCKS(device_id, blocks), cuda_threads, shared_mem_size, cuda_stream>>>(
            tokens, lprobs, output, batch_mul_beam_size, vocab_size, no_repeat_ngram_size, blocks, threads);
  }
}

template CUDA_LIB_EXPORT void CalculateNoRepeatNGram<int32_t, half>(const int32_t *tokens,
                                                                     half *lprobs,
                                                                     half *outpout,
                                                                     int step,
                                                                     int no_repeat_ngram_size,
                                                                     const uint32_t &device_id,
                                                                     int vocab_size_,
                                                                     int blocks,
                                                                     int shared_mem_size,
                                                                     cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateNoRepeatNGram<int32_t, float>(const int32_t *tokens,
                                                                     float *lprobs,
                                                                     float *output,
                                                                     int step,
                                                                     int no_repeat_ngram_size,
                                                                     const uint32_t &device_id,
                                                                     int vocab_size_,
                                                                     int blocks,
                                                                     int shared_mem_size,
                                                                     cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateNoRepeatNGram<int32_t, double>(const int32_t *tokens,
                                                                     double *lprobs,
                                                                     double *output,
                                                                     int step,
                                                                     int no_repeat_ngram_size,
                                                                     const uint32_t &device_id,
                                                                     int vocab_size_,
                                                                     int blocks,
                                                                     int shared_mem_size,
                                                                     cudaStream_t cuda_stream);
