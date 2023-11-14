/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel_operator.h"
#include "kernel_utils.h"
using namespace AscendC;

namespace {
constexpr int64_t kBufferNum = 2;
constexpr int64_t kUbSize = 192 * 1024;
const int64_t kDivisor = 4;
static __aicore__ inline int64_t CeilRound(int64_t value, int64_t kDivisor) {
  if (kDivisor == 0) {
    return 0;
  }
  return (value + kDivisor - 1) / kDivisor * kDivisor;
}
}  // namespace

template <typename T>
class KernelPromptKvCache {
 public:
  __aicore__ inline KernelPromptKvCache() {}

  __aicore__ inline void GetBatchIndex(GM_ADDR batch_index, int64_t ub) {
    int64_t batch_index_ub_size = CeilRound(ub, kDivisor);
    batch_index_gm_.SetGlobalBuffer((__gm__ int64_t *)batch_index, batch_index_ub_size);
    pipe_.InitBuffer(batch_index_queue_, 1, batch_index_ub_size * sizeof(int64_t));
    batch_index_tensor_ = batch_index_queue_.AllocTensor<int64_t>();
    DataCopy(batch_index_tensor_, batch_index_gm_, batch_index_ub_size);
    remain_ub_size_ = kUbSize - batch_index_ub_size * sizeof(int64_t);
  }

  __aicore__ inline void GetNewMaxSeqLen(GM_ADDR new_max_seq_len) {
    new_max_seq_len_gm_.SetGlobalBuffer((__gm__ int64_t *)new_max_seq_len, 4);
    pipe_.InitBuffer(new_max_seq_len_queue_, 1, CeilRound(1, kDivisor) * sizeof(int64_t));
    LocalTensor<int64_t> new_max_seq_len_tensor = new_max_seq_len_queue_.AllocTensor<int64_t>();
    DataCopy(new_max_seq_len_tensor, new_max_seq_len_gm_, CeilRound(1, kDivisor));
    pipe_barrier((pipe_t)PIPE_ALL);
    s_ = new_max_seq_len_tensor.GetValue(0);
    new_max_seq_len_queue_.FreeTensor(new_max_seq_len_tensor);
  }

  __aicore__ inline void UpdateCache(GM_ADDR cache, GM_ADDR update) {
    int64_t split_us = 1;
    int64_t block_us = us_ / split_us;
    while (kBufferNum * block_us * d_ * sizeof(T) > remain_ub_size_) {
      split_us++;
      block_us = (us_ + split_us - 1) / split_us;
    }

    int64_t former_block_us = block_us;
    int64_t tail_block_us = us_ - (split_us - 1) * former_block_us;
    pipe_.InitBuffer(update_queue_, kBufferNum, block_us * d_ * sizeof(T));

    for (int64_t i = 0; i < each_core_bs_num_; ++i) {
      int64_t bs_idx = core_idx_ * former_each_core_bs_num_ + i;
      int64_t ub_idx = bs_idx / h_;
      int64_t h_idx = bs_idx % h_;
      int64_t cache_b_idx = batch_index_tensor_.GetValue(ub_idx);
      pipe_barrier((pipe_t)PIPE_ALL);
      if (cache_b_idx < 0 || cache_b_idx >= b_) {
        continue;
      }

      for (int64_t j = 0; j < split_us; ++j) {
        int64_t u_block_len;
        if (j == split_us - 1) {
          u_block_len = tail_block_us * d_;
        } else {
          u_block_len = former_block_us * d_;
        }
        LocalTensor<T> update_in_local_tensor = update_queue_.AllocTensor<T>();
        update_gm_.SetGlobalBuffer(
          (__gm__ T *)update + ub_idx * update_b_stride_ + h_idx * update_h_stride_ + j * former_block_us * d_,
          u_block_len);
        out_gm_.SetGlobalBuffer(
          (__gm__ T *)cache + cache_b_idx * cache_b_stride_ + h_idx * cache_h_stride_ + j * former_block_us * d_,
          u_block_len);
        pipe_barrier((pipe_t)PIPE_ALL);
        DataCopy(update_in_local_tensor, update_gm_, u_block_len);
        update_queue_.EnQue(update_in_local_tensor);
        LocalTensor<T> update_in_local_tensor_out = update_queue_.DeQue<T>();
        pipe_barrier((pipe_t)PIPE_ALL);
        DataCopy(out_gm_, update_in_local_tensor_out, u_block_len);
        update_queue_.FreeTensor(update_in_local_tensor_out);
      }
    }
  }

  __aicore__ inline void Process(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len, GM_ADDR batch_index,
                                 GM_ADDR seq_len_axis, GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len,
                                 int64_t max_core_num, int64_t b, int64_t h, int64_t s, int64_t d, int64_t ub,
                                 int64_t us) {
    core_idx_ = GetBlockIdx();
    int64_t bs = ub * h;
    former_each_core_bs_num_ = (bs + max_core_num - 1) / max_core_num;
    core_num_ = (bs + former_each_core_bs_num_ - 1) / former_each_core_bs_num_;
    tail_each_core_bs_num_ = bs - (core_num_ - 1) * former_each_core_bs_num_;

    if (core_idx_ >= core_num_) {
      return;
    }
    if (g_coreType == AIC) {
      return;
    }

    GetBatchIndex(batch_index, ub);
    GetNewMaxSeqLen(new_max_seq_len);

    b_ = b * s / s_;
    h_ = h;
    d_ = d;
    ub_ = ub;
    us_ = us;

    if (core_idx_ != core_num_ - 1) {
      each_core_bs_num_ = former_each_core_bs_num_;
    } else {
      each_core_bs_num_ = tail_each_core_bs_num_;
    }

    cache_h_stride_ = s_ * d_;
    cache_b_stride_ = h_ * cache_h_stride_;

    update_h_stride_ = us_ * d_;
    update_b_stride_ = h_ * update_h_stride_;

    UpdateCache(cache, update);
    batch_index_queue_.FreeTensor(batch_index_tensor_);
  }

 private:
  // gm
  GlobalTensor<T> update_gm_;
  GlobalTensor<int64_t> batch_index_gm_;
  GlobalTensor<int64_t> new_max_seq_len_gm_;
  GlobalTensor<T> out_gm_;

  // local gm
  LocalTensor<int64_t> batch_index_tensor_;

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> update_queue_;
  TQue<QuePosition::VECIN, 1> batch_index_queue_;
  TQue<QuePosition::VECIN, 1> new_max_seq_len_queue_;

  int64_t remain_ub_size_ = 0;
  int64_t core_idx_ = 0;
  int64_t core_num_ = 0;
  int64_t each_core_bs_num_ = 0;
  int64_t former_each_core_bs_num_ = 0;
  int64_t tail_each_core_bs_num_ = 0;
  int64_t b_ = 0;
  int64_t h_ = 0;
  int64_t s_ = 0;
  int64_t d_ = 0;
  int64_t ub_ = 0;
  int64_t us_ = 0;
  int64_t ps_ = 0;

  int64_t cache_b_stride_ = 0;
  int64_t cache_h_stride_ = 0;
  int64_t update_b_stride_ = 0;
  int64_t update_h_stride_ = 0;
};

extern "C" __global__ __aicore__ void prompt_kv_cache(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len,
                                                      GM_ADDR batch_index, GM_ADDR seq_len_axis,
                                                      GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len, GM_ADDR out,
                                                      GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);

  if (TILING_KEY_IS(1)) {
    KernelPromptKvCache<int8_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len,
               tiling_data.core_num, tiling_data.b, tiling_data.h, tiling_data.s, tiling_data.d, tiling_data.ub,
               tiling_data.us);
  } else if (TILING_KEY_IS(2)) {
    KernelPromptKvCache<int16_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len,
               tiling_data.core_num, tiling_data.b, tiling_data.h, tiling_data.s, tiling_data.d, tiling_data.ub,
               tiling_data.us);
  } else if (TILING_KEY_IS(4)) {
    KernelPromptKvCache<int32_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len,
               tiling_data.core_num, tiling_data.b, tiling_data.h, tiling_data.s, tiling_data.d, tiling_data.ub,
               tiling_data.us);
  }
}
