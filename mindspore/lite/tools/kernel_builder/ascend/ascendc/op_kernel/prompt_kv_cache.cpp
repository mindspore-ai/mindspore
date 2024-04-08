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
constexpr int64_t kBufferNum = 1;
const int64_t kDivisor = 4;
static __aicore__ inline int64_t CeilRound(int64_t value, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return (value + divisor - 1) / divisor * divisor;
}
}  // namespace

template <typename T>
class KernelPromptKvCache {
 public:
  __aicore__ inline KernelPromptKvCache() {}

  __aicore__ inline void GetIndex(GM_ADDR batch_index, GM_ADDR valid_seq_len) {
    int64_t batch_index_ub_num = CeilRound(ub_, kDivisor);
    int64_t valid_seq_len_ub_num = CeilRound(ub_, kDivisor);
    batch_index_gm_.SetGlobalBuffer((__gm__ int64_t *)batch_index, batch_index_ub_num);

    int64_t total_num = batch_index_ub_num + valid_seq_len_ub_num;
    pipe_.InitBuffer(index_queue_, 1, total_num * sizeof(int64_t));
    batch_index_tensor_ = index_queue_.AllocTensor<int64_t>();
    DataCopy(batch_index_tensor_, batch_index_gm_, batch_index_ub_num);

    valid_seq_len_gm_.SetGlobalBuffer((__gm__ int64_t *)valid_seq_len, valid_seq_len_ub_num);
    valid_seq_len_tensor_ = batch_index_tensor_[batch_index_ub_num];
    DataCopy(valid_seq_len_tensor_, valid_seq_len_gm_, valid_seq_len_ub_num);
  }

  __aicore__ inline void UpdateCache(GM_ADDR cache, GM_ADDR update) {
    pipe_.InitBuffer(update_queue_, kBufferNum, former_block_us_ * d_ * sizeof(T));
    for (int64_t i = 0; i < each_core_bs_num_; ++i) {
      int64_t bh_idx = core_idx_ * former_each_core_bs_num_ + i;
      int64_t ub_idx = bh_idx / h_;
      int64_t h_idx = bh_idx % h_;
      pipe_barrier((pipe_t)PIPE_ALL);
      int64_t cache_b_idx = batch_index_tensor_.GetValue(ub_idx);
      int64_t s_idx = valid_seq_len_tensor_.GetValue(ub_idx);
      if (cache_b_idx < 0 || cache_b_idx >= b_) {
        continue;
      }
      if (s_idx < 0 || s_idx + us_ > s_) {
        continue;
      }

      for (int64_t j = 0; j < split_us_; ++j) {
        int64_t u_block_len;
        if (j == split_us_ - 1) {
          u_block_len = tail_block_us_ * d_;
        } else {
          u_block_len = former_block_us_ * d_;
        }
        LocalTensor<T> update_in_local_tensor = update_queue_.AllocTensor<T>();
        update_gm_.SetGlobalBuffer(
          (__gm__ T *)update + ub_idx * update_b_stride_ + h_idx * update_h_stride_ + j * former_block_us_ * d_,
          u_block_len);
        out_gm_.SetGlobalBuffer((__gm__ T *)cache + cache_b_idx * cache_b_stride_ + h_idx * cache_h_stride_ +
                                  s_idx * d_ + j * former_block_us_ * d_,
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

  __aicore__ inline void InitParam(GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    core_num_ = tiling_data.core_num;
    b_ = tiling_data.b;
    h_ = tiling_data.h;
    s_ = tiling_data.s;
    d_ = tiling_data.d;
    ub_ = tiling_data.ub;
    us_ = tiling_data.us;
    former_each_core_bs_num_ = tiling_data.former_each_core_bs_num;
    tail_each_core_bs_num_ = tiling_data.tail_each_core_bs_num;
    split_us_ = tiling_data.split_us;
    former_block_us_ = tiling_data.former_block_us;
    tail_block_us_ = tiling_data.tail_block_us;
  }

  __aicore__ inline void Process(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len, GM_ADDR batch_index,
                                 GM_ADDR seq_len_axis, GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len,
                                 GM_ADDR tiling_data) {
    core_idx_ = GetBlockIdx();
    InitParam(tiling_data);
    if (core_idx_ >= core_num_) {
      return;
    }
    if (g_coreType == AIC) {
      return;
    }
    if (core_idx_ != core_num_ - 1) {
      each_core_bs_num_ = former_each_core_bs_num_;
    } else {
      each_core_bs_num_ = tail_each_core_bs_num_;
    }

    GetIndex(batch_index, valid_seq_len);

    cache_h_stride_ = s_ * d_;
    cache_b_stride_ = h_ * cache_h_stride_;

    update_h_stride_ = us_ * d_;
    update_b_stride_ = h_ * update_h_stride_;

    UpdateCache(cache, update);
    index_queue_.FreeTensor(batch_index_tensor_);
  }

 private:
  // gm
  GlobalTensor<T> update_gm_;
  GlobalTensor<int64_t> valid_seq_len_gm_;
  GlobalTensor<int64_t> batch_index_gm_;
  GlobalTensor<int64_t> new_max_seq_len_gm_;
  GlobalTensor<T> out_gm_;

  // local gm
  LocalTensor<int64_t> valid_seq_len_tensor_;
  LocalTensor<int64_t> batch_index_tensor_;

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> update_queue_;
  TQue<QuePosition::VECIN, 1> index_queue_;
  TQue<QuePosition::VECIN, 1> new_max_seq_len_queue_;

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
  int64_t split_us_ = 0;
  int64_t former_block_us_ = 0;
  int64_t tail_block_us_ = 0;

  int64_t cache_b_stride_ = 0;
  int64_t cache_h_stride_ = 0;
  int64_t update_b_stride_ = 0;
  int64_t update_h_stride_ = 0;
};

template <typename T>
class KernelPromptKvCacheCopyAll {
 public:
  __aicore__ inline KernelPromptKvCacheCopyAll() {}

  __aicore__ inline void InitParam(GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    core_num_ = tiling_data.core_num;
    former_each_core_bs_num_ = tiling_data.former_each_core_bs_num;
    tail_each_core_bs_num_ = tiling_data.tail_each_core_bs_num;
    split_us_ = tiling_data.split_us;
    former_block_us_ = tiling_data.former_block_us;
    tail_block_us_ = tiling_data.tail_block_us;
  }

  __aicore__ inline void Process(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len, GM_ADDR batch_index,
                                 GM_ADDR seq_len_axis, GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len,
                                 GM_ADDR tiling_data) {
    core_idx_ = GetBlockIdx();
    InitParam(tiling_data);
    if (core_idx_ >= core_num_) {
      return;
    }
    if (g_coreType == AIC) {
      return;
    }
    if (core_idx_ != core_num_ - 1) {
      each_core_bs_num_ = former_each_core_bs_num_;
    } else {
      each_core_bs_num_ = tail_each_core_bs_num_;
    }
    pipe_barrier((pipe_t)PIPE_ALL);

    pipe_.InitBuffer(update_queue_, kBufferNum, former_block_us_ * sizeof(T));
    for (int64_t i = 0; i < split_us_; ++i) {
      int64_t u_block_len;
      if (i == split_us_ - 1) {
        u_block_len = tail_block_us_;
      } else {
        u_block_len = former_block_us_;
      }
      LocalTensor<T> update_in_local_tensor = update_queue_.AllocTensor<T>();
      update_gm_.SetGlobalBuffer((__gm__ T *)update + core_idx_ * former_each_core_bs_num_ + i * former_block_us_,
                                 u_block_len);
      out_gm_.SetGlobalBuffer((__gm__ T *)cache + core_idx_ * former_each_core_bs_num_ + i * former_block_us_,
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

 private:
  // gm
  GlobalTensor<T> update_gm_;
  GlobalTensor<T> out_gm_;

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> update_queue_;

  int64_t core_idx_ = 0;
  int64_t core_num_ = 0;
  int64_t each_core_bs_num_ = 0;
  int64_t former_each_core_bs_num_ = 0;
  int64_t tail_each_core_bs_num_ = 0;

  int64_t split_us_ = 0;
  int64_t former_block_us_ = 0;
  int64_t tail_block_us_ = 0;
};

extern "C" __global__ __aicore__ void prompt_kv_cache(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len,
                                                      GM_ADDR batch_index, GM_ADDR seq_len_axis,
                                                      GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len, GM_ADDR out,
                                                      GM_ADDR workspace, GM_ADDR tiling) {
  if (TILING_KEY_IS(1)) {
    KernelPromptKvCache<int8_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len, tiling);
  } else if (TILING_KEY_IS(2)) {
    KernelPromptKvCache<int16_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len, tiling);
  } else if (TILING_KEY_IS(4)) {
    KernelPromptKvCache<int32_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len, tiling);
  } else if (TILING_KEY_IS(10)) {
    KernelPromptKvCacheCopyAll<int8_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len, tiling);
  } else if (TILING_KEY_IS(20)) {
    KernelPromptKvCacheCopyAll<int16_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len, tiling);
  } else if (TILING_KEY_IS(40)) {
    KernelPromptKvCacheCopyAll<int32_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len, tiling);
  }
}
