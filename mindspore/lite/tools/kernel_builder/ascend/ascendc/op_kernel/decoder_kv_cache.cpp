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
using namespace AscendC;
namespace {
constexpr int32_t kBufferNum = 2;
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
class KernelDecoderKvCache {
 public:
  __aicore__ inline KernelDecoderKvCache() {}

  __aicore__ inline void GetNewMaxSeqLen(GM_ADDR new_max_seq_len) {
    new_max_seq_len_gm_.SetGlobalBuffer((__gm__ int64_t *)new_max_seq_len, 4);
    pipe_.InitBuffer(new_max_seq_len_queue_, 1, CeilRound(1, kDivisor) * sizeof(int64_t));
    LocalTensor<int64_t> new_max_seq_len_tensor = new_max_seq_len_queue_.AllocTensor<int64_t>();
    pipe_barrier((pipe_t)PIPE_ALL);
    DataCopy(new_max_seq_len_tensor, new_max_seq_len_gm_, CeilRound(1, kDivisor));
    pipe_barrier((pipe_t)PIPE_ALL);
    s_ = new_max_seq_len_tensor.GetValue(0);
    new_max_seq_len_queue_.FreeTensor(new_max_seq_len_tensor);
  }

  __aicore__ inline void GetValidSeqLen(GM_ADDR valid_seq_len, int64_t ub) {
    int64_t valid_seq_len_ub_size = CeilRound(ub, kDivisor);
    valid_seq_len_gm_.SetGlobalBuffer((__gm__ int64_t *)valid_seq_len, valid_seq_len_ub_size);
    pipe_.InitBuffer(valid_seq_len_queue_, 1, valid_seq_len_ub_size * sizeof(int64_t));
    valid_seq_len_tensor_ = valid_seq_len_queue_.AllocTensor<int64_t>();
    pipe_barrier((pipe_t)PIPE_ALL);
    DataCopy(valid_seq_len_tensor_, valid_seq_len_gm_, valid_seq_len_ub_size);
    pipe_barrier((pipe_t)PIPE_ALL);
    remain_ub_size_ -= valid_seq_len_ub_size * sizeof(int64_t);
  }

  __aicore__ inline void SplitBh(int64_t bh) {
    split_bh_ = 1;
    former_block_bh_ = bh;
    while (kBufferNum * former_block_bh_ * us_ * d_ * sizeof(T) >= remain_ub_size_) {
      split_bh_++;
      former_block_bh_ = (bh + split_bh_ - 1) / split_bh_;
    }
    tail_block_bh_ = bh - (split_bh_ - 1) * former_block_bh_;
  }

  __aicore__ inline void Update(GM_ADDR cache, GM_ADDR update, LocalTensor<T> update_in_local_tensor) {
    for (int64_t i = 0; i < split_bh_; i++) {
      int64_t block_bh;
      if (i != split_bh_ - 1) {
        block_bh = former_block_bh_;
      } else {
        block_bh = tail_block_bh_;
      }
      update_gm_.SetGlobalBuffer((__gm__ T *)update + core_idx_ * update_core_stride_ + i * former_block_bh_ * us_ * d_,
                                 block_bh * us_ * d_);

      DataCopy(update_in_local_tensor, update_gm_, block_bh * us_ * d_);
      pipe_barrier((pipe_t)PIPE_ALL);

      update_queue_.EnQue(update_in_local_tensor);
      LocalTensor<T> update_in_local_tensor_out = update_queue_.DeQue<T>();

      for (int64_t j = 0; j < block_bh; j++) {
        int64_t bh_idx = core_idx_ * former_bh_ + i * former_block_bh_ + j;
        auto b_idx = bh_idx / h_;
        pipe_barrier((pipe_t)PIPE_ALL);
        auto s_idx = valid_seq_len_tensor_.GetValue(b_idx);
        pipe_barrier((pipe_t)PIPE_ALL);
        if (s_idx < 0 || s_idx >= s_) {
          continue;
        }
        out_gm_.SetGlobalBuffer((__gm__ T *)cache + bh_idx * s_ * d_ + s_idx * d_, us_ * d_);
        int64_t src_offset = j * us_ * d_;
        pipe_barrier((pipe_t)PIPE_ALL);
        DataCopy(out_gm_, update_in_local_tensor_out[src_offset], us_ * d_);
      }
    }
  }

  __aicore__ inline void Process(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len, GM_ADDR batch_index,
                                 GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len, int64_t core_num, int64_t b,
                                 int64_t h, int64_t d, int64_t us) {
    core_idx_ = GetBlockIdx();
    former_bh_ = (b * h + core_num - 1) / core_num;
    core_num_ = (b * h + former_bh_ - 1) / former_bh_;
    if (core_idx_ >= core_num_) {
      return;
    }
    tail_bh_ = b * h - (core_num_ - 1) * former_bh_;

    b_ = b;
    h_ = h;
    d_ = d;
    us_ = us;

    GetNewMaxSeqLen(new_max_seq_len);
    GetValidSeqLen(valid_seq_len, b);

    update_core_stride_ = former_bh_ * us_ * d_;
    cache_core_stride_ = former_bh_ * s_ * d_;

    if (core_idx_ != core_num - 1) {
      SplitBh(former_bh_);
    } else {
      SplitBh(tail_bh_);
    }

    pipe_.InitBuffer(update_queue_, kBufferNum, former_block_bh_ * us_ * d_ * sizeof(T));
    LocalTensor<T> update_in_local_tensor = update_queue_.AllocTensor<T>();

    Update(cache, update, update_in_local_tensor);

    valid_seq_len_queue_.FreeTensor(valid_seq_len_tensor_);
    update_queue_.FreeTensor(update_in_local_tensor);
  }

 private:
  // gm
  GlobalTensor<T> update_gm_;
  GlobalTensor<int64_t> valid_seq_len_gm_;
  GlobalTensor<int64_t> new_max_seq_len_gm_;
  GlobalTensor<T> out_gm_;

  // local
  LocalTensor<int64_t> valid_seq_len_tensor_;

  TPipe pipe_;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, 1> update_queue_;
  TQue<QuePosition::VECIN, 1> valid_seq_len_queue_;
  TQue<QuePosition::VECIN, 1> new_max_seq_len_queue_;

  int64_t remain_ub_size_ = kUbSize;

  int64_t split_bh_ = 0;
  int64_t former_block_bh_ = 0;
  int64_t tail_block_bh_ = 0;

  int64_t core_idx_ = 0;
  int64_t cache_core_stride_ = 0;
  int64_t cache_block_length = 0;
  int64_t update_core_stride_ = 0;
  int64_t update_block_length_ = 0;

  int64_t core_num_ = 0;
  int64_t former_bh_ = 0;
  int64_t tail_bh_ = 0;
  int64_t b_ = 0;
  int64_t h_ = 0;
  int64_t s_ = 0;
  int64_t d_ = 0;
  int64_t us_ = 0;
};

extern "C" __global__ __aicore__ void decoder_kv_cache(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len,
                                                       GM_ADDR batch_index, GM_ADDR seq_len_axis,
                                                       GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len, GM_ADDR out,
                                                       GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);

  if (TILING_KEY_IS(1)) {
    KernelDecoderKvCache<int8_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, new_max_seq_len, cur_max_seq_len, tiling_data.core_num,
               tiling_data.b, tiling_data.h, tiling_data.d, tiling_data.us);
  } else if (TILING_KEY_IS(2)) {
    KernelDecoderKvCache<int16_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, new_max_seq_len, cur_max_seq_len, tiling_data.core_num,
               tiling_data.b, tiling_data.h, tiling_data.d, tiling_data.us);
  } else if (TILING_KEY_IS(4)) {
    KernelDecoderKvCache<int32_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, new_max_seq_len, cur_max_seq_len, tiling_data.core_num,
               tiling_data.b, tiling_data.h, tiling_data.d, tiling_data.us);
  }
}
