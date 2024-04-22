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
const int64_t kDivisor = 4;
static __aicore__ inline int64_t CeilRound(int64_t value, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return (value + divisor - 1) / divisor * divisor;
}
}  // namespace

template <typename T>
class KernelDecoderKvCache {
 public:
  __aicore__ inline KernelDecoderKvCache() {}

  __aicore__ inline void GetValidSeqLen(GM_ADDR valid_seq_len) {
    int64_t valid_seq_len_ub_size = CeilRound(b_, kDivisor);
    valid_seq_len_gm_.SetGlobalBuffer((__gm__ int64_t *)valid_seq_len, valid_seq_len_ub_size);
    pipe_.InitBuffer(valid_seq_len_queue_, 1, valid_seq_len_ub_size * sizeof(int64_t));
    valid_seq_len_tensor_ = valid_seq_len_queue_.AllocTensor<int64_t>();
    pipe_barrier((pipe_t)PIPE_ALL);
    DataCopy(valid_seq_len_tensor_, valid_seq_len_gm_, valid_seq_len_ub_size);
    pipe_barrier((pipe_t)PIPE_ALL);
  }

  __aicore__ inline void SplitBh() {
    if (core_idx_ != core_num_ - 1) {
      split_bh_ = f_split_bh_;
      former_block_bh_ = f_f_bh_;
      tail_block_bh_ = f_t_bh_;
    } else {
      split_bh_ = t_split_bh_;
      former_block_bh_ = t_f_bh_;
      tail_block_bh_ = t_t_bh_;
    }
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
        if (s_idx < 0) {
          continue;
        }
        if (s_idx >= s_) {
          s_idx = s_idx % s_;
        }
        out_gm_.SetGlobalBuffer((__gm__ T *)cache + bh_idx * s_ * d_ + s_idx * d_, us_ * d_);
        int64_t src_offset = j * us_ * d_;
        pipe_barrier((pipe_t)PIPE_ALL);
        DataCopy(out_gm_, update_in_local_tensor_out[src_offset], us_ * d_);
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
    us_ = tiling_data.us;
    former_bh_ = tiling_data.former_bh;
    tail_bh_ = tiling_data.tail_bh;
    f_split_bh_ = tiling_data.f_split_bh;
    f_f_bh_ = tiling_data.f_f_bh;
    f_t_bh_ = tiling_data.f_t_bh;
    t_split_bh_ = tiling_data.t_split_bh;
    t_f_bh_ = tiling_data.t_f_bh;
    t_t_bh_ = tiling_data.t_t_bh;
  }

  __aicore__ inline void Process(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len, GM_ADDR batch_index,
                                 GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len, GM_ADDR tiling) {
    InitParam(tiling);
    core_idx_ = GetBlockIdx();
    if (core_idx_ >= core_num_) {
      return;
    }

    GetValidSeqLen(valid_seq_len);
    update_core_stride_ = former_bh_ * us_ * d_;
    cache_core_stride_ = former_bh_ * s_ * d_;

    SplitBh();

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

  int64_t split_bh_ = 0;
  int64_t former_block_bh_ = 0;
  int64_t tail_block_bh_ = 0;

  int64_t core_idx_ = 0;
  int64_t cache_core_stride_ = 0;
  int64_t cache_block_length = 0;
  int64_t update_core_stride_ = 0;
  int64_t update_block_length_ = 0;

  int64_t core_num_ = 0;
  int64_t b_ = 0;
  int64_t h_ = 0;
  int64_t s_ = 0;
  int64_t d_ = 0;
  int64_t us_ = 0;

  int64_t former_bh_ = 0;
  int64_t tail_bh_ = 0;
  int64_t f_split_bh_ = 0;
  int64_t f_f_bh_ = 0;
  int64_t f_t_bh_ = 0;
  int64_t t_split_bh_ = 0;
  int64_t t_f_bh_ = 0;
  int64_t t_t_bh_ = 0;
};

extern "C" __global__ __aicore__ void decoder_kv_cache(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len,
                                                       GM_ADDR batch_index, GM_ADDR seq_len_axis,
                                                       GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len, GM_ADDR out,
                                                       GM_ADDR workspace, GM_ADDR tiling) {
  if (TILING_KEY_IS(1)) {
    KernelDecoderKvCache<int8_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, new_max_seq_len, cur_max_seq_len, tiling);
  } else if (TILING_KEY_IS(2)) {
    KernelDecoderKvCache<int16_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, new_max_seq_len, cur_max_seq_len, tiling);
  } else if (TILING_KEY_IS(4)) {
    KernelDecoderKvCache<int32_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, new_max_seq_len, cur_max_seq_len, tiling);
  }
}
