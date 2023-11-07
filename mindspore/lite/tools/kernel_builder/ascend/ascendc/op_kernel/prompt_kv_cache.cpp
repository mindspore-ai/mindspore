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
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t UB_SIZE = 192 * 1024;
const int64_t divisor = 4;
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

  __aicore__ inline void GetBatchIndex(GM_ADDR batch_index) {
    // get batch index
    batch_index_gm.SetGlobalBuffer((__gm__ int64_t *)batch_index, 4);
    pipe.InitBuffer(batch_index_queue, 1, CeilRound(1, divisor) * sizeof(int64_t));
    LocalTensor<int64_t> batch_index_tensor = batch_index_queue.AllocTensor<int64_t>();
    DataCopy(batch_index_tensor, batch_index_gm, CeilRound(1, divisor));
    pipe_barrier((pipe_t)PIPE_ALL);
    batch_index_ = batch_index_tensor.GetValue(0);
    batch_index_queue.FreeTensor(batch_index_tensor);
  }

  __aicore__ inline void GetNewMaxSeqLen(GM_ADDR new_max_seq_len) {
    // get new_max_seq_len_queue
    new_max_seq_len_gm.SetGlobalBuffer((__gm__ int64_t *)new_max_seq_len, 4);
    pipe.InitBuffer(new_max_seq_len_queue, 1, CeilRound(1, divisor) * sizeof(int64_t));
    LocalTensor<int64_t> new_max_seq_len_tensor = new_max_seq_len_queue.AllocTensor<int64_t>();
    DataCopy(new_max_seq_len_tensor, new_max_seq_len_gm, CeilRound(1, divisor));
    pipe_barrier((pipe_t)PIPE_ALL);
    s_ = new_max_seq_len_tensor.GetValue(0);
    new_max_seq_len_queue.FreeTensor(new_max_seq_len_tensor);
  }

  __aicore__ inline void Process(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len, GM_ADDR batch_index,
                                 GM_ADDR seq_len_axis, GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len,
                                 int64_t core_num, int64_t b, int64_t h, int64_t d, int64_t us) {
    core_idx_ = GetBlockIdx();
    former_each_core_h_num_ = (h + core_num - 1) / core_num;
    core_num_ = (h + former_each_core_h_num_ - 1) / former_each_core_h_num_;
    tail_each_core_h_num_ = h - (core_num_ - 1) * former_each_core_h_num_;

    if (core_idx_ >= core_num_) {
      return;
    }
    if (g_coreType == AIC) {
      return;
    }

    GetBatchIndex(batch_index);
    GetNewMaxSeqLen(new_max_seq_len);

    b_ = b;
    h_ = h;
    d_ = d;
    us_ = us;
    if (core_idx_ != core_num_ - 1) {
      each_core_bs_num = former_each_core_h_num_;
    } else {
      each_core_bs_num = tail_each_core_h_num_;
    }
    update_block_length = us_ * d_;
    cache_block_length = s_ * d_;
    int64_t split_us = 1;
    int64_t block_us = us_ / split_us;
    while (BUFFER_NUM * block_us * d_ * sizeof(T) >= UB_SIZE) {
      split_us++;
      block_us = (us_ + split_us - 1) / split_us;
    }

    int64_t former_block_us = block_us;
    int64_t tail_block_us = us - (split_us - 1) * former_block_us;
    pipe.InitBuffer(update_queue, BUFFER_NUM, former_block_us * d_ * sizeof(T));
    int64_t u_block_len = former_block_us * d_;
    for (int64_t i = 0; i < each_core_bs_num; ++i) {
      int64_t h_idx = core_idx_ * former_each_core_h_num_ + i;
      for (int64_t j = 0; j < split_us; ++j) {
        if (j == split_us - 1) {
          u_block_len = tail_block_us * d_;
        }
        LocalTensor<T> update_in_local_tensor = update_queue.AllocTensor<T>();
        update_gm.SetGlobalBuffer((__gm__ T *)update + h_idx * update_block_length + j * former_block_us * d_,
                                  u_block_len);
        out_gm.SetGlobalBuffer(
          (__gm__ T *)cache + batch_index_ * h_ * s_ * d_ + h_idx * cache_block_length + j * former_block_us * d_,
          u_block_len);
        DataCopy(update_in_local_tensor, update_gm, u_block_len);
        update_queue.EnQue(update_in_local_tensor);
        LocalTensor<T> update_in_local_tensor_out = update_queue.DeQue<T>();
        pipe_barrier((pipe_t)PIPE_ALL);
        DataCopy(out_gm, update_in_local_tensor_out, u_block_len);
        update_queue.FreeTensor(update_in_local_tensor_out);
      }
    }
  }

 private:
  // gm
  GlobalTensor<T> cache_gm;
  GlobalTensor<T> update_gm;
  GlobalTensor<int64_t> batch_index_gm;
  GlobalTensor<int64_t> new_max_seq_len_gm;
  GlobalTensor<T> out_gm;
  GlobalTensor<T> padding_gm;

  TPipe pipe;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, 1> update_queue;
  TQue<QuePosition::VECIN, 1> batch_index_queue;
  TQue<QuePosition::VECIN, 1> new_max_seq_len_queue;
  TQue<QuePosition::VECOUT, 1> padding_queue;

  int64_t core_idx_ = 0;
  int64_t cache_former_block_length = 0;
  int64_t cache_block_length = 0;
  int64_t update_former_block_length = 0;
  int64_t update_block_length = 0;

  int64_t core_num_ = 0;
  int64_t each_core_bs_num = 0;
  int64_t former_each_core_h_num_ = 0;
  int64_t tail_each_core_h_num_ = 0;
  int64_t b_ = 0;
  int64_t h_ = 0;
  int64_t s_ = 0;
  int64_t batch_index_ = 0;
  int64_t d_ = 0;
  int64_t us_ = 0;
};

extern "C" __global__ __aicore__ void prompt_kv_cache(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len,
                                                      GM_ADDR batch_index, GM_ADDR seq_len_axis,
                                                      GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len, GM_ADDR out,
                                                      GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);

  if (TILING_KEY_IS(1)) {
    KernelPromptKvCache<int8_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len,
               tiling_data.core_num, tiling_data.b, tiling_data.h, tiling_data.d, tiling_data.us);
  } else if (TILING_KEY_IS(2)) {
    KernelPromptKvCache<int16_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len,
               tiling_data.core_num, tiling_data.b, tiling_data.h, tiling_data.d, tiling_data.us);
  } else if (TILING_KEY_IS(4)) {
    KernelPromptKvCache<int32_t> op;
    op.Process(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len,
               tiling_data.core_num, tiling_data.b, tiling_data.h, tiling_data.d, tiling_data.us);
  }
}
