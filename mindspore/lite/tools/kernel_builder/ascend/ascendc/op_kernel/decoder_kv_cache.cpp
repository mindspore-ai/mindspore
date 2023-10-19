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
constexpr int32_t BUFFER_NUM = 1;
const int64_t divisor = 4;
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
  __aicore__ inline void Init(GM_ADDR cache, GM_ADDR update, GM_ADDR valid_seq_len, GM_ADDR batch_index,
                              GM_ADDR new_max_seq_len, GM_ADDR cur_max_seq_len, int64_t core_num, int64_t b, int64_t h,
                              int64_t d, int64_t us) {
    core_idx_ = GetBlockIdx();
    former_each_core_bs_num_ = (b + core_num - 1) / core_num;
    core_num_ = (b + former_each_core_bs_num_ - 1) / former_each_core_bs_num_;

    if (core_idx_ >= core_num_) {
      return;
    }

    tail_each_core_bs_num_ = b - (core_num_ - 1) * former_each_core_bs_num_;

    valid_seq_len_gm.SetGlobalBuffer((__gm__ int64_t *)valid_seq_len, b);
    new_max_seq_len_gm.SetGlobalBuffer((__gm__ int64_t *)new_max_seq_len, b);

    // new_max_seq_len_queue
    pipe.InitBuffer(new_max_seq_len_queue, 1, CeilRound(1, divisor) * sizeof(int64_t));
    LocalTensor<int64_t> new_max_seq_len_tensor = new_max_seq_len_queue.AllocTensor<int64_t>();
    DataCopy(new_max_seq_len_tensor, new_max_seq_len_gm, CeilRound(1, divisor));
    pipe_barrier((pipe_t)PIPE_ALL);
    s_ = new_max_seq_len_tensor.GetValue(0);
    new_max_seq_len_queue.FreeTensor(new_max_seq_len_tensor);

    b_ = b;
    h_ = h;
    d_ = d;
    us_ = us;

    update_former_block_length = former_each_core_bs_num_ * us * d;
    cache_former_block_length = former_each_core_bs_num_ * s_ * d;
    if (core_idx_ != core_num_ - 1) {
      update_block_length = update_former_block_length;
      cache_block_length = cache_former_block_length;
    } else {
      update_block_length = tail_each_core_bs_num_ * us * d;
      cache_block_length = tail_each_core_bs_num_ * s_ * d;
    }
    update_gm.SetGlobalBuffer((__gm__ T *)update + core_idx_ * update_former_block_length, update_block_length);
    out_gm.SetGlobalBuffer((__gm__ T *)cache + core_idx_ * cache_former_block_length, cache_block_length);

    pipe.InitBuffer(update_queue, BUFFER_NUM, update_block_length * sizeof(T));
    pipe.InitBuffer(valid_seq_len_queue, 1, CeilRound(b, divisor) * sizeof(int64_t));
  }

  __aicore__ inline void Process() {
    if (g_coreType == AIC) {
      return;
    }
    if (core_idx_ >= core_num_) {
      return;
    }
    CopyIn();
    CopyOut();
  }

  __aicore__ inline void CopyIn() {
    LocalTensor<T> update_in_local_tensor = update_queue.AllocTensor<T>();
    DataCopy(update_in_local_tensor, update_gm, update_block_length);
    update_queue.EnQue(update_in_local_tensor);
    LocalTensor<int64_t> index_local_tensor = valid_seq_len_queue.AllocTensor<int64_t>();
    DataCopy(index_local_tensor, valid_seq_len_gm, CeilRound(b_, divisor));
    valid_seq_len_queue.EnQue(index_local_tensor);
  }

  __aicore__ inline void CopyOut() {
    LocalTensor<T> update_in_local_tensor = update_queue.DeQue<T>();
    LocalTensor<int64_t> index_local_tensor = valid_seq_len_queue.DeQue<int64_t>();
    size_t dst_bs_stride = s_ * d_;
    size_t src_bs_stride = us_ * d_;

    uint32_t each_core_bs_num = 0;
    if (core_idx_ != core_num_ - 1) {
      each_core_bs_num = former_each_core_bs_num_;
    } else {
      each_core_bs_num = tail_each_core_bs_num_;
    }

    for (size_t each_core_bs_idx = 0; each_core_bs_idx < each_core_bs_num; ++each_core_bs_idx) {
      auto bs_idx = core_idx_ * former_each_core_bs_num_ + each_core_bs_idx;
      auto real_b = bs_idx / h_;
      pipe_barrier((pipe_t)PIPE_ALL);
      auto index = index_local_tensor.GetValue(real_b);
      pipe_barrier((pipe_t)PIPE_ALL);
      if (index < 0 || index >= s_) {
        continue;
      }
      size_t dst_offset = each_core_bs_idx * dst_bs_stride + index * d_;
      size_t src_offset = each_core_bs_idx * src_bs_stride;
      DataCopy(out_gm[dst_offset], update_in_local_tensor[src_offset], d_);
    }

    update_queue.FreeTensor(update_in_local_tensor);
    valid_seq_len_queue.FreeTensor(index_local_tensor);
  }

 private:
  // gm
  GlobalTensor<T> cache_gm;
  GlobalTensor<T> update_gm;
  GlobalTensor<int64_t> valid_seq_len_gm;
  GlobalTensor<int64_t> new_max_seq_len_gm;
  GlobalTensor<T> out_gm;

  TPipe pipe;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> update_queue;
  TQue<QuePosition::VECIN, BUFFER_NUM> valid_seq_len_queue;
  TQue<QuePosition::VECIN, BUFFER_NUM> new_max_seq_len_queue;

  int64_t core_idx_ = 0;
  int64_t cache_former_block_length = 0;
  int64_t cache_block_length = 0;
  int64_t update_former_block_length = 0;
  int64_t update_block_length = 0;

  int64_t core_num_ = 0;
  int64_t former_each_core_bs_num_ = 0;
  int64_t tail_each_core_bs_num_ = 0;
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
    op.Init(cache, update, valid_seq_len, batch_index, new_max_seq_len, cur_max_seq_len, tiling_data.core_num,
            tiling_data.b, tiling_data.h, tiling_data.d, tiling_data.us);
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    KernelDecoderKvCache<int16_t> op;
    op.Init(cache, update, valid_seq_len, batch_index, new_max_seq_len, cur_max_seq_len, tiling_data.core_num,
            tiling_data.b, tiling_data.h, tiling_data.d, tiling_data.us);
    op.Process();
  } else if (TILING_KEY_IS(4)) {
    KernelDecoderKvCache<int32_t> op;
    op.Init(cache, update, valid_seq_len, batch_index, new_max_seq_len, cur_max_seq_len, tiling_data.core_num,
            tiling_data.b, tiling_data.h, tiling_data.d, tiling_data.us);
    op.Process();
  }
}
