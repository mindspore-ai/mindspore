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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPLVSL_UTILS_LAYER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPLVSL_UTILS_LAYER_H_

#include "tikcfw/kernel_operator.h"
#include "tikcfw/impl/kernel_utils.h"
#include "tikcfw/interface/kernel_operator_intf.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
using AscendC::AIV;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

namespace mindspore::ascend_native {
class KernelVsl {
 public:
  __aicore__ inline KernelVsl() = default;
  __aicore__ inline void CopyVSL(KernelVsl *vsl_helper, __gm__ KernelVsl *vsl_helper_gm) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(vsl_helper);
    auto vsl_helper_32 = reinterpret_cast<__gm__ uint32_t *>(vsl_helper_gm);

    for (int i = 0; i < sizeof(KernelVsl) / sizeof(uint32_t); i++, ptr++) {
      *ptr = *(vsl_helper_32 + i);
    }
    return;
  }
  __aicore__ inline void CopyPipe(TPipe *pipe, __gm__ TPipe *pipe_gm) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(pipe);
    auto pipe_32 = reinterpret_cast<__gm__ uint32_t *>(pipe_gm);

    for (int i = 0; i < sizeof(TPipe) / sizeof(uint32_t); i++, ptr++) {
      *ptr = *(pipe_32 + i);
    }
    return;
  }
  __aicore__ inline void Init(GM_ADDR q_seq_len, GM_ADDR kv_seq_len, GM_ADDR q_padding_offset,
                              GM_ADDR kv_padding_offset, GM_ADDR mode, int actual_token, int batch, int seq_len,
                              TPipe *pipe) {
    max_token_ = batch * seq_len;
    actual_token_ = actual_token;
    seq_len_ = seq_len;
    batch_ = batch;
    pipe_ = pipe;

    q_padding_offset_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(q_padding_offset), max_token_);
    kv_padding_offset_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(kv_padding_offset), max_token_);
    q_seq_len_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(q_seq_len), batch_);
    kv_seq_len_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(kv_seq_len), batch_);
    mode_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(mode), batch_);
    pipe_->InitBuffer(q_seq_len_local_ptr_, ALIGN32(batch_ * sizeof(int)));
    pipe_->InitBuffer(kv_seq_len_local_ptr_, ALIGN32(batch_ * sizeof(int)));
    pipe_->InitBuffer(mode_local_ptr_, ALIGN32(batch_ * sizeof(int)));

    q_seq_len_local_ = q_seq_len_local_ptr_.Get<int>();
    kv_seq_len_local_ = kv_seq_len_local_ptr_.Get<int>();
    mode_local_ = mode_local_ptr_.Get<int>();

    // copy need to be align to 32 bytes according to type (8 int's are 32 bytes)
    DataCopy(q_seq_len_local_, q_seq_len_global_, ALIGN(batch_, 8));
    DataCopy(kv_seq_len_local_, kv_seq_len_global_, ALIGN(batch_, 8));
    DataCopy(mode_local_, mode_global_, ALIGN(batch_, 8));
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void GetBatchId(int token_id, int *batch_id) {
    *batch_id = (q_padding_offset_global_.GetValue(token_id) + token_id) / seq_len_;
  }

  __aicore__ inline void GetSeqLen(int batch_id, int *act_seq_len) {
    *act_seq_len = q_seq_len_local_.GetValue(batch_id);
  }
  __aicore__ inline void GetKVSeqLen(int batch_id, int *act_seq_len) {
    *act_seq_len = kv_seq_len_local_.GetValue(batch_id);
  }

  __aicore__ inline void GetSeqId(int token_id, int *seq_id) {
    *seq_id = (q_padding_offset_global_.GetValue(token_id) + token_id) % seq_len_;
  }

  __aicore__ inline void GetKVSeqId(int token_id, int *seq_id) {
    *seq_id = (kv_padding_offset_global_.GetValue(token_id) + token_id) % seq_len_;
  }
  __aicore__ inline void GetTokenIdByBatch(int batch_id, int *token_id) {
    *token_id = 0;
    for (int i = 0; i < batch_id; i++) {
      *token_id += q_seq_len_local_.GetValue(i);
    }
  }
  __aicore__ inline void GetIncrementalMode(int batch_id, bool *incremental) {
    *incremental = (mode_local_.GetValue(batch_id) > 0);
  }
  __aicore__ inline void GetActualOffset(int batch_id, int *offset) {
    *offset = 0;
    for (int i = 0; i < batch_id; i++) {
      *offset += (mode_local_.GetValue(i) > 0) ? 1 : seq_len_;
    }
  }
  __aicore__ inline void GetActualBatchAndToken(int *batch_id, int *token_id, int current_elem) {
    for (int i = 0; i < batch_; i++) {
      *token_id = current_elem;
      *batch_id = i;
      current_elem -= (mode_local_.GetValue(i) > 0) ? 1 : seq_len_;
      if (current_elem < 0) break;
    }
  }

 private:
  int batch_, max_token_, actual_token_, seq_len_;
  TPipe *pipe_;

  GlobalTensor<int> q_seq_len_global_;
  GlobalTensor<int> kv_seq_len_global_;
  GlobalTensor<int> q_padding_offset_global_;
  GlobalTensor<int> kv_padding_offset_global_;
  GlobalTensor<int> mode_global_;

  LocalTensor<int> q_seq_len_local_;
  LocalTensor<int> kv_seq_len_local_;
  LocalTensor<int> mode_local_;

  TBuf<QuePosition::VECIN> q_seq_len_local_ptr_;
  TBuf<QuePosition::VECIN> kv_seq_len_local_ptr_;
  TBuf<QuePosition::VECIN> q_padding_offset_local_ptr_;
  TBuf<QuePosition::VECIN> kv_padding_offset_local_ptr_;
  TBuf<QuePosition::VECIN> mode_local_ptr_;
};

}  // namespace mindspore::ascend_native
#endif
