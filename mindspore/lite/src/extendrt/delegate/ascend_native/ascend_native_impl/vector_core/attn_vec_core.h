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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_ATTN_VEC_CORE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_ATTN_VEC_CORE_H_

#include "tikcfw/kernel_operator.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/param.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/vsl_utils.h"

using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

namespace mindspore::ascend_native {
template <typename T, int PipeSize>
class KernelPermuteQKV {
 public:
  __aicore__ inline KernelPermuteQKV() = default;
  __aicore__ inline void Init(GM_ADDR qkv_ptr, GM_ADDR bias_ptr, GM_ADDR q_ptr, GM_ADDR k_cache_ptr,
                              GM_ADDR v_cache_ptr, GM_ADDR q_seq_len, GM_ADDR kv_seq_len, GM_ADDR q_padding_offset,
                              GM_ADDR kv_padding_offset, GM_ADDR mode, int actual_token, int batch, int seq_len,
                              int head_num, int head_size, int total_blocks, int per_core_blocks) {
    // initialize VSL
    vsl_helper_.Init(q_seq_len, kv_seq_len, q_padding_offset, kv_padding_offset, mode, actual_token, batch, seq_len,
                     &pipe_);
    int D = head_num * head_size;
    head_size_ = head_size;
    head_num_ = head_num;
    max_seq_len_ = seq_len;
    batch_ = batch;
    actual_token_ = actual_token;
    total_blocks_ = total_blocks;
    per_core_blocks_ = per_core_blocks;
    constexpr int EmbeddingFactor = 3;
    qkv_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(qkv_ptr), actual_token_ * D * EmbeddingFactor);
    q_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q_ptr), batch_ * max_seq_len_ * D);
    k_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k_cache_ptr), batch_ * max_seq_len_ * D);
    v_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v_cache_ptr), batch_ * max_seq_len_ * D);
    bias_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias_ptr), EmbeddingFactor * D);
    int head_align_size = ALIGN32(sizeof(T) * D);
    copy_size_ = head_align_size / sizeof(T);
    pipe_.InitBuffer(in_, PipeSize, head_align_size);
    pipe_.InitBuffer(out_, PipeSize, head_align_size);
    pipe_.InitBuffer(bias_tmp_buf_, ALIGN32(sizeof(T) * EmbeddingFactor * D));
    bias_local_ = bias_tmp_buf_.Get<T>();
    DataCopy(bias_local_, bias_global_, ALIGN(EmbeddingFactor * D, 16));
    pipe_barrier(PIPE_ALL);
  }
  __aicore__ inline void Process() {
    int block_id = GetBlockIdx();
    constexpr int EmbeddingFactor = 3;  // q, k and v
    for (int i = 0; i < per_core_blocks_; i++) {
      int token_id = block_id * per_core_blocks_ + i;
      if (token_id >= total_blocks_) break;
      int batch_id;
      bool incremental;
      vsl_helper_.GetBatchId(token_id, &batch_id);
      vsl_helper_.GetIncrementalMode(batch_id, &incremental);
      int src_offset = token_id * EmbeddingFactor * head_size_ * head_num_;
      int stride = head_num_ * head_size_;
      copyIn(src_offset);
      process(bias_local_);
      int dstOffset;
      GetQOffset(token_id, &dstOffset);
      CopyOut(q_global_, dstOffset);

      copyIn(src_offset + stride);
      process(bias_local_[stride]);
      GetKVOffset(token_id, incremental, &dstOffset);
      CopyOut(k_global_, dstOffset);

      copyIn(src_offset + 2 * stride);
      process(bias_local_[2 * stride]);
      CopyOut(v_global_, dstOffset);
    }
  }

 private:
  __aicore__ inline void copyIn(int offset) {
    LocalTensor<T> srcLocal = in_.template AllocTensor<T>();
    DataCopy(srcLocal, qkv_global_[offset], copy_size_);
    in_.EnQue(srcLocal);
  }

  __aicore__ inline void process(const LocalTensor<T> &bias) {
    LocalTensor<T> srcLocal = in_.template DeQue<T>();
    LocalTensor<T> dstLocal = out_.template AllocTensor<T>();
    Add(dstLocal, srcLocal, bias, copy_size_);
    out_.EnQue(dstLocal);
    in_.FreeTensor(srcLocal);
  }

  __aicore__ inline void CopyOut(const GlobalTensor<T> &output, int offset) {
    LocalTensor<T> dstLocal = out_.template DeQue<T>();
    DataCopy(output[offset], dstLocal, copy_size_);
    out_.FreeTensor(dstLocal);
  }
  __aicore__ inline void GetTokenIdFromIdx(int idx, int *token_id) { *token_id = idx / head_num_; }
  __aicore__ inline void GetHeadIdIdFromIdx(int idx, int *head_id) { *head_id = idx % head_num_; }

  __aicore__ inline void GetQOffset(int token_id, int *offset) {
    // GetPackOffset(token_id, head_id, false, offset);
    bool incremental;
    int batch_id, seq_id;
    vsl_helper_.GetBatchId(token_id, &batch_id);
    vsl_helper_.GetSeqId(token_id, &seq_id);

    vsl_helper_.GetIncrementalMode(batch_id, &incremental);
    int cur_offset;
    vsl_helper_.GetActualOffset(batch_id, &cur_offset);
    *offset = cur_offset * head_size_ * head_num_;
    if (!incremental) {
      *offset += seq_id * head_size_ * head_num_;
    }
  }
  __aicore__ inline void GetKVOffset(int token_id, bool incremental, int *offset) {
    GetPaddedOffset(token_id, incremental, offset);
  }
  __aicore__ inline void GetPaddedOffset(int token_id, bool incremental, int *offset) {
    int batch_id, seq_id;
    vsl_helper_.GetBatchId(token_id, &batch_id);
    vsl_helper_.GetSeqId(token_id, &seq_id);

    int off = batch_id * max_seq_len_ * head_size_ * head_num_;
    // B x S x H x d
    off += seq_id * head_size_ * head_num_;
    if (incremental) {
      int kv_seq_id;
      vsl_helper_.GetKVSeqLen(token_id, &kv_seq_id);
      // Compute the kv_seq_id
      off += (kv_seq_id - 1) * head_size_ * head_num_;
    }
    *offset = off;
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, PipeSize> in_;
  TQue<QuePosition::VECOUT, PipeSize> out_;
  GlobalTensor<T> q_global_, k_global_, v_global_, qkv_global_, bias_global_;
  LocalTensor<T> q_local_, k_local_, v_local_, bias_local_;
  TBuf<QuePosition::VECIN> bias_tmp_buf_;
  KernelVsl vsl_helper_;
  int batch_, max_seq_len_, head_num_, head_size_, actual_token_;
  int total_blocks_, per_core_blocks_;
  int copy_size_;
};
template <typename T, int PipeSize>
__aicore__ void KernelQKVPermuteOperator(GM_ADDR qkv_ptr, GM_ADDR bias_ptr, GM_ADDR q_ptr, GM_ADDR k_cache_ptr,
                                         GM_ADDR v_cache_ptr, GM_ADDR q_seq_len, GM_ADDR kv_seq_len,
                                         GM_ADDR q_padding_offset, GM_ADDR kv_padding_offset, GM_ADDR mode,
                                         int actual_token, int batch, int seq_len, int head_num, int head_size,
                                         int total_blocks, int per_core_blocks) {
  KernelPermuteQKV<T, PipeSize> op;
  op.Init(qkv_ptr,            // qkv
          bias_ptr,           // bias
          q_ptr,              // q
          k_cache_ptr,        // k
          v_cache_ptr,        // v
          q_seq_len,          // q_seq_len
          kv_seq_len,         // kv_seq_len
          q_padding_offset,   // qpadding
          kv_padding_offset,  // kvpadding
          mode,               // mode
          actual_token, batch, seq_len, head_num, head_size, total_blocks, per_core_blocks);
  op.Process();
}
// transpose

template <uint32_t PipeSize, uint32_t ChunkSize, typename DataType>
class KernelTranspose0213 {
 public:
  __aicore__ inline KernelTranspose0213() = default;
  __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, GM_ADDR seq_lens, GM_ADDR padding_offset, GM_ADDR mode,
                              uint32_t elem_per_core, uint32_t total_token, uint32_t batch_size, uint32_t seq_len,
                              uint32_t head_num, uint32_t head_size) {
    vsl_helper_.Init(seq_lens, seq_lens, padding_offset, padding_offset, mode, total_token, batch_size, seq_len,
                     &pipe_);
    elem_per_core_ = elem_per_core;
    total_token_ = total_token;
    head_size_ = head_size;
    head_num_ = head_num;
    seq_len_ = seq_len;
    batch_size_ = batch_size;
    int D = head_num * head_size;
    actual_chunk_size_ = (ChunkSize > D) ? D : ChunkSize;
    chunk_num_ = UP_DIV(D, actual_chunk_size_);
    src_global_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(src), batch_size_ * seq_len_ * D);
    dst_global_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(dst), total_token_ * D);
    pipe_.InitBuffer(in_queue_, PipeSize, sizeof(DataType) * actual_chunk_size_);
    pipe_.InitBuffer(out_queue_, PipeSize, sizeof(DataType) * actual_chunk_size_);
  }

  __aicore__ inline void Process() {
    int block_id = GetBlockIdx();
    for (int t = 0; t < elem_per_core_; t++) {
      int token_id = block_id * elem_per_core_ + t;
      if (token_id < total_token_) {
        int batch_id;
        bool incremental;
        vsl_helper_.GetBatchId(token_id, &batch_id);
        vsl_helper_.GetIncrementalMode(batch_id, &incremental);
        for (int c = 0; c < chunk_num_; c++) {
          uint32_t chunk_offset = c * actual_chunk_size_;
          uint32_t actual_elem = (head_size_ * head_num_) > (chunk_offset + actual_chunk_size_)
                                   ? actual_chunk_size_
                                   : (head_size_ * head_num_) - chunk_offset;
          uint32_t outOffset = token_id * head_size_ * head_num_ + chunk_offset;
          CopyInData(token_id, incremental, chunk_offset, actual_elem);
          Compute(actual_elem);
          CopyOut(outOffset, actual_elem);
        }
      }
    }
  }

 private:
  __aicore__ inline void GetPaddedOffset(int token_id, bool incremental, int *offset) {
    int batch_id, seq_id;
    vsl_helper_.GetBatchId(token_id, &batch_id);
    vsl_helper_.GetSeqId(token_id, &seq_id);

    int cur_offset;
    vsl_helper_.GetActualOffset(batch_id, &cur_offset);
    int off = cur_offset * head_size_ * head_num_;
    // B x S x H x d
    if (!incremental) {
      off += seq_id * head_size_ * head_num_;
    }
    *offset = off;
  }
  __aicore__ inline void CopyInData(uint32_t h_token_id, bool incremental, uint32_t chunk_offset,
                                    uint32_t actual_elem) {
    LocalTensor<DataType> input_x_local = in_queue_.template AllocTensor<DataType>();
    int offset;
    GetPaddedOffset(h_token_id, incremental, &offset);
    offset = offset + chunk_offset;
    DataCopy(input_x_local, src_global_[offset], actual_elem);
    in_queue_.template EnQue(input_x_local);
  }
  __aicore__ inline void Compute(uint32_t actual_elem) {
    LocalTensor<DataType> input_x_local = in_queue_.template DeQue<DataType>();
    LocalTensor<DataType> output_local = out_queue_.template AllocTensor<DataType>();
    DataCopy(output_local, input_x_local, actual_elem);
    out_queue_.template EnQue<DataType>(output_local);
    in_queue_.FreeTensor(input_x_local);
  }

  __aicore__ inline void CopyOut(int offset, uint32_t actual_elem) {
    LocalTensor<DataType> output_local = out_queue_.template DeQue<DataType>();
    DataCopy(dst_global_[offset], output_local, actual_elem);
    out_queue_.template FreeTensor(output_local);
  }

 private:
  GlobalTensor<DataType> src_global_;
  GlobalTensor<DataType> dst_global_;
  TPipe pipe_;
  TQue<QuePosition::VECIN, PipeSize> in_queue_;
  TQue<QuePosition::VECOUT, PipeSize> out_queue_;
  uint32_t bufferSize_ = 0;
  uint32_t head_size_, head_num_, total_token_, batch_size_, seq_len_;
  uint32_t elem_per_core_;
  uint32_t chunk_num_;
  uint32_t actual_chunk_size_;
  KernelVsl vsl_helper_;
};

template <uint32_t PipeSize, uint32_t ChunkSize, typename DataType>
__aicore__ void KernelTranspose0213Operator(GM_ADDR src, GM_ADDR dst, GM_ADDR seq_lens, GM_ADDR padding_offset,
                                            GM_ADDR mode, uint32_t elem_per_core, uint32_t total_token,
                                            uint32_t batch_size, uint32_t seq_len, uint32_t head_num,
                                            uint32_t head_size) {
  KernelTranspose0213<PipeSize, ChunkSize, DataType> op;
  op.Init(src, dst, seq_lens, padding_offset, mode, elem_per_core, total_token, batch_size, seq_len, head_num,
          head_size);
  op.Process();
}

class KernelVSLCreate {
 public:
  __aicore__ inline KernelVSLCreate() = default;
  __aicore__ inline void Init(GM_ADDR batch_valid_len_gm, GM_ADDR position_idx_gm, GM_ADDR q_seq_len_gm,
                              GM_ADDR kv_seq_len_gm, GM_ADDR q_padding_offset_gm, GM_ADDR kv_padding_offset_gm,
                              GM_ADDR mode, GM_ADDR token_num_gm, uint32_t batch_size, uint32_t max_seq_len) {
    max_seq_len_ = max_seq_len;
    batch_size_ = batch_size;
    batch_valid_len_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(batch_valid_len_gm), batch_size_);
    position_idx_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(position_idx_gm), batch_size * max_seq_len);
    q_seq_len_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(q_seq_len_gm), batch_size);
    kv_seq_len_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(kv_seq_len_gm), batch_size);
    q_padding_offset_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(q_padding_offset_gm),
                                             batch_size * max_seq_len);
    kv_padding_offset_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(kv_padding_offset_gm),
                                              batch_size * max_seq_len);
    mode_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(mode), batch_size);
    token_num_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(token_num_gm), 2);
    pipe_.InitBuffer(q_tmp_, ALIGN32(max_seq_len * sizeof(int)));
    pipe_.InitBuffer(kv_tmp_, ALIGN32(max_seq_len * sizeof(int)));
    q_tmp_local_ = q_tmp_.Get<int>();
    kv_tmp_local_ = kv_tmp_.Get<int>();
  }

  __aicore__ inline void Process() {
    int q_total_seq_len = 0;
    int kv_total_seq_len = 0;
    int q_cum_offset = 0;
    int kv_cum_offset = 0;
    for (int batch_id = 0; batch_id < batch_size_; batch_id++) {
      int seq_pos = max_seq_len_;
      if (position_idx_global_.GetValue(0) > 0) {
        seq_pos = 1;
      }
      int incremental = (position_idx_global_.GetValue(batch_id * seq_pos) > 0) ? 1 : 0;
      mode_global_.SetValue(batch_id, incremental);
      int token_num = batch_valid_len_global_.GetValue(batch_id);
      token_num = (token_num == -1) ? -1 : token_num + 1;
      int q_token_num = (incremental && token_num != -1) ? 1 : token_num;
      q_seq_len_global_.SetValue(batch_id, q_token_num);
      kv_seq_len_global_.SetValue(batch_id, token_num);
      int q_seq_len = (q_token_num == -1) ? 0 : q_token_num;
      Muls(q_tmp_local_, q_tmp_local_, 0, ALIGN_BY_TYPE(q_seq_len, sizeof(int), 32));
      Muls(kv_tmp_local_, kv_tmp_local_, 0, ALIGN_BY_TYPE(token_num, sizeof(int), 32));
      Adds(q_tmp_local_, q_tmp_local_, q_cum_offset, ALIGN_BY_TYPE(q_seq_len, sizeof(int), 32));
      Adds(kv_tmp_local_, kv_tmp_local_, kv_cum_offset, ALIGN_BY_TYPE(token_num, sizeof(int), 32));
      pipe_barrier(PIPE_ALL);
      DataCopy(q_padding_offset_global_[q_total_seq_len], q_tmp_local_, ALIGN_BY_TYPE(q_seq_len, sizeof(int), 32));
      DataCopy(kv_padding_offset_global_[kv_total_seq_len], kv_tmp_local_, ALIGN_BY_TYPE(token_num, sizeof(int), 32));
      pipe_barrier(PIPE_ALL);
      q_cum_offset += max_seq_len_ - q_seq_len;
      kv_cum_offset += max_seq_len_ - token_num;
      q_total_seq_len += q_seq_len;
      kv_total_seq_len += token_num;
    }
    token_num_global_.SetValue(0, q_total_seq_len);
    token_num_global_.SetValue(1, kv_total_seq_len);
  }

 private:
  GlobalTensor<int> batch_valid_len_global_;
  GlobalTensor<int> position_idx_global_;
  GlobalTensor<int> q_seq_len_global_;
  GlobalTensor<int> kv_seq_len_global_;
  GlobalTensor<int> q_padding_offset_global_;
  GlobalTensor<int> kv_padding_offset_global_;
  GlobalTensor<int> mode_global_;
  GlobalTensor<int> token_num_global_;
  TBuf<QuePosition::VECCALC> q_tmp_;
  TBuf<QuePosition::VECCALC> kv_tmp_;
  LocalTensor<int> q_tmp_local_;
  LocalTensor<int> kv_tmp_local_;
  TPipe pipe_;
  uint32_t max_seq_len_, batch_size_;
};

template <int pipeSize, int ChunkSize, typename T>
class KernelAddScatter {
 public:
  __aicore__ inline KernelAddScatter() {}
  __aicore__ inline void Init(GM_ADDR in1, GM_ADDR in2, GM_ADDR out, uint32_t token_num, uint32_t hidden_size,
                              uint32_t elem_per_core, GM_ADDR token_to_token_gm) {
    elem_per_core_ = elem_per_core;
    token_num_ = token_num;
    hidden_size_ = hidden_size;
    actual_chunk_size_ = (ChunkSize > hidden_size) ? hidden_size : ChunkSize;
    chunk_num_ = UP_DIV(hidden_size, actual_chunk_size_);
    src1_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(in1), token_num * hidden_size);
    src2_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(in2), token_num * hidden_size);
    dst_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), token_num * hidden_size);
    is_scatter_ = (token_to_token_gm != nullptr);
    if (is_scatter_) {
      token_to_token_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(token_to_token_gm), token_num);
    }
    pipe_.InitBuffer(inQueueX1_, pipeSize, ALIGN32(actual_chunk_size_ * sizeof(T)));
    pipe_.InitBuffer(inQueueX2_, pipeSize, ALIGN32(actual_chunk_size_ * sizeof(T)));
    pipe_.InitBuffer(outQueue_, pipeSize, ALIGN32(actual_chunk_size_ * sizeof(T)));
  }
  __aicore__ inline void Process() {
    int block_id = GetBlockIdx();
    for (int t = 0; t < elem_per_core_; t++) {
      int token_id = block_id * elem_per_core_ + t;
      int token_in_1 = token_id;
      if (token_id < token_num_) {
        if (is_scatter_) {
          token_in_1 = token_to_token_global_.GetValue(token_id);
        }
        for (int c = 0; c < chunk_num_; c++) {
          uint32_t chunk_offset = c * actual_chunk_size_;
          uint32_t actual_elem =
            hidden_size_ > (chunk_offset + actual_chunk_size_) ? actual_chunk_size_ : hidden_size_ - chunk_offset;
          int offset_in_1 = token_in_1 * hidden_size_ + chunk_offset;
          if (token_in_1 == -1) {
            offset_in_1 = -1;
          }
          uint32_t offset = token_id * hidden_size_ + chunk_offset;
          CopyIn(offset_in_1, offset, actual_elem);
          Compute(actual_elem, token_in_1);
          CopyOut(offset, actual_elem);
        }
      }
    }
  }

 private:
  __aicore__ inline void CopyIn(int offset_in_1, uint32_t offset, uint32_t size) {
    LocalTensor<T> src1Local = inQueueX1_.template AllocTensor<T>();
    LocalTensor<T> src2Local = inQueueX2_.template AllocTensor<T>();
    uint32_t cpyElem = ALIGN32(size * sizeof(T)) / sizeof(T);
    if (offset_in_1 != -1) {
      DataCopy(src1Local, src1_global_[offset_in_1], cpyElem);
    }
    DataCopy(src2Local, src2_global_[offset], cpyElem);
    inQueueX1_.EnQue(src1Local);
    inQueueX2_.EnQue(src2Local);
  }
  __aicore__ inline void Compute(uint32_t size, int token_id) {
    LocalTensor<T> dstLocal = outQueue_.template AllocTensor<T>();
    LocalTensor<T> src1Local = inQueueX1_.template DeQue<T>();
    LocalTensor<T> src2Local = inQueueX2_.template DeQue<T>();
    if (token_id != -1) {
      Add<T>(dstLocal, src1Local, src2Local, size);
    } else {
      DataCopy(dstLocal, src2Local, size);
    }
    outQueue_.template EnQue<T>(dstLocal);
    inQueueX1_.FreeTensor(src1Local);
    inQueueX2_.FreeTensor(src2Local);
  }
  __aicore__ inline void CopyOut(uint32_t offset, uint32_t size) {
    LocalTensor<T> dstLocal = outQueue_.template DeQue<T>();
    uint32_t cpyElem = ALIGN32(size * sizeof(T)) / sizeof(T);
    DataCopy(dst_global_[offset], dstLocal, cpyElem);
    outQueue_.FreeTensor(dstLocal);
  }

 private:
  GlobalTensor<T> src1_global_;
  GlobalTensor<T> src2_global_;
  GlobalTensor<T> dst_global_;
  GlobalTensor<int> token_to_token_global_;
  TPipe pipe_;
  TQue<QuePosition::VECIN, pipeSize> inQueueX1_;
  TQue<QuePosition::VECIN, pipeSize> inQueueX2_;
  TQue<QuePosition::VECOUT, pipeSize> outQueue_;
  uint32_t actual_chunk_size_ = 0;
  uint32_t chunk_num_ = 0;
  uint32_t elem_per_core_ = 0;
  uint32_t token_num_ = 0;
  uint32_t hidden_size_ = 0;
  bool is_scatter_;
};

__aicore__ void KernelCreateVSLOperator(GM_ADDR batch_valid_len_gm, GM_ADDR position_idx_gm, GM_ADDR q_seq_len_gm,
                                        GM_ADDR kv_seq_len_gm, GM_ADDR q_padding_offset_gm,
                                        GM_ADDR kv_padding_offset_gm, GM_ADDR mode_gm, GM_ADDR token_num_gm,
                                        uint32_t batch_size, uint32_t max_seq_len) {
  KernelVSLCreate op;
  op.Init(batch_valid_len_gm, position_idx_gm, q_seq_len_gm, kv_seq_len_gm, q_padding_offset_gm, kv_padding_offset_gm,
          mode_gm, token_num_gm, batch_size, max_seq_len);
  op.Process();
}

template <uint32_t PipeSize, uint32_t ChunkSize, typename DataType>
class KernelVocabEmbedding {
 public:
  __aicore__ inline KernelVocabEmbedding() = default;
  __aicore__ inline void Init(GM_ADDR position_idx, GM_ADDR embedding_table, GM_ADDR out, GM_ADDR seq_lens,
                              GM_ADDR padding_offset, GM_ADDR mode, uint32_t elem_per_core, uint32_t total_token,
                              uint32_t batch_size, uint32_t seq_len, uint32_t hidden_size) {
    vsl_helper_.Init(seq_lens, seq_lens, padding_offset, padding_offset, mode, total_token, batch_size, seq_len,
                     &pipe_);
    elem_per_core_ = elem_per_core;
    total_token_ = total_token;
    hidden_size_ = hidden_size;
    seq_len_ = seq_len;
    actual_chunk_size_ = (ChunkSize > hidden_size) ? hidden_size : ChunkSize;
    chunk_num_ = UP_DIV(hidden_size, actual_chunk_size_);
    position_idx_global_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(position_idx), batch_size_ * seq_len_);
    embedding_table_global_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(embedding_table),
                                            seq_len_ * hidden_size);
    out_global_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(out), total_token_ * hidden_size);
    pipe_.InitBuffer(embedding_table_queue_, PipeSize, sizeof(DataType) * actual_chunk_size_);
    pipe_.InitBuffer(out_queue_, PipeSize, sizeof(DataType) * actual_chunk_size_);
  }

  __aicore__ inline void Process() {
    int block_id = GetBlockIdx();
    for (int t = 0; t < elem_per_core_; t++) {
      uint32_t token_id = block_id * elem_per_core_ + t;
      if (token_id < total_token_) {
        for (int c = 0; c < chunk_num_; c++) {
          uint32_t chunk_offset = c * actual_chunk_size_;
          uint32_t actual_elem =
            hidden_size_ > (chunk_offset + actual_chunk_size_) ? actual_chunk_size_ : hidden_size_ - chunk_offset;
          uint32_t offset = token_id * hidden_size_ * 3 + chunk_offset;
          CopyInData(token_id, chunk_offset, actual_elem);
          Compute(actual_elem);
          CopyOut(offset, actual_elem);
        }
      }
    }
  }

 private:
  __aicore__ inline void GetPaddedOffset(int token_id, int *offset) {
    int batch_id, seq_id;
    bool is_inc;
    vsl_helper_.GetBatchId(token_id, &batch_id);
    vsl_helper_.GetIncrementalMode(batch_id, &is_inc);
    vsl_helper_.GetSeqId(token_id, &seq_id);
    *offset = batch_id * seq_len_ + seq_id;
  }
  __aicore__ inline void CopyInData(uint32_t h_token_id, int32_t chunk_offset, uint32_t actual_elem) {
    LocalTensor<DataType> embedding_table_local = embedding_table_queue_.template AllocTensor<DataType>();
    int in_offset;
    GetPaddedOffset(h_token_id, &in_offset);
    int token = position_idx_global_.GetValue(in_offset);
    int offset = token * hidden_size_ + chunk_offset;
    DataCopy(embedding_table_local, embedding_table_global_[offset], actual_elem);
    embedding_table_queue_.template EnQue(embedding_table_local);
  }
  __aicore__ inline void Compute(uint32_t actual_elem) {
    LocalTensor<DataType> embedding_table_local = embedding_table_queue_.template DeQue<DataType>();
    LocalTensor<DataType> output_local = out_queue_.template AllocTensor<DataType>();
    DataCopy(output_local, embedding_table_local, actual_elem);
    out_queue_.template EnQue<DataType>(output_local);
    embedding_table_queue_.FreeTensor(embedding_table_local);
  }

  __aicore__ inline void CopyOut(int offset, uint32_t actual_elem) {
    LocalTensor<DataType> output_local = out_queue_.template DeQue<DataType>();
    DataCopy(out_global_[offset], output_local, actual_elem);
    out_queue_.template FreeTensor(output_local);
  }

 private:
  GlobalTensor<DataType> position_idx_global_;
  GlobalTensor<DataType> embedding_table_global_;
  GlobalTensor<DataType> out_global_;
  TPipe pipe_;
  TQue<QuePosition::VECIN, PipeSize> embedding_table_queue_;
  TQue<QuePosition::VECOUT, PipeSize> out_queue_;
  uint32_t bufferSize_ = 0;
  uint32_t hidden_size_, total_token_, batch_size_, seq_len_;
  uint32_t elem_per_core_;
  uint32_t chunk_num_;
  uint32_t actual_chunk_size_;
  KernelVsl vsl_helper_;
};

template <uint32_t PipeSize, uint32_t ChunkSize, typename DataType>
__aicore__ void KernelVocabEmbeddingOperator(GM_ADDR position_idx, GM_ADDR embedding_table, GM_ADDR out,
                                             GM_ADDR seq_lens, GM_ADDR padding_offset, GM_ADDR mode,
                                             uint32_t elem_per_core, uint32_t total_token, uint32_t batch_size,
                                             uint32_t seq_len, uint32_t hidden_size) {
  KernelVocabEmbedding<PipeSize, ChunkSize, DataType> op;
  op.Init(position_idx, embedding_table, out, seq_lens, padding_offset, mode, elem_per_core, total_token, batch_size,
          seq_len, hidden_size);
  op.Process();
}

// gather
template <uint32_t PipeSize, uint32_t ChunkSize, typename DataType>
class KernelGatherHead {
 public:
  __aicore__ inline KernelGatherHead() = default;
  __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR seq_len_gm, GM_ADDR padding_offset_gm,
                              GM_ADDR mode_gm, uint32_t elem_per_core, uint32_t total_token, uint32_t batch_size,
                              uint32_t seq_len, uint32_t hidden_size) {
    vsl_helper_.Init(seq_len_gm, seq_len_gm, padding_offset_gm, padding_offset_gm, mode_gm, total_token, batch_size,
                     seq_len, &pipe_);
    elem_per_core_ = elem_per_core;
    total_token_ = total_token;
    hidden_size_ = hidden_size;
    batch_size_ = batch_size;
    actual_chunk_size_ = (ChunkSize > hidden_size) ? hidden_size : ChunkSize;
    chunk_num_ = UP_DIV(hidden_size, actual_chunk_size_);
    src_global_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(src_gm), total_token * hidden_size);
    dst_global_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(dst_gm), batch_size * hidden_size);
    pipe_.InitBuffer(src_queue_, PipeSize, sizeof(DataType) * actual_chunk_size_);
    pipe_.InitBuffer(dst_queue_, PipeSize, sizeof(DataType) * actual_chunk_size_);
  }

  __aicore__ inline void Process() {
    int block_id = GetBlockIdx();
    for (int t = 0; t < elem_per_core_; t++) {
      uint32_t batch_id = block_id * elem_per_core_ + t;
      if (batch_id < batch_size_) {
        for (int c = 0; c < chunk_num_; c++) {
          uint32_t chunk_offset = c * actual_chunk_size_;
          uint32_t actual_elem =
            hidden_size_ > (chunk_offset + actual_chunk_size_) ? actual_chunk_size_ : hidden_size_ - chunk_offset;
          uint32_t offset = batch_id * hidden_size_ + chunk_offset;
          CopyInData(batch_id, chunk_offset, actual_elem);
          Compute(actual_elem);
          CopyOut(offset, actual_elem);
        }
      }
    }
  }

 private:
  __aicore__ inline void GetInOffset(int batch_id, int *offset) {
    int token_id, seq_len;
    vsl_helper_.GetTokenIdByBatch(batch_id, &token_id);
    vsl_helper_.GetSeqLen(batch_id, &seq_len);
    *offset = (token_id + seq_len - 1) * hidden_size_;
  }
  __aicore__ inline void CopyInData(uint32_t batch_id, int32_t chunk_offset, uint32_t actual_elem) {
    LocalTensor<DataType> src_local = src_queue_.template AllocTensor<DataType>();
    int in_offset;
    GetInOffset(batch_id, &in_offset);
    DataCopy(src_local, src_global_[in_offset], actual_elem);
    src_queue_.template EnQue(src_local);
  }
  __aicore__ inline void Compute(uint32_t actual_elem) {
    LocalTensor<DataType> src_local = src_queue_.template DeQue<DataType>();
    LocalTensor<DataType> dst_local = dst_queue_.template AllocTensor<DataType>();
    DataCopy(dst_local, src_local, actual_elem);
    dst_queue_.template EnQue<DataType>(dst_local);
    src_queue_.FreeTensor(src_local);
  }

  __aicore__ inline void CopyOut(int offset, uint32_t actual_elem) {
    LocalTensor<DataType> dst_local = dst_queue_.template DeQue<DataType>();
    DataCopy(dst_global_[offset], dst_local, actual_elem);
    dst_queue_.template FreeTensor(dst_local);
  }

 private:
  GlobalTensor<DataType> src_global_;
  GlobalTensor<DataType> dst_global_;
  TPipe pipe_;
  TQue<QuePosition::VECIN, PipeSize> src_queue_;
  TQue<QuePosition::VECOUT, PipeSize> dst_queue_;
  uint32_t hidden_size_, total_token_, batch_size_, seq_len_;
  uint32_t elem_per_core_;
  uint32_t chunk_num_;
  uint32_t actual_chunk_size_;
  KernelVsl vsl_helper_;
};

template <uint32_t PipeSize, uint32_t ChunkSize, typename DataType>
__aicore__ void KernelGatherHeadOperator(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR seq_len_gm, GM_ADDR padding_offset_gm,
                                         GM_ADDR mode_gm, uint32_t elem_per_core, uint32_t total_token,
                                         uint32_t batch_size, uint32_t seq_len, uint32_t hidden_size) {
  KernelGatherHead<PipeSize, ChunkSize, DataType> op;
  op.Init(src_gm, dst_gm, seq_len_gm, padding_offset_gm, mode_gm, elem_per_core, total_token, batch_size, seq_len,
          hidden_size);
  op.Process();
}

/// create token to token and expert count
class KernelCreateCountExpert {
 public:
  __aicore__ inline KernelCreateCountExpert() {}
  __aicore__ inline void Init(GM_ADDR expert_ids, GM_ADDR out, GM_ADDR seq_lens, GM_ADDR padding_offset, GM_ADDR mode,
                              uint32_t moe_num, uint32_t expert_num, float capacity, uint32_t batch_size,
                              uint32_t seq_len, uint32_t moe_id, bool is_query, uint32_t elem_per_core) {
    vsl_helper_.Init(seq_lens, seq_lens, padding_offset, padding_offset, mode, batch_size * seq_len, batch_size,
                     seq_len, &pipe_);
    seq_len_ = seq_len;
    batch_size_ = batch_size;
    expert_num_ = expert_num;
    elem_per_core_ = elem_per_core;
    moe_id_ = moe_id;
    is_query_ = is_query;
    max_capacity_ = UP_DIV((capacity * seq_len), expert_num);
    expert_ids_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(expert_ids), moe_num * batch_size_ * seq_len_);
    out_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(out), batch_size * expert_num);
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    pipe_.InitBuffer(expert_ids_queue_, 1, ALIGN32(sizeof(int) * seq_len_));
    pipe_.InitBuffer(count_tmp_, sizeof(int) * align_expert_num);
    pipe_.InitBuffer(expert_count_queue_, 1, sizeof(int) * align_expert_num);
    count_tmp_local_ = count_tmp_.Get<int>();
  }

  __aicore__ inline void Process() {
    int block_id = GetBlockIdx();
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    for (int t = 0; t < elem_per_core_; t++) {
      uint32_t batch_id = block_id * elem_per_core_ + t;
      if (batch_id < batch_size_) {
        CopyInData(batch_id);
        LocalTensor<int> expert_ids_local = expert_ids_queue_.template DeQue<int>();
        Muls(count_tmp_local_, count_tmp_local_, 0, align_expert_num);
        int actual_seq;
        vsl_helper_.GetSeqLen(batch_id, &actual_seq);
        actual_seq = (is_query_) ? actual_seq - 1 : actual_seq;
        int expert_id_last_token = expert_ids_local.GetValue(actual_seq);
        for (int seq_id = 0; seq_id < actual_seq; seq_id++) {
          int expert_id = expert_ids_local.GetValue(seq_id);
          int cur_count = count_tmp_local_.GetValue(expert_id);
          if (!(expert_id >= expert_num_ || expert_id < 0)) {
            if (cur_count < max_capacity_) {
              if (!(is_query_ && (expert_id_last_token != expert_id))) {
                count_tmp_local_.SetValue(expert_id, cur_count + 1);
              }
            }
          }
        }
        // if is_query calculate the last token only
        if (is_query_) {
          int cur_count = count_tmp_local_.GetValue(expert_id_last_token);
          if (!(expert_id_last_token >= expert_num_ || expert_id_last_token < 0)) {
            if (cur_count < max_capacity_) {
              count_tmp_local_.SetValue(expert_id_last_token, 1);
            } else {
              count_tmp_local_.SetValue(expert_id_last_token, 0);
            }
          }
        }
        LocalTensor<int> dst_local = expert_count_queue_.template AllocTensor<int>();
        DataCopy(dst_local, count_tmp_local_, align_expert_num);
        expert_count_queue_.template EnQue(dst_local);
        CopyOutData(batch_id);
        pipe_barrier(PIPE_ALL);
        expert_ids_queue_.FreeTensor(expert_ids_local);
      }
    }
  }
  __aicore__ inline void CopyInData(int batch_id) {
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    LocalTensor<int> expert_local = expert_ids_queue_.template AllocTensor<int>();
    bool is_inc;
    vsl_helper_.GetIncrementalMode(batch_id, &is_inc);
    int seq = (is_inc) ? 1 : seq_len_;
    if (is_inc) {
      expert_local.SetValue(0, expert_ids_global_.GetValue(moe_id_ * batch_size_ * seq + batch_id * seq));
    } else {
      DataCopy(expert_local, expert_ids_global_[moe_id_ * batch_size_ * seq + batch_id * seq],
               ALIGN32(seq * sizeof(int)) / sizeof(int));
    }
    expert_ids_queue_.template EnQue(expert_local);
  }

  __aicore__ inline void CopyOutData(int batch_id) {
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    LocalTensor<int> dst_local = expert_count_queue_.template DeQue<int>();
    DataCopy(out_global_[batch_id * align_expert_num], dst_local, align_expert_num);
    expert_count_queue_.FreeTensor(dst_local);
  }

 private:
  GlobalTensor<int> expert_ids_global_;
  GlobalTensor<int> out_global_;
  TQue<QuePosition::VECIN, 1> expert_ids_queue_;
  TQue<QuePosition::VECOUT, 1> expert_count_queue_;
  TBuf<QuePosition::VECCALC> count_tmp_;
  LocalTensor<int> count_tmp_local_;
  TPipe pipe_;
  uint32_t expert_num_, batch_size_, seq_len_, moe_id_;
  int max_capacity_;
  uint32_t elem_per_core_;
  bool is_query_;
  KernelVsl vsl_helper_;
};

class KernelCreateMoeParam {
 public:
  __aicore__ inline KernelCreateMoeParam() {}
  __aicore__ inline void Init(GM_ADDR expert_ids, GM_ADDR expert_count_by_batch, GM_ADDR expert_count,
                              GM_ADDR token_to_token, GM_ADDR seq_lens, GM_ADDR padding_offset, GM_ADDR mode,
                              uint32_t expert_num, uint32_t moe_num, uint32_t batch_size, uint32_t seq_len,
                              uint32_t total_token, uint32_t moe_id, float capacity, bool is_query,
                              uint32_t elem_per_core) {
    vsl_helper_.Init(seq_lens, seq_lens, padding_offset, padding_offset, mode, total_token, batch_size, seq_len,
                     &pipe_);
    seq_len_ = seq_len;
    batch_size_ = batch_size;
    expert_num_ = expert_num;
    elem_per_core_ = elem_per_core;
    moe_id_ = moe_id;
    is_query_ = is_query;
    inc_max_capacity_ = UP_DIV((capacity * batch_size), expert_num);
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    expert_ids_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(expert_ids), moe_num * batch_size_ * seq_len_);
    expert_count_by_batch_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(expert_count_by_batch),
                                                  batch_size_ * expert_num);
    token_to_token_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(token_to_token), batch_size_ * seq_len_);
    expert_count_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(expert_count), expert_num);
    pipe_.InitBuffer(expert_count_tmp_, sizeof(int) * align_expert_num * batch_size_);
    expert_count_tmp_local_ = expert_count_tmp_.Get<int>();
    pipe_.InitBuffer(count_tmp_, sizeof(int) * align_expert_num);
    count_tmp_local_ = count_tmp_.Get<int>();
    pipe_.InitBuffer(expert_by_batch_queue_, 1, sizeof(int) * align_expert_num * batch_size_);
    pipe_.InitBuffer(expert_count_queue_, 1, sizeof(int) * align_expert_num);
    pipe_.InitBuffer(expert_ids_queue_, 1, ALIGN32(sizeof(int) * seq_len_));
  }

  __aicore__ inline void Process() {
    int block_id = GetBlockIdx();
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    bool is_inc;
    vsl_helper_.GetIncrementalMode(0, &is_inc);
    CopyInCountExpert();
    LocalTensor<int> expert_count_by_batch_local = expert_by_batch_queue_.template DeQue<int>();
    // calculate count tokens per expert to 0...batch_id
    DataCopy(expert_count_tmp_local_, expert_count_by_batch_local, align_expert_num);
    pipe_barrier(PIPE_ALL);
    for (int i = 1; i < batch_size_; i++) {
      Add(expert_count_tmp_local_[i * align_expert_num], expert_count_by_batch_local[i * align_expert_num],
          expert_count_tmp_local_[(i - 1) * align_expert_num], align_expert_num);
    }

    for (int t = 0; t < elem_per_core_; t++) {
      uint32_t batch_id = block_id * elem_per_core_ + t;
      if (batch_id < batch_size_) {
        CopyInExpertIds(batch_id);
        LocalTensor<int> expert_ids_local = expert_ids_queue_.template DeQue<int>();
        Muls(count_tmp_local_, count_tmp_local_, 0, align_expert_num);
        // copy out count tokens per expert
        if (batch_id == 0) {
          LocalTensor<int> dst_local = expert_count_queue_.template AllocTensor<int>();
          DataCopy(dst_local, expert_count_tmp_local_[(batch_size_ - 1) * align_expert_num], align_expert_num);
          expert_count_queue_.template EnQue(dst_local);
          CopyOutData();
        }
        int actual_seq;
        // get actual seq for current batch
        vsl_helper_.GetSeqLen(batch_id, &actual_seq);
        int current_token_not_expert = 0;
        for (int seq_id = 0; seq_id < actual_seq; seq_id++) {
          // get expert id to current token
          int seq = (is_inc) ? 1 : seq_len_;
          int expert_id = expert_ids_local.GetValue(seq_id);
          bool is_expert;
          // check if expert id is valid
          is_expert = !(expert_id >= expert_num_ || expert_id < 0);
          // not expert if inc and there is token expert to expert_id until current batch current seq_id
          is_expert = is_expert && !(is_inc && batch_id > 0 &&
                                     expert_count_tmp_local_.GetValue((batch_id - 1) * align_expert_num + expert_id) +
                                         count_tmp_local_.GetValue(expert_id) >
                                       inc_max_capacity_ - 1);
          // not expert if is query and the it's not the last token (useless)
          is_expert = is_expert && !(is_query_ && seq_id < actual_seq - 1);
          // not expert if expert id for current batch pass max capacity (expert_count_by_batch_local include the max
          // tokens per batch and expert)
          is_expert = is_expert && (count_tmp_local_.GetValue(expert_id) <
                                    expert_count_by_batch_local.GetValue(batch_id * expert_num_ + expert_id));
          if (is_expert) {
            // calculate token id after gather for expert
            SetTokenId(batch_id, seq_id, expert_id, is_inc);
          } else {
            // set -1 for token_id after gather for token that will not expert
            int token_offset;
            vsl_helper_.GetTokenIdByBatch(batch_id, &token_offset);
            token_to_token_global_.SetValue(token_offset + seq_id, -1);
          }
        }
        pipe_barrier(PIPE_ALL);
        expert_ids_queue_.FreeTensor(expert_ids_local);
      }
    }
    pipe_barrier(PIPE_ALL);
    expert_by_batch_queue_.FreeTensor(expert_count_by_batch_local);
  }
  __aicore__ inline void CopyInCountExpert() {
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    LocalTensor<int> expert_local = expert_by_batch_queue_.template AllocTensor<int>();
    DataCopy(expert_local, expert_count_by_batch_global_, align_expert_num * batch_size_);
    expert_by_batch_queue_.template EnQue(expert_local);
  }
  __aicore__ inline void SetTokenId(int batch_id, int seq_id, int expert_id, bool is_inc) {
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    int token_id = 0;
    // add count tokens until current expert id
    for (int i = 0; i < expert_id; i++) {
      int expert_count = expert_count_tmp_local_.GetValue((batch_size_ - 1) * align_expert_num + i);
      token_id += (is_inc && expert_count > 0) ? (expert_count > inc_max_capacity_) ? inc_max_capacity_ : expert_count
                                               : expert_count;
    }
    if (batch_id > 0) {
      // add count tokens for expert id until current batch
      token_id += expert_count_tmp_local_.GetValue((batch_id - 1) * align_expert_num + expert_id);
    }
    // add count tokens for expert id in batch id until current token
    int cur_count = count_tmp_local_.GetValue(expert_id);
    token_id += cur_count;
    count_tmp_local_.SetValue(expert_id, cur_count + 1);
    // calculate current token offset in compress data
    int token_offset;
    vsl_helper_.GetTokenIdByBatch(batch_id, &token_offset);
    token_to_token_global_.SetValue(token_offset + seq_id, token_id);
  }
  __aicore__ inline void CopyOutData() {
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    LocalTensor<int> dst_local = expert_count_queue_.template DeQue<int>();
    DataCopy(expert_count_global_, dst_local, align_expert_num);
    expert_count_queue_.FreeTensor(dst_local);
  }
  __aicore__ inline void CopyInExpertIds(int batch_id) {
    int align_expert_num = ALIGN32(sizeof(int) * expert_num_) / sizeof(int);
    LocalTensor<int> expert_local = expert_ids_queue_.template AllocTensor<int>();
    bool is_inc;
    vsl_helper_.GetIncrementalMode(batch_id, &is_inc);
    int seq = (is_inc) ? 1 : seq_len_;
    if (is_inc) {
      expert_local.SetValue(0, expert_ids_global_.GetValue(moe_id_ * batch_size_ * seq + batch_id * seq));
    } else {
      DataCopy(expert_local, expert_ids_global_[moe_id_ * batch_size_ * seq + batch_id * seq],
               ALIGN32(seq * sizeof(int)) / sizeof(int));
    }
    expert_ids_queue_.template EnQue(expert_local);
  }

 private:
  GlobalTensor<int> expert_ids_global_;
  GlobalTensor<int> expert_count_global_;
  GlobalTensor<int> token_to_token_global_;
  GlobalTensor<int> expert_count_by_batch_global_;
  TQue<QuePosition::VECIN, 1> expert_by_batch_queue_;
  TQue<QuePosition::VECOUT, 1> expert_count_queue_;
  TQue<QuePosition::VECIN, 1> expert_ids_queue_;
  TPipe pipe_;
  uint32_t expert_num_, batch_size_, seq_len_, moe_id_;
  int inc_max_capacity_;
  uint32_t elem_per_core_;
  bool is_query_;
  KernelVsl vsl_helper_;
  TBuf<QuePosition::VECCALC> count_tmp_, expert_count_tmp_;
  LocalTensor<int> count_tmp_local_, expert_count_tmp_local_;
};
}  // namespace mindspore::ascend_native
#endif
