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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_NORMALIZATION_LAYER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_NORMALIZATION_LAYER_H_

#include "tikcfw/kernel_operator.h"
#include "tikcfw/impl/kernel_utils.h"
#include "tikcfw/interface/kernel_operator_intf.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/vsl_utils.h"
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
namespace mindspore::ascend_native {

template <uint32_t pipeSize, uint32_t chunkSize, typename T, bool isReuseSource = false>
class KernelLayernorm {
 public:
  __aicore__ inline KernelLayernorm() = default;
  __aicore__ inline uint32_t InitReduceWorkspaceSize() {
    constexpr int block_size = 32;
    constexpr int repeat_size = 256;
    int type_size = sizeof(T);
    // Number of elements a block can hold
    int elements_per_block = block_size / type_size;
    // Number of elements that can be processed in a repeat
    int elements_per_repeat = repeat_size / type_size;
    int first_max_repeat = UP_DIV(actual_chunk_size_, elements_per_repeat);
    int iter1_align_end = UP_DIV(first_max_repeat, elements_per_block) * elements_per_block;
    return iter1_align_end;
  }
  __aicore__ inline void Init(GM_ADDR inputx_gm, GM_ADDR inputy_gm, GM_ADDR bias_gm, GM_ADDR gamma_gm, GM_ADDR beta_gm,
                              GM_ADDR output_gm, GM_ADDR output_norm_gm, GM_ADDR input_ids_gm, GM_ADDR input_pos_gm,
                              GM_ADDR emmbeding_word_gm, GM_ADDR emmbeding_pos_gm, uint32_t token_per_block,
                              uint32_t total_token, uint32_t h_length, float h_length_float, T epsilon,
                              uint32_t batch_size, uint32_t v_length, uint32_t seq_length, GM_ADDR seq_len_gm,
                              GM_ADDR padding_offset_gm, GM_ADDR mode_gm, GM_ADDR token_to_token_gm) {
    int blckId = GetBlockIdx();
    token_per_block_ = token_per_block;
    total_token_ = total_token;
    h_length_ = h_length;
    epsilon_ = epsilon;
    float_h_length_ = h_length_float;
    actual_chunk_size_ = (chunkSize > h_length_) ? h_length_ : chunkSize;
    chunk_num_ = UP_DIV(h_length_, actual_chunk_size_);
    uint32_t input_size = h_length_ * total_token_;
    uint32_t reduce_work_space_size = InitReduceWorkspaceSize();
    seq_length_ = seq_length;
    // Setup global pointers
    is_embedding_ = (input_ids_gm != nullptr);
    is_moe_gather_ = (token_to_token_gm != nullptr);
    if (!is_embedding_) {
      inputx_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputx_gm), input_size);
    }
    if (is_moe_gather_) {
      token_to_token_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(token_to_token_gm), total_token);
    }
    gamma_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma_gm), h_length_);
    beta_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(beta_gm), h_length_);
    output_norm_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output_norm_gm), input_size);
    is_residual_ = (inputy_gm != nullptr);
    if (is_residual_) {
      inputy_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputy_gm), input_size);
      pipe_.InitBuffer(in_queue_y_, pipeSize, ALIGN32(sizeof(T) * actual_chunk_size_));
      output_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output_gm), input_size);
      pipe_.InitBuffer(out_queue_, pipeSize, ALIGN32(sizeof(T) * actual_chunk_size_));
    }
    is_bias_ = (bias_gm != nullptr);
    if (is_bias_) {
      bias_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias_gm), h_length_);
      pipe_.InitBuffer(in_buf_bias_, ALIGN32(sizeof(T) * h_length_));
      bias_tmp_local_ = in_buf_bias_.Get<T>();
    }

    if (is_embedding_) {
      vsl_helper_.Init(seq_len_gm, seq_len_gm, padding_offset_gm, padding_offset_gm, mode_gm, total_token, batch_size,
                       seq_length, &pipe_);
      output_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output_gm), h_length_ * total_token_);
      pipe_.InitBuffer(out_queue_, pipeSize, ALIGN32(sizeof(T) * actual_chunk_size_));
      input_ids_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(input_ids_gm), total_token_);
      input_pos_global_.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(input_pos_gm), total_token_);
      emmbeding_word_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(emmbeding_word_gm), v_length * h_length_);
      emmbeding_pos_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(emmbeding_pos_gm), seq_length * h_length_);
      pipe_.InitBuffer(in_embedd_queue_, pipeSize, ALIGN32(sizeof(T) * actual_chunk_size_));
      pipe_.InitBuffer(in_embedd_pos_queue_, pipeSize, ALIGN32(sizeof(T) * actual_chunk_size_));
    }

    pipe_.InitBuffer(in_queue_x_, pipeSize, ALIGN32(sizeof(T) * actual_chunk_size_));
    pipe_.InitBuffer(out_norm_queue_, pipeSize, ALIGN32(sizeof(T) * actual_chunk_size_));
    pipe_.InitBuffer(in_buf_gamma_, ALIGN32(sizeof(T) * h_length_));
    pipe_.InitBuffer(in_buf_beta_, ALIGN32(sizeof(T) * h_length_));
    pipe_.InitBuffer(x_tmp_, ALIGN32(sizeof(T) * h_length_));
    pipe_.InitBuffer(reduce_sum_tmp_, ALIGN32(sizeof(T) * chunk_num_));
    pipe_.InitBuffer(reduce_sum_of_sqr_tmp_, ALIGN32(sizeof(T) * chunk_num_));
    pipe_.InitBuffer(reduce_output_tmp_, ALIGN32(sizeof(T) * actual_chunk_size_));
    pipe_.InitBuffer(reduce_tmp_, ALIGN32(sizeof(T) * reduce_work_space_size));
    gamma_tmp_local_ = in_buf_gamma_.Get<T>();
    beta_tmp_local_ = in_buf_beta_.Get<T>();
    x_tmp_local_ = x_tmp_.Get<T>();
    reduce_tmp_local_ = reduce_tmp_.Get<T>();
    reduce_sum_tmp_local_ = reduce_sum_tmp_.Get<T>();
    reduce_sum_of_sqr_tmp_local_ = reduce_sum_of_sqr_tmp_.Get<T>();
    reduce_output_tmp_local_ = reduce_output_tmp_.Get<T>();
  }
  __aicore__ inline void Process() {
    int blckId = GetBlockIdx();
    CopyInWeight();
    pipe_barrier(PIPE_ALL);
    for (int t = 0; t < token_per_block_; t++) {
      uint32_t token_id = blckId * token_per_block_ + t;
      if (token_id < total_token_) {
        for (int c = 0; c < chunk_num_; c++) {
          uint32_t chunk_offset = c * actual_chunk_size_;
          uint32_t actual_elem =
            h_length_ > (chunk_offset + actual_chunk_size_) ? actual_chunk_size_ : h_length_ - chunk_offset;
          CopyInData(token_id, c, actual_elem);
          ComputeReduction(c, actual_elem);
          CopyOut(token_id * ALIGN_BY_TYPE(h_length_, sizeof(T), 32) + chunk_offset, actual_elem);
        }
        ComputeMeanAndVar();
        pipe_barrier(PIPE_ALL);
        int out_token_id = token_id;
        if (is_moe_gather_) {
          out_token_id = token_to_token_global_.GetValue(token_id);
        }
        for (int c = 0; c < chunk_num_; c++) {
          uint32_t chunk_offset = c * actual_chunk_size_;
          uint32_t actual_elem =
            h_length_ > (chunk_offset + actual_chunk_size_) ? actual_chunk_size_ : h_length_ - chunk_offset;
          Normalize(c, actual_elem);
          CopyNormOut(out_token_id * ALIGN_BY_TYPE(h_length_, sizeof(T), 32) + chunk_offset, actual_elem, out_token_id);
          pipe_barrier(PIPE_ALL);
        }
      }
    }
  }

 private:
  __aicore__ inline void CopyInWeight() {
    DataCopy(gamma_tmp_local_, gamma_global_, ALIGN_BY_TYPE(h_length_, sizeof(T), 32));
    DataCopy(beta_tmp_local_, beta_global_, ALIGN_BY_TYPE(h_length_, sizeof(T), 32));
    if (is_bias_) {
      DataCopy(bias_tmp_local_, bias_global_, ALIGN_BY_TYPE(h_length_, sizeof(T), 32));
    }
  }
  __aicore__ inline void CopyInData(uint32_t token_id, uint32_t chunk_id, uint32_t actual_elem) {
    LocalTensor<T> inputx_local = in_queue_x_.template AllocTensor<T>();
    if (is_embedding_) {
      int batch_id, seq_id, q_seq_len;
      bool is_inc = false;
      vsl_helper_.GetBatchId(token_id, &batch_id);
      vsl_helper_.GetSeqId(token_id, &seq_id);
      vsl_helper_.GetIncrementalMode(batch_id, &is_inc);
      q_seq_len = is_inc ? 1 : seq_length_;
      int32_t embedding_offset =
        input_ids_global_.GetValue(batch_id * q_seq_len + seq_id) * h_length_ + chunk_id * actual_chunk_size_;
      int32_t embedding_pos_offset =
        input_pos_global_.GetValue(batch_id * q_seq_len + seq_id) * h_length_ + chunk_id * actual_chunk_size_;
      LocalTensor<T> input_embedd_local = in_embedd_queue_.template AllocTensor<T>();
      LocalTensor<T> input_pos_embedd_local = in_embedd_pos_queue_.template AllocTensor<T>();
      DataCopy(input_embedd_local, emmbeding_word_global_[embedding_offset], ALIGN_BY_TYPE(actual_elem, sizeof(T), 32));
      DataCopy(input_pos_embedd_local, emmbeding_pos_global_[embedding_pos_offset],
               ALIGN_BY_TYPE(actual_elem, sizeof(T), 32));
      pipe_barrier(PIPE_ALL);  // synchronize before Add
      Add(inputx_local, input_embedd_local, input_pos_embedd_local, actual_elem);
      pipe_barrier(PIPE_ALL);  // synchronize before free
      in_embedd_queue_.FreeTensor(input_embedd_local);
      in_embedd_pos_queue_.FreeTensor(input_pos_embedd_local);
    } else {
      DataCopy(inputx_local,
               inputx_global_[token_id * ALIGN_BY_TYPE(h_length_, sizeof(T), 32) + chunk_id * actual_chunk_size_],
               ALIGN_BY_TYPE(actual_elem, sizeof(T), 32));
    }
    if (is_residual_) {
      LocalTensor<T> inputy_local = in_queue_y_.template AllocTensor<T>();
      DataCopy(inputy_local,
               inputy_global_[token_id * ALIGN_BY_TYPE(h_length_, sizeof(T), 32) + chunk_id * actual_chunk_size_],
               ALIGN_BY_TYPE(actual_elem, sizeof(T), 32));
      in_queue_y_.template EnQue(inputy_local);
    }
    in_queue_x_.template EnQue(inputx_local);
  }
  __aicore__ inline void ComputeReduction(uint32_t chunk_id, uint32_t actual_elem) {
    uint32_t chunk_offset = chunk_id * actual_chunk_size_;
    LocalTensor<T> inputx_local = in_queue_x_.template DeQue<T>();
    LocalTensor<T> inputy_local;

    if (is_residual_) {
      LocalTensor<T> output_local = out_queue_.template AllocTensor<T>();
      if (is_bias_) {
        Add(inputx_local, inputx_local, bias_tmp_local_[chunk_offset], actual_elem);
      }
      inputy_local = in_queue_y_.template DeQue<T>();
      Add(inputx_local, inputx_local, inputy_local, actual_elem);
      DataCopy(output_local, inputx_local, ALIGN_BY_TYPE(actual_elem, sizeof(T), 32));
      out_queue_.template EnQue<T>(output_local);
    }
    if (is_embedding_) {
      LocalTensor<T> output_local = out_queue_.template AllocTensor<T>();
      DataCopy(output_local, inputx_local, ALIGN_BY_TYPE(actual_elem, sizeof(T), 32));
      out_queue_.template EnQue<T>(output_local);
    }
    DataCopy(x_tmp_local_[chunk_offset], inputx_local, ALIGN_BY_TYPE(actual_elem, sizeof(T), 32));
    // do sigma and sigma^2
    ReduceSum(reduce_output_tmp_local_, inputx_local, reduce_tmp_local_, actual_elem);
    reduce_sum_tmp_local_.SetValue(chunk_id, reduce_output_tmp_local_.GetValue(0));
    Mul(inputx_local, inputx_local, inputx_local, actual_elem);
    ReduceSum(reduce_output_tmp_local_, inputx_local, reduce_tmp_local_, actual_elem);
    reduce_sum_of_sqr_tmp_local_.SetValue(chunk_id, reduce_output_tmp_local_.GetValue(0));

    pipe_barrier(PIPE_ALL);  // synchronize before free
    in_queue_x_.FreeTensor(inputx_local);
    if (is_residual_) {
      in_queue_y_.FreeTensor(inputy_local);
    }
  }

  __aicore__ inline void ComputeMeanAndVar() {
    float sum = 0.0f;
    float sum_of_squares = 0.0f;
    if (chunk_num_ == 1) {
      sum = static_cast<float>(reduce_sum_tmp_local_.GetValue(0));
      sum_of_squares = static_cast<float>(reduce_sum_of_sqr_tmp_local_.GetValue(0));
    } else if (chunk_num_ == 2) {
      sum =
        static_cast<float>(reduce_sum_tmp_local_.GetValue(0)) + static_cast<float>(reduce_sum_tmp_local_.GetValue(1));
      sum_of_squares = static_cast<float>(reduce_sum_of_sqr_tmp_local_.GetValue(0)) +
                       static_cast<float>(reduce_sum_of_sqr_tmp_local_.GetValue(1));
    } else {
      ReduceSum(reduce_output_tmp_local_, reduce_sum_tmp_local_, reduce_tmp_local_, chunk_num_);
      sum = static_cast<float>(reduce_output_tmp_local_.GetValue(0));
      ReduceSum(reduce_output_tmp_local_, reduce_sum_of_sqr_tmp_local_, reduce_tmp_local_, chunk_num_);
      sum_of_squares = static_cast<float>(reduce_output_tmp_local_.GetValue(0));
    }

    float n = float_h_length_;
    float tmp = ((sum_of_squares - (sum * sum / n)) / n);
    var_ = (T)(1.0f / static_cast<float>(sqrt(tmp + static_cast<float>(epsilon_))));
    mean_ = (T)(0 - (sum / n));
  }

  __aicore__ inline void Normalize(uint32_t chunk_id, uint32_t actual_elem) {
    LocalTensor<T> output_norm_local = out_norm_queue_.template AllocTensor<T>();
    // normalize tensor
    Adds(output_norm_local, x_tmp_local_[chunk_id * actual_chunk_size_], mean_, actual_elem);
    Muls(output_norm_local, output_norm_local, var_, actual_elem);
    // Mul with Gamma and add bias
    Mul(output_norm_local, output_norm_local, gamma_tmp_local_, actual_elem);
    Add(output_norm_local, output_norm_local, beta_tmp_local_, actual_elem);
    out_norm_queue_.template EnQue<T>(output_norm_local);
  }

  __aicore__ inline void CopyOut(int offset, uint32_t actual_elem) {
    if (is_residual_ || is_embedding_) {
      LocalTensor<T> output_local_ = out_queue_.template DeQue<T>();
      DataCopy(output_global_[offset], output_local_, ALIGN_BY_TYPE(actual_elem, sizeof(T), 32));
      out_queue_.FreeTensor(output_local_);
    }
  }
  __aicore__ inline void CopyNormOut(int offset, uint32_t actual_elem, int token_id) {
    LocalTensor<T> output_norm_local_ = out_norm_queue_.template DeQue<T>();
    if (token_id != -1) {
      DataCopy(output_norm_global_[offset], output_norm_local_, ALIGN_BY_TYPE(actual_elem, sizeof(T), 32));
    }
    out_norm_queue_.FreeTensor(output_norm_local_);
  }

 private:
  GlobalTensor<T> inputx_global_;
  GlobalTensor<T> inputy_global_;
  GlobalTensor<int> input_ids_global_;
  GlobalTensor<int> input_pos_global_;
  GlobalTensor<T> gamma_global_;
  GlobalTensor<T> beta_global_;
  GlobalTensor<T> bias_global_;
  GlobalTensor<T> emmbeding_word_global_;
  GlobalTensor<T> emmbeding_pos_global_;
  GlobalTensor<T> output_global_;
  GlobalTensor<T> output_norm_global_;
  GlobalTensor<int> token_to_token_global_;

  LocalTensor<T> x_tmp_local_, reduce_tmp_local_;
  LocalTensor<T> reduce_sum_tmp_local_, reduce_sum_of_sqr_tmp_local_, reduce_output_tmp_local_;
  LocalTensor<T> gamma_tmp_local_, beta_tmp_local_, bias_tmp_local_;
  KernelVsl vsl_helper_;
  TPipe pipe_;
  TQue<QuePosition::VECIN, pipeSize> in_queue_x_;
  TQue<QuePosition::VECIN, pipeSize> in_queue_y_;
  TQue<QuePosition::VECIN, pipeSize> in_embedd_queue_;
  TQue<QuePosition::VECIN, pipeSize> in_embedd_pos_queue_;
  TQue<QuePosition::VECOUT, pipeSize> out_queue_;
  TQue<QuePosition::VECOUT, pipeSize> out_norm_queue_;

  TBuf<QuePosition::VECCALC> x_tmp_, in_tmp_x_;
  TBuf<QuePosition::VECCALC> reduce_tmp_;
  TBuf<QuePosition::VECCALC> reduce_sum_tmp_, reduce_sum_of_sqr_tmp_, reduce_output_tmp_;
  TBuf<QuePosition::VECIN> in_buf_gamma_, in_buf_beta_, in_buf_bias_;

  bool is_residual_;
  bool is_embedding_;
  bool is_bias_;
  bool is_moe_gather_;
  uint32_t token_per_block_;
  uint32_t total_token_;

  uint32_t chunk_num_;
  uint32_t actual_chunk_size_;
  int seq_length_;
  uint32_t h_length_;
  T epsilon_;
  float float_h_length_;
  T mean_;
  T var_;
};

template <uint32_t pipeSize, uint32_t chunkSize, typename T, bool isReuseSource = false>
__aicore__ void KernelLayernormOperator(GM_ADDR inputx_gm, GM_ADDR inputy_gm, GM_ADDR bias_gm, GM_ADDR gamma_gm,
                                        GM_ADDR beta_gm, GM_ADDR output_gm, GM_ADDR output_norm_gm,
                                        GM_ADDR input_ids_gm, GM_ADDR input_pos_gm, GM_ADDR emmbeding_word_gm,
                                        GM_ADDR emmbeding_pos_gm, uint32_t token_num, uint32_t total_token,
                                        uint32_t h_length, float float_h_length, T epsilon, uint32_t batch_size = 1,
                                        uint32_t v_length = 1, uint32_t seq_length = 1, GM_ADDR seq_len_gm = nullptr,
                                        GM_ADDR padding_offset_gm = nullptr, GM_ADDR mode_gm = nullptr,
                                        GM_ADDR token_to_token_gm = nullptr) {
  if (g_coreType == AIV) {
    KernelLayernorm<pipeSize, chunkSize, T, isReuseSource> op;
    op.Init(inputx_gm, inputy_gm, bias_gm, gamma_gm, beta_gm, output_gm, output_norm_gm, input_ids_gm, input_pos_gm,
            emmbeding_word_gm, emmbeding_pos_gm, token_num, total_token, h_length, float_h_length, epsilon, batch_size,
            v_length, seq_length, seq_len_gm, padding_offset_gm, mode_gm, token_to_token_gm);
    op.Process();
  }
}
}  // namespace mindspore::ascend_native
#endif
