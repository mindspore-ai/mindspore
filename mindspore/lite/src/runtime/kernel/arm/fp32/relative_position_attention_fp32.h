/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_RELATIVE_POSITION_ATTENTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_RELATIVE_POSITION_ATTENTION_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32/attention_fp32.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore::kernel {
// inputs: 0:Q 1:K 2:V 3:P 4:WQ 5:WK 6:WV 7:WP 8:PU 9:PV 10:WO 11:BQ 12:BK 13:BV 14:BO 15:output
// if use_bias == true: has BQ BK BV BO inputs
class RelativePositionAttentionCPUKernel : public InnerKernel {
 public:
  RelativePositionAttentionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<RelativePositionAttentionParameter *>(op_parameter_);
  }

  ~RelativePositionAttentionCPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

 private:
  // check inputs
  int CheckInputs();
  int CheckWeights();
  int CheckBiases();
  // prepare const inputs
  int PrepareParam();
  int PrepareWeights();
  int PrepareBiases();
  // pack variable inputs
  int PackRunBuffersInputs();
  int PackRunBuffersEmbeddings(int batch, int num_heads, int depth);
  int PackRunBuffersLogits(int batch, int num_heads, int depth);
  int PackRunBuffersAttention(int batch, int num_heads, int depth);
  int PackRunBuffers();
  // free packed data
  void FreePackedRunBuffers();
  void FreePackedWeights();
  void FreePackedBiases();
  void FreeAllPackData();

 private:
  // input tensors
  lite::Tensor *input_q_tensor_ = nullptr;
  lite::Tensor *input_k_tensor_ = nullptr;
  lite::Tensor *input_v_tensor_ = nullptr;
  lite::Tensor *input_p_tensor_ = nullptr;
  lite::Tensor *weight_q_tensor_ = nullptr;
  lite::Tensor *weight_k_tensor_ = nullptr;
  lite::Tensor *weight_v_tensor_ = nullptr;
  lite::Tensor *weight_p_tensor_ = nullptr;
  lite::Tensor *weight_o_tensor_ = nullptr;
  lite::Tensor *pos_u_tensor_ = nullptr;
  lite::Tensor *pos_v_tensor_ = nullptr;
  lite::Tensor *bias_q_tensor_ = nullptr;
  lite::Tensor *bias_k_tensor_ = nullptr;
  lite::Tensor *bias_v_tensor_ = nullptr;
  lite::Tensor *bias_o_tensor_ = nullptr;

  // packed data
  Matrix input_q_mat_{};   // [1, q_seq, d_model]
  Matrix input_k_mat_{};   // [1, k_seq, d_model]
  Matrix input_v_mat_{};   // [1, v_seq, d_model] // v_seq = k_seq
  Matrix input_p_mat_{};   // [1, p_seq, d_model] // p_seq/2 = k_seq
  Matrix weight_q_mat_{};  // [d_model, d_model]
  Matrix weight_k_mat_{};  // [d_model, d_model]
  Matrix weight_v_mat_{};  // [d_model, d_model]
  Matrix weight_p_mat_{};  // [d_model, d_model]
  Matrix weight_o_mat_{};  // [d_model, d_model]
  Matrix bias_q_mat_{};    // [d_model]
  Matrix bias_k_mat_{};    // [d_model]
  Matrix bias_v_mat_{};    // [d_model]
  Matrix bias_o_mat_{};    // [d_model]
  Matrix pos_u_mat_{};     // [num_heads, depth] // num_heads * depth = d_model
  Matrix pos_v_mat_{};     // [num_heads, depth]

  // run buffer
  // q * wq
  // [1, q_seq, d_model] reshaped to [1, q_seq, num_heads, depth]
  Matrix q2wq_mat_{};
  // (q2wq_mat_ + pu) or (q2wq_mat_ + pv)
  // [1, q_seq, num_heads, depth]
  Matrix q2wq_with_pos_mat_{};
  // transpose from q_with_pos_mat_, perm = [0,2,1,3], [1, num_heads, q_seq, depth]
  Matrix q2wq_with_pu_trans_mat_{};
  // transpose from q_with_pos_mat_, perm = [0,2,1,3], [1, num_heads, q_seq, depth]
  Matrix q2wq_with_pv_trans_mat_{};

  // k * wk
  // [1, k_seq, d_model] reshaped to [1, k_seq, num_heads, depth]
  Matrix k2wk_mat_{};
  // transpose from k2wk_mat_, perm = [0,2,3,1], [1, num_heads, depth, k_seq]
  Matrix k2wk_trans_mat_{};

  // p * wp
  // [1, p_seq, d_model] reshaped to [1, p_seq, num_heads, depth]
  Matrix p2wp_mat_{};
  // transpose from p2wp_mat_, perm = [0,2,3,1], [1, num_heads, depth, p_seq]
  Matrix p2wp_trans_mat_{};

  // v * wv
  // [1, v_seq, d_model] reshaped to [1, v_seq, num_heads, depth]
  Matrix v2wv_mat_{};
  // transpose from v2wv_mat_, perm = [0,2,1,3], [1, num_heads, v_seq, depth]
  Matrix v2wv_trans_mat_{};

  // q_with_pu_trans_mat_ * k2wk_trans_mat_
  // [1, num_heads, q_seq, k_seq]
  Matrix logits_with_u_mat_{};
  // q2wq_with_pv_trans_mat_ * p2wp_trans_mat_
  // [1, num_heads, q_seq, p_seq]
  Matrix logits_with_v_mat_{};
  // relative shifted from logits_with_v_mat_, [1, num_heads, q_seq, p_seq / 2]
  Matrix logits_with_v_shifted_mat_{};
  // logits_with_u_mat_ + logits_with_v_shifted_mat_
  // [1, num_heads, q_seq, k_seq]
  Matrix logits_mat_{};
  // softmax(logits_mat_)
  // [1, num_heads, q_seq, k_seq]
  Matrix softmax_mat_{};
  // softmax_mat_ * v2wv_trans_mat_
  // [1, num_heads, q_seq, depth]
  Matrix logits2v_mat_{};
  // transpose from logits2v_mat_, perm = [0,2,1,3], [1, q_seq, num_heads, depth] reshaped to [1, q_seq, d_model]
  Matrix logits2v_trans_mat_{};
  // logits2v_trans_mat_ * o
  // [1, q_seq, d_model]
  Matrix output_mat_{};

  // pad from logits_with_v_mat_ each matrix, [q_seq, p_seq + 1]
  Matrix logits_with_v_pad_mat_{};

  RelativePositionAttentionParameter *param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_RELATIVE_POSITION_ATTENTION_H_
