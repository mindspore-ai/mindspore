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

#include "src/runtime/kernel/arm/fp32/relative_position_attention_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/matmul_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
RelativePositionAttentionCPUKernel::~RelativePositionAttentionCPUKernel() { FreeAllPackData(); }

namespace {
constexpr int kActivationTensorShapeSize = 3;
constexpr int kActivationTensorBatch = 1;
constexpr int kTensorShapeBatchIndex = 0;
constexpr int k3DimsLeftMatrixDeepIndex = 2;
constexpr int kRightMatrixDeepIndex = 0;
constexpr int kRelativePositionHasBiasInputSize = 15;

bool AttentionActivationTensorCheck(lite::Tensor *tensor) {
  if (tensor == nullptr || tensor->data_type() != kNumberTypeFloat32 ||
      tensor->shape().size() != kActivationTensorShapeSize ||
      tensor->shape().at(kTensorShapeBatchIndex) != kActivationTensorBatch) {
    return false;
  }
  return true;
}
}  // namespace

int RelativePositionAttentionCPUKernel::CheckInputs() {
  input_q_tensor_ = this->in_tensors_.at(0);
  if (!AttentionActivationTensorCheck(input_q_tensor_)) {
    MS_LOG(ERROR) << "input_q is abnormal.";
    return RET_ERROR;
  }
  input_k_tensor_ = this->in_tensors_.at(1);
  if (!AttentionActivationTensorCheck(input_k_tensor_)) {
    MS_LOG(ERROR) << "input_k is abnormal.";
    return RET_ERROR;
  }
  input_v_tensor_ = this->in_tensors_.at(2);
  if (!AttentionActivationTensorCheck(input_v_tensor_)) {
    MS_LOG(ERROR) << "input_v is abnormal.";
    return RET_ERROR;
  }
  input_p_tensor_ = this->in_tensors_.at(3);
  if (!AttentionActivationTensorCheck(input_p_tensor_)) {
    MS_LOG(ERROR) << "input_p is abnormal.";
    return RET_ERROR;
  }
  // Sequence length Q / 2 should be equal to sequence length of K
  if (input_p_tensor_->shape().at(1) / 2 != input_k_tensor_->shape().at(1)) {
    MS_LOG(ERROR) << "Sequence length of input_p / 2 != sequence length of input_k";
    return RET_ERROR;
  }
  // Sequence length of V should be equal to sequence length of K
  if (input_v_tensor_->shape().at(1) != input_k_tensor_->shape().at(1)) {
    MS_LOG(ERROR) << "Sequence length of input_v != sequence length of input_k";
    return RET_ERROR;
  }
  if (input_q_tensor_->shape().at(k3DimsLeftMatrixDeepIndex) != weight_q_tensor_->shape().at(kRightMatrixDeepIndex)) {
    MS_LOG(ERROR) << "Shapes of input_q and weight_q are mismatched.";
    return RET_ERROR;
  }
  if (input_k_tensor_->shape().at(k3DimsLeftMatrixDeepIndex) != weight_k_tensor_->shape().at(kRightMatrixDeepIndex)) {
    MS_LOG(ERROR) << "Shapes of input_k and weight_k are mismatched.";
    return RET_ERROR;
  }
  if (input_v_tensor_->shape().at(k3DimsLeftMatrixDeepIndex) != weight_v_tensor_->shape().at(kRightMatrixDeepIndex)) {
    MS_LOG(ERROR) << "Shapes of input_v and weight_v are mismatched.";
    return RET_ERROR;
  }
  if (input_v_tensor_->shape().at(k3DimsLeftMatrixDeepIndex) != weight_o_tensor_->shape().at(kRightMatrixDeepIndex)) {
    MS_LOG(ERROR) << "Shapes of input_v and weight_o are mismatched.";
    return RET_ERROR;
  }
  return RET_OK;
}

namespace {
constexpr int kWeightTensorShapeSize = 2;
bool AttentionWeightTensorCheck(lite::Tensor *tensor) {
  if (tensor == nullptr || !tensor->IsConst() || tensor->data_type() != kNumberTypeFloat32 ||
      tensor->shape().size() != kWeightTensorShapeSize) {
    return false;
  }
  return true;
}
}  // namespace

int RelativePositionAttentionCPUKernel::CheckWeights() {
  weight_q_tensor_ = this->in_tensors_.at(4);
  if (!AttentionWeightTensorCheck(weight_q_tensor_)) {
    MS_LOG(ERROR) << "weight_q is abnormal.";
    return RET_ERROR;
  }
  weight_k_tensor_ = this->in_tensors_.at(5);
  if (!AttentionWeightTensorCheck(weight_k_tensor_)) {
    MS_LOG(ERROR) << "weight_k is abnormal.";
    return RET_ERROR;
  }
  weight_v_tensor_ = this->in_tensors_.at(6);
  if (!AttentionWeightTensorCheck(weight_v_tensor_)) {
    MS_LOG(ERROR) << "weight_v is abnormal.";
    return RET_ERROR;
  }
  weight_p_tensor_ = this->in_tensors_.at(7);
  if (!AttentionWeightTensorCheck(weight_p_tensor_)) {
    MS_LOG(ERROR) << "weight_p is abnormal.";
    return RET_ERROR;
  }
  pos_u_tensor_ = this->in_tensors_.at(8);
  if (!AttentionWeightTensorCheck(pos_u_tensor_)) {
    MS_LOG(ERROR) << "pos_u is abnormal.";
    return RET_ERROR;
  }
  pos_v_tensor_ = this->in_tensors_.at(9);
  if (!AttentionWeightTensorCheck(pos_v_tensor_)) {
    MS_LOG(ERROR) << "pos_v is abnormal.";
    return RET_ERROR;
  }
  weight_o_tensor_ = this->in_tensors_.at(10);
  if (!AttentionWeightTensorCheck(weight_o_tensor_)) {
    MS_LOG(ERROR) << "weight_o is abnormal.";
    return RET_ERROR;
  }
  if (weight_q_tensor_->shape().at(1) != weight_k_tensor_->shape().at(1)) {
    MS_LOG(ERROR) << "Shapes of weight_q and weight_k are mismatched.";
    return RET_ERROR;
  }
  return RET_OK;
}

namespace {
bool AttentionBiasTensorCheck(lite::Tensor *tensor) {
  if (tensor == nullptr || !tensor->IsConst() || tensor->data_type() != kNumberTypeFloat32 ||
      tensor->shape().size() != 1) {
    return false;
  }
  return true;
}
}  // namespace

int RelativePositionAttentionCPUKernel::CheckBiases() {
  if (this->in_tensors_.size() == kRelativePositionHasBiasInputSize) {
    param_->use_bias_ = true;
  }
  if (!param_->use_bias_) {
    return RET_OK;
  }
  bias_q_tensor_ = this->in_tensors_.at(11);
  bias_k_tensor_ = this->in_tensors_.at(12);
  bias_v_tensor_ = this->in_tensors_.at(13);
  bias_o_tensor_ = this->in_tensors_.at(14);
  if (!AttentionBiasTensorCheck(bias_q_tensor_)) {
    MS_LOG(ERROR) << "bias_q is abnormal.";
    return RET_ERROR;
  }
  if (!AttentionBiasTensorCheck(bias_k_tensor_)) {
    MS_LOG(ERROR) << "bias_k is abnormal.";
    return RET_ERROR;
  }
  if (!AttentionBiasTensorCheck(bias_v_tensor_)) {
    MS_LOG(ERROR) << "bias_v is abnormal.";
    return RET_ERROR;
  }
  if (!AttentionBiasTensorCheck(bias_o_tensor_)) {
    MS_LOG(ERROR) << "bias is abnormal.";
    return RET_ERROR;
  }
  if (bias_o_tensor_->shape().at(0) != weight_o_tensor_->shape().at(1)) {
    MS_LOG(ERROR) << "Shapes of weight_o and bias_o are mismatched.";
    return RET_ERROR;
  }
  if (bias_q_tensor_->shape().at(0) != weight_q_tensor_->shape().at(1)) {
    MS_LOG(ERROR) << "Shapes of weight_q and bias_q are mismatched.";
    return RET_ERROR;
  }
  if (bias_k_tensor_->shape().at(0) != weight_k_tensor_->shape().at(1)) {
    MS_LOG(ERROR) << "Shapes of weight_k and bias_k are mismatched.";
    return RET_ERROR;
  }
  if (bias_v_tensor_->shape().at(0) != weight_v_tensor_->shape().at(1)) {
    MS_LOG(ERROR) << "Shapes of weight_v and bias_v are mismatched.";
    return RET_ERROR;
  }
  return RET_OK;
}

namespace {
constexpr int kQSeqIndexInQ = 1;
constexpr int kKSeqIndexInK = 1;
constexpr int kVSeqIndexInV = 1;
constexpr int kPSeqIndexInP = 1;
constexpr int kNumHeadIndexInPositionU = 0;
constexpr int kDModelIndexInQ = 2;
}  // namespace

int RelativePositionAttentionCPUKernel::PrepareParam() {
#ifdef ENABLE_AVX
  param_->row_tile_ = C6NUM;
  param_->col_tile_ = C16NUM;
  param_->bias_tile_ = C16NUM;
#elif defined(ENABLE_ARM32)
  param_->row_tile_ = C12NUM;
  param_->col_tile_ = C4NUM;
  param_->bias_tile_ = C4NUM;
#elif defined(ENABLE_SSE)
  param_->row_tile_ = C4NUM;
  param_->col_tile_ = C8NUM;
  param_->bias_tile_ = C8NUM;
#else
  param_->row_tile_ = C12NUM;
  param_->col_tile_ = C8NUM;
  param_->bias_tile_ = C8NUM;
#endif
  param_->num_heads_ = pos_u_tensor_->shape().at(kNumHeadIndexInPositionU);
  param_->batch_ = input_q_tensor_->shape().at(kTensorShapeBatchIndex);
  param_->d_model_ = input_q_tensor_->shape().at(kDModelIndexInQ);
  param_->q_seq_ = input_q_tensor_->shape().at(kQSeqIndexInQ);
  param_->k_seq_ = input_k_tensor_->shape().at(kKSeqIndexInK);
  param_->v_seq_ = input_v_tensor_->shape().at(kVSeqIndexInV);
  param_->p_seq_ = input_p_tensor_->shape().at(kPSeqIndexInP);
  if (param_->num_heads_ <= 1) {
    MS_LOG(ERROR) << "RelativePositionAttention only support multi-heads.";
    return RET_ERROR;
  }
  if (param_->d_model_ % param_->num_heads_ != 0) {
    MS_LOG(ERROR) << "D_model should be a integer multiple of num_heads.";
    return RET_ERROR;
  }
  return RET_OK;
}

namespace {
constexpr int kLeftMatrixBatchDimIndex = 0;
constexpr int kLeftMatrixRowDimIndex = 1;
constexpr int kLeftMatrixColDimIndex = 2;
inline int PackLeftTensor(const lite::Tensor &tensor, Matrix *matrix, int row_tile, const AllocatorPtr &allocator) {
  MS_ASSERT(matrix != nullptr);
  MS_ASSERT(allocator != nullptr);
  MS_ASSERT(matrix->data_ == nullptr);
  matrix->data_ = reinterpret_cast<float *>(tensor.data());
  matrix->is_transpose_ = false;
  // Left tensor is in [batch, row, col] shape
  matrix->batch_ = tensor.shape().at(kLeftMatrixBatchDimIndex);
  matrix->row_ = tensor.shape().at(kLeftMatrixRowDimIndex);
  matrix->col_ = tensor.shape().at(kLeftMatrixColDimIndex);
  auto size = LeftMatrixPackElementSize(matrix, row_tile) * sizeof(float);
  MS_ASSERT(size != 0);
  matrix->packed_data_ = reinterpret_cast<float *>(allocator->Malloc(size));
  return PackLeftMatrix(matrix, row_tile);
}

inline int MallocLeftTensor(Matrix *mat, int row_tile, const AllocatorPtr &allocator, bool need_pack_data = true) {
  MS_ASSERT(mat != nullptr);
  MS_ASSERT(allocator != nullptr);
  MS_ASSERT(mat->data_ == nullptr);
  mat->data_ = reinterpret_cast<float *>(allocator->Malloc(mat->batch_ * mat->row_ * mat->col_ * sizeof(float)));
  if (mat->data_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  if (need_pack_data) {
    auto size = LeftMatrixPackElementSize(mat, row_tile) * sizeof(float);
    MS_ASSERT(size != 0);
    mat->packed_data_ = reinterpret_cast<float *>(allocator->Malloc(size));
    if (mat->packed_data_ == nullptr) {
      return RET_MEMORY_FAILED;
    }
  }
  return RET_OK;
}

inline int MallocRightTensor(Matrix *mat, int col_tile, const AllocatorPtr &allocator, bool need_pack_data = true) {
  MS_ASSERT(mat != nullptr);
  MS_ASSERT(allocator != nullptr);
  MS_ASSERT(mat->data_ == nullptr);
  mat->data_ = reinterpret_cast<float *>(allocator->Malloc(mat->batch_ * mat->row_ * mat->col_ * sizeof(float)));
  if (mat->data_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  if (need_pack_data) {
    auto size = RightMatrixPackElementSize(mat, col_tile) * sizeof(float);
    MS_ASSERT(size != 0);
    mat->packed_data_ = reinterpret_cast<float *>(allocator->Malloc(size));
    if (mat->packed_data_ == nullptr) {
      return RET_MEMORY_FAILED;
    }
  }
  return RET_OK;
}

inline int PackRightTensor(const lite::Tensor &tensor, Matrix *matrix, int col_tile) {
  MS_ASSERT(matrix != nullptr);
  MS_ASSERT(matrix->data_ == nullptr);
  matrix->data_ = reinterpret_cast<float *>(tensor.data());
  matrix->batch_ = 1;
  matrix->is_transpose_ = false;
  matrix->row_ = tensor.shape().at(0);
  matrix->col_ = tensor.shape().at(1);
  return PackRightMatrix(matrix, col_tile);
}

inline int PackBiasTensor(const lite::Tensor &tensor, Matrix *matrix, int bias_tile) {
  MS_ASSERT(matrix != nullptr);
  MS_ASSERT(matrix->data_ == nullptr);
  matrix->data_ = reinterpret_cast<float *>(tensor.data());
  matrix->batch_ = 1;
  matrix->is_transpose_ = false;
  matrix->row_ = 1;
  matrix->col_ = tensor.shape().at(0);
  return PackAttentionBias(matrix, bias_tile);
}

inline void PackPositionTensor(const lite::Tensor &tensor, Matrix *matrix) {
  MS_ASSERT(matrix != nullptr);
  MS_ASSERT(matrix->data_ == nullptr);
  matrix->batch_ = 1;
  matrix->is_transpose_ = false;
  matrix->row_ = matrix->packed_row_ = 1;
  matrix->col_ = matrix->packed_col_ = tensor.shape().at(0) * tensor.shape().at(1);
  matrix->data_ = matrix->packed_data_ = reinterpret_cast<float *>(tensor.data());
}
}  // namespace

int RelativePositionAttentionCPUKernel::PrepareWeights() {
  FreePackedWeights();

  auto ret = PackRightTensor(*weight_q_tensor_, &weight_q_mat_, param_->col_tile_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack weight Q failed";
    return RET_ERROR;
  }
  ret = PackRightTensor(*weight_k_tensor_, &weight_k_mat_, param_->col_tile_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack weight K failed";
    return RET_ERROR;
  }
  ret = PackRightTensor(*weight_v_tensor_, &weight_v_mat_, param_->col_tile_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack weight V failed";
    return RET_ERROR;
  }
  ret = PackRightTensor(*weight_p_tensor_, &weight_p_mat_, param_->col_tile_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack weight P failed";
    return RET_ERROR;
  }
  ret = PackRightTensor(*weight_o_tensor_, &weight_o_mat_, param_->col_tile_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack weight O failed";
    return RET_ERROR;
  }
  return RET_OK;
}
// inputs: 0:Q 1:K 2:V 3:P 4:WQ 5:WK 6:WV 7:WP 8:PU 9:PV 10:WO 11:BQ 12:BK 13:BV 14:BO
// if use_bias == true: has BQ BK BV BO inputs
int RelativePositionAttentionCPUKernel::PrepareBiases() {
  if (!param_->use_bias_) {
    return RET_OK;
  }
  FreePackedBiases();

  auto ret = PackBiasTensor(*bias_q_tensor_, &bias_q_mat_, param_->bias_tile_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack bias Q failed";
    return RET_ERROR;
  }
  ret = PackBiasTensor(*bias_k_tensor_, &bias_k_mat_, param_->bias_tile_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack bias K failed";
    return RET_ERROR;
  }
  ret = PackBiasTensor(*bias_v_tensor_, &bias_v_mat_, param_->bias_tile_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack bias V failed";
    return RET_ERROR;
  }
  ret = PackBiasTensor(*bias_o_tensor_, &bias_o_mat_, param_->bias_tile_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack bias O failed";
    return RET_ERROR;
  }
  PackPositionTensor(*pos_u_tensor_, &pos_u_mat_);
  PackPositionTensor(*pos_v_tensor_, &pos_v_mat_);
  return RET_OK;
}

int RelativePositionAttentionCPUKernel::PackRunBuffersInputs() {
  MS_ASSERT(ms_context_ != nullptr && ms_context_->allocator != nullptr);
  if (input_q_mat_.data_ != nullptr || input_q_mat_.packed_data_ != nullptr || input_k_mat_.data_ != nullptr ||
      input_k_mat_.packed_data_ != nullptr || input_v_mat_.data_ != nullptr || input_v_mat_.packed_data_ != nullptr ||
      input_p_mat_.data_ != nullptr || input_p_mat_.packed_data_ != nullptr) {
    MS_LOG(ERROR) << "Run buffer data should not be packed.";
    return RET_ERROR;
  }
  auto input_q_tensor_data = reinterpret_cast<float *>(input_q_tensor_->data());
  auto input_k_tensor_data = reinterpret_cast<float *>(input_k_tensor_->data());
  auto input_v_tensor_data = reinterpret_cast<float *>(input_v_tensor_->data());
  auto input_p_tensor_data = reinterpret_cast<float *>(input_p_tensor_->data());
  if (input_q_tensor_data == nullptr || input_k_tensor_data == nullptr || input_v_tensor_data == nullptr ||
      input_p_tensor_data == nullptr) {
    MS_LOG(ERROR) << "Input data is nullptr.";
    return RET_ERROR;
  }

  auto ret = PackLeftTensor(*input_q_tensor_, &input_q_mat_, param_->row_tile_, ms_context_->allocator);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack input Q failed";
    return RET_ERROR;
  }
  ret = PackLeftTensor(*input_k_tensor_, &input_k_mat_, param_->row_tile_, ms_context_->allocator);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack input K failed";
    return RET_ERROR;
  }
  ret = PackLeftTensor(*input_v_tensor_, &input_v_mat_, param_->row_tile_, ms_context_->allocator);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack input V failed";
    return RET_ERROR;
  }
  ret = PackLeftTensor(*input_p_tensor_, &input_p_mat_, param_->row_tile_, ms_context_->allocator);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "Pack input P failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int RelativePositionAttentionCPUKernel::PackRunBuffersEmbeddings(int batch, int num_heads, int depth) {
  MS_ASSERT(ms_context_ != nullptr && ms_context_->allocator != nullptr);
  // since &q2wq_mat_ can not be nullptr, so we ignore result of function
  (void)InitMatrix(&q2wq_mat_, batch * param_->q_seq_, num_heads, depth, false);
  auto ret = MallocLeftTensor(&q2wq_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc q2wq buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&q2wq_with_pos_mat_, batch * param_->q_seq_, num_heads, depth, false);
  ret = MallocLeftTensor(&q2wq_with_pos_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc q2wq_with_pos buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&q2wq_with_pu_trans_mat_, batch * num_heads, param_->q_seq_, depth, false);
  ret = MallocLeftTensor(&q2wq_with_pu_trans_mat_, param_->row_tile_, ms_context_->allocator);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc q2wq_with_pu_trans buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&q2wq_with_pv_trans_mat_, batch * num_heads, param_->q_seq_, depth, false);
  ret = MallocLeftTensor(&q2wq_with_pv_trans_mat_, param_->row_tile_, ms_context_->allocator);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc q2wq_with_pv_trans buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&k2wk_mat_, batch * param_->k_seq_, num_heads, depth, false);
  ret = MallocLeftTensor(&k2wk_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc k2wk buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&k2wk_trans_mat_, batch * num_heads, depth, param_->k_seq_, false);
  ret = MallocRightTensor(&k2wk_trans_mat_, param_->col_tile_, ms_context_->allocator);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc k2wk_trans result buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&p2wp_mat_, batch * param_->p_seq_, num_heads, depth, false);
  ret = MallocLeftTensor(&p2wp_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc p2wp buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&p2wp_trans_mat_, batch * num_heads, depth, param_->p_seq_, false);
  ret = MallocRightTensor(&p2wp_trans_mat_, param_->col_tile_, ms_context_->allocator);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc p2wp_trans result buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&v2wv_mat_, batch * param_->v_seq_, num_heads, depth, false);
  ret = MallocLeftTensor(&v2wv_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc v2wv buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&v2wv_trans_mat_, batch * num_heads, param_->v_seq_, depth, false);
  ret = MallocRightTensor(&v2wv_trans_mat_, param_->col_tile_, ms_context_->allocator);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc v2wv_trans buffer failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int RelativePositionAttentionCPUKernel::PackRunBuffersLogits(int batch, int num_heads, int depth) {
  MS_ASSERT(ms_context_ != nullptr && ms_context_->allocator != nullptr);
  // [batch, num_heads, q_seq_len, k_seq_len] // don't need pack
  (void)InitMatrix(&logits_with_u_mat_, batch * num_heads, param_->q_seq_, param_->k_seq_, false);
  auto ret = MallocLeftTensor(&logits_with_u_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc logits_with_u buffer failed";
    return RET_ERROR;
  }
  // [batch, num_heads, q_seq_len, p_seq_len] // don't need pack
  (void)InitMatrix(&logits_with_v_mat_, batch * num_heads, param_->q_seq_, param_->p_seq_, false);
  ret = MallocLeftTensor(&logits_with_v_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc logits_with_v buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&logits_with_v_pad_mat_, 1, param_->q_seq_, param_->p_seq_ + 1, false);
  ret = MallocLeftTensor(&logits_with_v_pad_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc logits_with_v_pad buffer failed";
    return RET_ERROR;
  }
  // relative shift output shape is [batch * num_heads, q_seq, p_seq / 2]
  (void)InitMatrix(&logits_with_v_shifted_mat_, batch * num_heads, param_->q_seq_, param_->p_seq_ / 2, false);
  ret = MallocLeftTensor(&logits_with_v_shifted_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc logits_with_v_shifted buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&logits_mat_, batch * num_heads, param_->q_seq_, param_->k_seq_, false);
  ret = MallocLeftTensor(&logits_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc logits buffer failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int RelativePositionAttentionCPUKernel::PackRunBuffersAttention(int batch, int num_heads, int depth) {
  MS_ASSERT(ms_context_ != nullptr && ms_context_->allocator != nullptr);
  auto output_tensor = this->out_tensors_.at(0);

  (void)InitMatrix(&softmax_mat_, batch * num_heads, param_->q_seq_, param_->k_seq_, false);
  auto ret = MallocLeftTensor(&softmax_mat_, param_->row_tile_, ms_context_->allocator);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc softmax buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&logits2v_mat_, batch * num_heads, param_->q_seq_, depth, false);
  ret = MallocLeftTensor(&logits2v_mat_, param_->row_tile_, ms_context_->allocator, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc logits2v buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&logits2v_trans_mat_, batch * param_->q_seq_, num_heads, depth, false);
  ret = MallocLeftTensor(&logits2v_trans_mat_, param_->row_tile_, ms_context_->allocator);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc logits2v_trans buffer failed";
    return RET_ERROR;
  }
  (void)InitMatrix(&output_mat_, batch, param_->q_seq_, param_->d_model_, false);
  output_mat_.data_ = reinterpret_cast<float *>(output_tensor->data());
  if (output_mat_.data_ == nullptr) {
    MS_LOG(ERROR) << "Output buffer is not malloced";
    return RET_ERROR;
  }
  return RET_OK;
}

int RelativePositionAttentionCPUKernel::PackRunBuffers() {
  if (ms_context_ == nullptr || ms_context_->allocator == nullptr) {
    MS_LOG(ERROR) << "Allocator is nullptr.";
    return RET_ERROR;
  }
  int batch = param_->batch_;
  int num_heads = param_->num_heads_;
  int d_model = param_->d_model_;
  int depth = d_model / num_heads;
  auto ret = PackRunBuffersInputs();
  if (ret != RET_OK) {
    return ret;
  }
  ret = PackRunBuffersEmbeddings(batch, num_heads, depth);
  if (ret != RET_OK) {
    return ret;
  }
  ret = PackRunBuffersLogits(batch, num_heads, depth);
  if (ret != RET_OK) {
    return ret;
  }
  ret = PackRunBuffersAttention(batch, num_heads, depth);
  if (ret != RET_OK) {
    return ret;
  }
  return RET_OK;
}

namespace {
inline void FreeData(float **ptr, AllocatorPtr allocator = nullptr) {
  if (ptr == nullptr) {
    return;
  }
  if (allocator == nullptr) {
    free(*ptr);
  } else {
    allocator->Free(*ptr);
  }
  *ptr = nullptr;
}
}  // namespace

void RelativePositionAttentionCPUKernel::FreePackedWeights() {
  FreeData(&(weight_q_mat_.packed_data_));
  FreeData(&(weight_k_mat_.packed_data_));
  FreeData(&(weight_v_mat_.packed_data_));
  FreeData(&(weight_p_mat_.packed_data_));
  FreeData(&(weight_o_mat_.packed_data_));
}

void RelativePositionAttentionCPUKernel::FreePackedBiases() {
  FreeData(&(bias_q_mat_.packed_data_));
  FreeData(&(bias_k_mat_.packed_data_));
  FreeData(&(bias_v_mat_.packed_data_));
  FreeData(&(bias_o_mat_.packed_data_));
}

void RelativePositionAttentionCPUKernel::FreePackedRunBuffers() {
  if (ms_context_ == nullptr || ms_context_->allocator == nullptr) {
    return;
  }
  auto allocator = ms_context_->allocator;
  FreeData(&(input_q_mat_.packed_data_), allocator);
  FreeData(&(input_k_mat_.packed_data_), allocator);
  FreeData(&(input_v_mat_.packed_data_), allocator);
  FreeData(&(input_p_mat_.packed_data_), allocator);

  FreeData(&(q2wq_mat_.data_), allocator);
  FreeData(&(q2wq_with_pos_mat_.data_), allocator);
  FreeData(&(q2wq_with_pu_trans_mat_.data_), allocator);
  FreeData(&(q2wq_with_pu_trans_mat_.packed_data_), allocator);
  FreeData(&(q2wq_with_pv_trans_mat_.data_), allocator);
  FreeData(&(q2wq_with_pv_trans_mat_.packed_data_), allocator);

  FreeData(&(k2wk_mat_.data_), allocator);
  FreeData(&(k2wk_trans_mat_.data_), allocator);
  FreeData(&(k2wk_trans_mat_.packed_data_), allocator);

  FreeData(&(p2wp_mat_.data_), allocator);
  FreeData(&(p2wp_trans_mat_.data_), allocator);
  FreeData(&(p2wp_trans_mat_.packed_data_), allocator);

  FreeData(&(v2wv_mat_.data_), allocator);
  FreeData(&(v2wv_trans_mat_.data_), allocator);
  FreeData(&(v2wv_trans_mat_.packed_data_), allocator);

  FreeData(&(logits_with_u_mat_.data_), allocator);
  FreeData(&(logits_with_v_mat_.data_), allocator);
  FreeData(&(logits_with_v_pad_mat_.data_), allocator);
  FreeData(&(logits_with_v_shifted_mat_.data_), allocator);
  FreeData(&(logits_mat_.data_), allocator);
  FreeData(&(softmax_mat_.data_), allocator);
  FreeData(&(softmax_mat_.packed_data_), allocator);
  FreeData(&(logits2v_mat_.data_), allocator);
  FreeData(&(logits2v_trans_mat_.data_), allocator);
  FreeData(&(logits2v_trans_mat_.packed_data_), allocator);
}

void RelativePositionAttentionCPUKernel::FreeAllPackData() {
  FreePackedRunBuffers();
  FreePackedWeights();
  FreePackedBiases();
}

int RelativePositionAttentionCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_11D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto ret = CheckWeights();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CheckWeights failed.";
    return RET_ERROR;
  }
  ret = CheckBiases();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CheckBiases failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int RelativePositionAttentionCPUKernel::ReSize() {
  auto ret = CheckInputs();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CheckInputs failed.";
    return RET_ERROR;
  }

  ret = PrepareParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareParam error.";
    return RET_ERROR;
  }

  ret = PrepareWeights();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareWeights error.";
    return RET_ERROR;
  }

  ret = PrepareBiases();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareBiases error.";
    return RET_ERROR;
  }
  return RET_OK;
}

int RelativePositionAttentionCPUKernel::Run() {
  auto ret = PackRunBuffers();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PackRunBuffers error.";
    return RET_ERROR;
  }

  QWithPosition(param_, &input_q_mat_, &weight_q_mat_, &bias_q_mat_, &q2wq_mat_, &pos_u_mat_, &pos_v_mat_,
                &q2wq_with_pos_mat_, &q2wq_with_pu_trans_mat_, &q2wq_with_pv_trans_mat_);
  KMulWeightK(param_, &input_k_mat_, &weight_k_mat_, &bias_k_mat_, &k2wk_mat_, &k2wk_trans_mat_);
  VMulWeightV(param_, &input_v_mat_, &weight_v_mat_, &bias_v_mat_, &v2wv_mat_, &v2wv_trans_mat_);
  PMulWeightP(param_, &input_p_mat_, &weight_p_mat_, &p2wp_mat_, &p2wp_trans_mat_);
  CalculateLogits(param_, &q2wq_with_pu_trans_mat_, &q2wq_with_pv_trans_mat_, &k2wk_trans_mat_, &p2wp_trans_mat_,
                  &logits_with_u_mat_, &logits_with_v_mat_, &logits_with_v_pad_mat_, &logits_with_v_shifted_mat_,
                  &logits_mat_);
  RelPosAttention(param_, &logits_mat_, &softmax_mat_, &v2wv_trans_mat_, &logits2v_mat_, &logits2v_trans_mat_,
                  &weight_o_mat_, &bias_o_mat_, &output_mat_);
  FreePackedRunBuffers();
  return RET_OK;
}
}  // namespace mindspore::kernel
