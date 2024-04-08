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

#include "extendrt/delegate/ascend_native/ascend_native_encoder_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "ops/encoder_layer.h"
#include "extendrt/delegate/plugin/tensorrt_executor_plugin.h"
#ifdef MS_ENABLE_ASCEND_DISTRIBUTION
#include "extendrt/delegate/ascend_native/ascend_native_impl/hccl_adapter.h"
#endif
namespace mindspore::kernel {
using mindspore::ops::kNameEncoderLayer;
void *AscendNativeEncoderKernel::prompt_mask_ = nullptr;

std::vector<int32_t> AscendNativeEncoderKernel::getOutputDimensions() {
  std::vector<int32_t> dims;
  if (param_.is_query_) {
    dims.push_back(param_.batch_size_);
    dims.push_back(param_.vocab_size_);
  } else if (param_.is_ln3) {
    dims.push_back(param_.batch_size_ * param_.seq_);
    dims.push_back(param_.hid_dim_);
  } else {
    dims.push_back(param_.batch_size_);
    dims.push_back(param_.seq_);
    dims.push_back(param_.hid_dim_);
  }
  return dims;
}

void AscendNativeEncoderKernel::PrintParam() {
  std::cout << " Batch=" << param_.batch_size_ << " head#=" << param_.head_num_ << " headSize=" << param_.head_size_
            << " hid_dim=" << param_.hid_dim_ << " ffn_hid_dim_=" << param_.ffn_hid_dim_
            << " expert_num=" << param_.expert_num_ << " rank_id_=" << param_.rank_id_
            << " rank_num_=" << param_.rank_num_ << " scale=" << param_.scale_
            << " is_embedding_=" << param_.is_embedding_ << " is_query=" << param_.is_query_
            << " is_ln3=" << param_.is_ln3 << " is_moe_=" << param_.is_moe_ << " rank_id_=" << param_.rank_id_
            << " rank_num_=" << param_.rank_num_ << " token_num_=" << param_.token_num_
            << " token_num2_=" << param_.token_num2_ << " incremental_mode_=" << param_.incremental_mode_
            << " act_kv_seq_=" << param_.act_kv_seq_[0] << " act_kv_seq_=" << param_.act_q_seq_[0] << std::endl;
}

int AscendNativeEncoderKernel::InferShape() {
  out_tensors_[0]->set_shape(getOutputDimensions());
  out_tensors_[0]->set_data_type(TypeId::kNumberTypeFloat16);
  return kSuccess;
}

int AscendNativeEncoderKernel::prepareOfflineEncoderWeight() {
  auto ret = ZeroOutBias();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "failed to ZeroOutBias ";
    return ret;
  }
  return kSuccess;
}

int AscendNativeEncoderKernel::ZeroOutBias() {
  if (param_.rank_num_ == Num1) {
    return kSuccess;
  }
  auto t = driver_input_tensors_.at(ENCODER_FFN_PROJ_BIAS_IDX);
  if (t == nullptr) {
    // some models are without bias
    return kSuccess;
  }

  auto expert_num = param_.expert_num_;
  auto len = param_.hid_dim_;
  auto size = expert_num * len * sizeof(uint16_t);
  uint16_t *h_t = reinterpret_cast<uint16_t *>(malloc(size));
  if (h_t == nullptr) {
    MS_LOG(ERROR) << "failed to malloc " << size;
    return kLiteError;
  }
  ascend_native::CopyDTH(h_t, t, size);
  auto device_len = len / param_.rank_num_;
  auto start = device_len * param_.rank_id_;
  auto end = device_len * (param_.rank_id_ + 1);
  for (int e = 0; e < expert_num; e++) {
    uint16_t *tmp_t = &h_t[e * len];
    for (int i = 0; i < start; i++) {
      tmp_t[i] = 0;
    }
    for (int i = end; i < len; i++) {
      tmp_t[i] = 0;
    }
  }
  ascend_native::CopyHTD(t, h_t, size);
  free(h_t);
  return kSuccess;
}

int AscendNativeEncoderKernel::TransposeProjW() {
  auto t = driver_input_tensors_.at(ENCODER_PROJECTION_IDX);
  if (t == nullptr) {
    MS_LOG(ERROR) << "Attention projection is nullptr ";
    return kLiteError;
  }

  int m = param_.hid_dim_;
  int n = param_.head_num_ * param_.head_size_;
  auto len = m * n;
  auto size = len * sizeof(uint16_t);

  uint16_t *src = reinterpret_cast<uint16_t *>(malloc(size));
  uint16_t *dst = reinterpret_cast<uint16_t *>(malloc(size));
  if (src == nullptr || dst == nullptr) {
    MS_LOG(ERROR) << "failed to malloc " << size;
    return kLiteError;
  }
  ascend_native::CopyDTH(src, t, size);
  for (int i = 0; i < len; i++) {
    int r = i / n;
    int c = i % n;
    int offset = c * m + r;
    dst[offset] = src[i];
  }
  ascend_native::CopyHTD(t, dst, size);
  free(src);
  free(dst);

  return kSuccess;
}
int AscendNativeEncoderKernel::CreateMask() {
  aclFloat16 *mask = reinterpret_cast<aclFloat16 *>(malloc(sizeof(aclFloat16) * param_.seq_ * param_.kv_seq_));
  if (mask == nullptr) {
    MS_LOG(ERROR) << "Ascend native encoder kernel malloc mask failed.";
    return kLiteError;
  }
  for (int j = 0; j < param_.seq_; j++) {
    for (int k = 0; k < param_.kv_seq_; k++) {
      if (k > j) {
        mask[j * param_.kv_seq_ + k] = (aclFloat16)10000.0f;
      } else {
        mask[j * param_.kv_seq_ + k] = (aclFloat16)0.0f;
      }
    }
  }
  prompt_mask_ =
    ascend_native::MallocCopy(reinterpret_cast<void *>(mask), sizeof(aclFloat16) * param_.seq_ * param_.kv_seq_);
  free(mask);
  return kSuccess;
}

int AscendNativeEncoderKernel::InitEncoderInputs() {
  // get encoder primitive
  auto encoder_op = AsOps<ops::EncoderLayer>();
  if (encoder_op == nullptr) {
    MS_LOG(ERROR) << "convert to primitive encoder failed for " << get_name();
    return kLiteError;
  }
  int idx = 1;  // idx 0 is from tensor or input_ids
  if (encoder_op->get_use_past()) {
    // setup k, v cache
    driver_input_tensors_.at(ENCODER_V_CACHE_IDX) = in_tensors_.at(idx++)->device_data();
    driver_input_tensors_.at(ENCODER_K_CACHE_IDX) = in_tensors_.at(idx++)->device_data();
  }
  driver_input_tensors_.at(ENCODER_LN1_GAMMA_IDX) = in_tensors_.at(idx++)->device_data();
  driver_input_tensors_.at(ENCODER_LN1_BETA_IDX) = in_tensors_.at(idx++)->device_data();
  idx += param_.is_cross_;  // skip position ids
  if (param_.is_query_) {
    driver_input_tensors_.at(ENCODER_DENSE_Q_IDX) = in_tensors_.at(idx++)->device_data();
    driver_input_tensors_.at(ENCODER_DENSE_KV_CONCAT_IDX) = in_tensors_.at(idx++)->device_data();
  } else {
    driver_input_tensors_.at(ENCODER_DENSE_CONCAT_IDX) = in_tensors_.at(idx++)->device_data();
  }
  driver_input_tensors_.at(ENCODER_DENSE_BIAS_IDX) = in_tensors_.at(idx++)->device_data();
  mask_tensor_idx_ = idx;
  auto mask = in_tensors_.at(idx++);
  auto mask_len = mask->shape().size();
  param_.seq_ = mask->shape().at(mask_len - C2NUM);
  param_.kv_seq_ = mask->shape().at(mask_len - C1NUM);
  param_.batch_size_ = mask->shape().at(C0NUM);
  driver_input_tensors_.at(ENCODER_PROJECTION_IDX) = in_tensors_.at(idx++)->device_data();
  driver_input_tensors_.at(ENCODER_PROJECTION_BIAS_IDX) = in_tensors_.at(idx++)->device_data();

  driver_input_tensors_.at(ENCODER_LN2_GAMMA_IDX) = in_tensors_.at(idx++)->device_data();
  driver_input_tensors_.at(ENCODER_LN2_BETA_IDX) = in_tensors_.at(idx++)->device_data();

  // setup mask
  if (prompt_mask_ == nullptr) {
    auto ret = CreateMask();
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "failed to CreateMask ";
      return ret;
    }
  }
  driver_input_tensors_.at(ENCODER_MASK_IDX) = prompt_mask_;

  // setup moe
  if (param_.is_moe_) {
    expert_tensor_idx_ = idx;
    auto expert = in_tensors_.at(idx++);
    param_.moe_num_ = expert->shape().at(0);
  }

  driver_input_tensors_.at(ENCODER_FFN_OUT_IDX) = in_tensors_.at(idx++)->device_data();
  driver_input_tensors_.at(ENCODER_FFN_OUT_BIAS_IDX) = in_tensors_.at(idx++)->device_data();
  driver_input_tensors_.at(ENCODER_FFN_PROJ_IDX) = in_tensors_.at(idx++)->device_data();
  driver_input_tensors_.at(ENCODER_FFN_PROJ_BIAS_IDX) = in_tensors_.at(idx++)->device_data();
  if (param_.is_ln3) {
    driver_input_tensors_.at(ENCODER_LN3_GAMMA_IDX) = in_tensors_.at(idx++)->device_data();
    driver_input_tensors_.at(ENCODER_LN3_BETA_IDX) = in_tensors_.at(idx++)->device_data();
  }

  // setup query layer
  if (param_.is_query_) {
    auto t = in_tensors_.at(idx++);
    param_.vocab_size_ = t->shape().at(C0NUM);
    driver_input_tensors_.at(ENCODER_V_EMBEDDING_IDX) = t->device_data();
    driver_input_tensors_.at(ENCODER_QUERY_EMBEDDING_IDX) = in_tensors_.at(idx++)->device_data();
  }
  // setup embedding

  if (param_.is_embedding_) {
    auto t = in_tensors_.at(idx++);
    param_.vocab_size_ = t->shape().at(C0NUM);
    driver_input_tensors_.at(ENCODER_V_EMBEDDING_IDX) = t->device_data();
    driver_input_tensors_.at(ENCODER_P_EMBEDDING_IDX) = in_tensors_.at(idx++)->device_data();
  }
  return kSuccess;
}

int AscendNativeEncoderKernel::InitEncoderParam() {
  // get encoder primitive
  auto encoder_op = AsOps<ops::EncoderLayer>();
  if (encoder_op == nullptr) {
    MS_LOG(ERROR) << "convert to primitive encoder failed for " << get_name();
    return kLiteError;
  }
  // setup normalization 1 parameters
  param_.eps1_ = encoder_op->get_eps_layernorm1();
  param_.rank_id_ = 0;
  param_.rank_num_ = C1NUM;
#ifdef MS_ENABLE_ASCEND_DISTRIBUTION
  auto &hccl = HcclAdapter::GetInstance();
  param_.rank_id_ = hccl.get_rank();
  param_.rank_num_ = hccl.get_size();
#endif
  // setup attention param
  param_.head_num_ = encoder_op->get_head_num();
  param_.head_size_ = encoder_op->get_head_size();
  param_.hid_dim_ = param_.rank_num_ * param_.head_num_ * param_.head_size_;
  param_.is_query_ = encoder_op->get_query_layer();
  param_.is_cross_ = param_.is_query_;
  param_.scale_ = encoder_op->get_scale();
  param_.eps2_ = encoder_op->get_eps_layernorm2();
  param_.is_moe_ = encoder_op->get_moe();
  param_.expert_num_ = param_.is_moe_ ? encoder_op->get_expert_num() : 1;
  param_.capacity_ = 1.1f;  // encoder_op->get_capacity_factor();
  param_.moe_id_ = encoder_op->get_expert_offset_id();
  param_.ffn_hid_dim_ = encoder_op->get_ffn_hidden_size();
  // setup normalization 3 parameters - if exist
  param_.is_ln3 = encoder_op->get_layer_norm();
  if (param_.is_ln3) {
    param_.eps3_ = encoder_op->get_eps_layernorm3();
  }
  param_.is_embedding_ = encoder_op->get_embedding_layer();
  param_.incremental_mode_ = false;
  param_.expert_to_tokens_ = reinterpret_cast<int *>(malloc(sizeof(int) * param_.expert_num_));
  param_.act_kv_seq_ = nullptr;
  param_.act_q_seq_ = nullptr;
  return kSuccess;
}

int AscendNativeEncoderKernel::Prepare() {
  ascend_native::SetContext(const_cast<void *>(acl_ctx_));
  auto ret = InitEncoderParam();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Ascend native encoder kernel InitEncoderParam failed.";
    return kLiteError;
  }
  ret = InitEncoderInputs();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Ascend native encoder kernel InitEncoderInputs failed.";
    return kLiteError;
  }
  ret = prepareOfflineEncoderWeight();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Ascend native encoder kernel prepareOfflineEncoderWeight failed.";
    return kLiteError;
  }
  ascend_native::pangu_encoder_prepare<aclFloat16>(&param_, &driver_input_tensors_, &ws_size_, &encoder_executer_);
  if (encoder_executer_ == nullptr) {
    MS_LOG(ERROR) << "Ascend native encoder kernel failed to create executer.";
    return kLiteError;
  }
  return kSuccess;
}

int AscendNativeEncoderKernel::Run() {
  param_.ctx = const_cast<void *>(acl_ctx_);
  // set up io
  int input_num = in_tensors_.size() - 1;
  if (param_.is_embedding_) {
    driver_input_tensors_.at(ENCODER_INPUT_IDS_IDX) = in_tensors_.at(0)->device_data();
  } else {
    driver_input_tensors_.at(ENCODER_INPUT_IDX) = in_tensors_.at(0)->device_data();
  }
  if (param_.is_query_) {
    driver_output_tensors_.at(HEAD_OUTPUT_IDX) = out_tensors_.at(0)->device_data();
  } else if (param_.is_ln3) {
    driver_output_tensors_.at(NORM_OUTPUT_IDX) = out_tensors_.at(0)->device_data();
  } else {
    driver_output_tensors_.at(ENCODER_OUTPUT_IDX) = out_tensors_.at(0)->device_data();
  }
  param_.token_num_ = *(reinterpret_cast<int *>(in_tensors_.at(input_num)->data()));
  param_.token_num2_ =
    *(reinterpret_cast<int *>(reinterpret_cast<uint8_t *>(in_tensors_.at(input_num--)->data()) + sizeof(int)));
  param_.incremental_mode_ = *(reinterpret_cast<int *>(in_tensors_.at(input_num)->data()));
  driver_input_tensors_.at(ENCODER_MODE_IDX) = in_tensors_.at(input_num--)->device_data();
  driver_input_tensors_.at(ENCODER_PADDING_KV_IDX) = in_tensors_.at(input_num--)->device_data();
  driver_input_tensors_.at(ENCODER_PADDING_Q_IDX) = in_tensors_.at(input_num--)->device_data();
  param_.act_kv_seq_ = reinterpret_cast<int *>(in_tensors_.at(input_num)->data());
  driver_input_tensors_.at(ENCODER_SEQ_LEN_KV_IDX) = in_tensors_.at(input_num--)->device_data();
  param_.act_q_seq_ = reinterpret_cast<int *>(in_tensors_.at(input_num)->data());
  driver_input_tensors_.at(ENCODER_SEQ_LEN_Q_IDX) = in_tensors_.at(input_num--)->device_data();
  driver_input_tensors_.at(ENCODER_BATCH_VALID_LENGTH_IDX) = in_tensors_.at(input_num--)->device_data();
  driver_input_tensors_.at(ENCODER_POS_IDS_IDX) = in_tensors_.at(input_num--)->device_data();
  driver_input_tensors_.at(ENCODER_EXPERT_IDS_IDX) = nullptr;
  if (expert_tensor_idx_ != -1) {
    driver_input_tensors_.at(ENCODER_EXPERT_IDS_IDX) = in_tensors_.at(expert_tensor_idx_)->device_data();
  }
  ascend_native::pangu_encoder_run<aclFloat16>(encoder_executer_, &driver_input_tensors_, &driver_output_tensors_,
                                               &param_, get_workspace(), get_sys_workspace(),
                                               const_cast<void *>(alt_stream_), const_cast<void *>(stream_));
  return kSuccess;
}

int AscendNativeEncoderKernel::ReSize() { return lite::RET_OK; }

REGISTER_ASCEND_NATIVE_CREATOR(kNameEncoderLayer, AscendNativeEncoderKernel)
}  // namespace mindspore::kernel
