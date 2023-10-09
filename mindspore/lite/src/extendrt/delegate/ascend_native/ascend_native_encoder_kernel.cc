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

namespace mindspore::kernel {
using mindspore::ops::kNameEncoderLayer;

std::vector<int32_t> AscendNativeEncoderKernel::GetOutputDimensions() {
  std::vector<int32_t> dims;
  if (param_.is_query_) {
    dims.push_back(param_.B);
    dims.push_back(param_.embedding_param.get_vcobalary_size());
  } else if (param_.is_last_norm_) {
    dims.push_back(param_.B * param_.MAX_N);
    dims.push_back(param_.D);
  } else {
    dims.push_back(param_.B);
    dims.push_back(param_.MAX_N);
    dims.push_back(param_.D);
  }
  return dims;
}

int AscendNativeEncoderKernel::InferShape() {
  out_tensors_[0]->set_shape(GetOutputDimensions());
  out_tensors_[0]->set_data_type(TypeId::kNumberTypeFloat16);
  return kSuccess;
}

int AscendNativeEncoderKernel::InitEncoderParam() {
  // get encoder primitive
  auto encoder_op = AsOps<ops::EncoderLayer>();
  if (encoder_op == nullptr) {
    MS_LOG(ERROR) << "convert to primitive encoder failed for " << get_name();
    return kLiteError;
  }
  int idx = 1;  // idx 0 is from tensor or input_ids
  if (encoder_op->get_use_past()) {
    // setup k, v cache
    param_.attn_param.set_v_cache(in_tensors_.at(idx++)->device_data());
    param_.attn_param.set_k_cache(in_tensors_.at(idx++)->device_data());
  }
  // setup normalization 1 parameters
  param_.norm1.set_eps(encoder_op->get_eps_layernorm1());
  param_.norm1.set_gamma(in_tensors_.at(idx++)->device_data());
  param_.norm1.set_beta(in_tensors_.at(idx++)->device_data());
  // setup comm param
  param_.comm_param.set_rank_id(0);
  param_.comm_param.set_rank_num(1);
#ifdef LITE_ASCEND_DISTRIBUTION_TBD_FLAG
  uint32_t rank_id = GetDeviceInfo(context_);
  param_.comm_param.set_rank_id(rank_id);
  param_.comm_param.set_rank_num(Num4);
#endif
  // setup attention param
  param_.attn_param.set_head_number(encoder_op->get_head_num());
  param_.attn_param.set_head_size(encoder_op->get_head_size());
  param_.attn_param.set_hidden_dim(param_.attn_param.get_head_number() * param_.attn_param.get_head_size() *
                                   param_.comm_param.get_rank_num());
  param_.H = encoder_op->get_head_num();
  param_.HS = encoder_op->get_head_size();
  param_.D = param_.H * param_.HS * param_.comm_param.get_rank_num();
  param_.is_query_ = encoder_op->get_query_layer();
  param_.attn_param.set_is_cross(encoder_op->get_query_layer());
  param_.attn_param.set_scale(encoder_op->get_scale());
  if (param_.attn_param.get_is_cross()) idx++;  // skip posotion ids
  param_.attn_param.set_qkv_weight(in_tensors_.at(idx++)->device_data());
  if (param_.attn_param.get_is_cross()) {
    param_.attn_param.set_kv_weight(in_tensors_.at(idx++)->device_data());
  }
  param_.attn_param.set_qkv_bias(in_tensors_.at(idx++)->device_data());
  param_.attn_param.set_q_seq_len(in_tensors_.at(C0NUM)->shape()[C1NUM]);
  param_.attn_param.set_kv_seq_len(in_tensors_.at(C0NUM)->shape()[C1NUM]);
  param_.MAX_N = in_tensors_.at(C0NUM)->shape()[C1NUM];
  param_.B = in_tensors_.at(in_tensors_.size() - Num2)->shape()[C0NUM];
  idx++;  // skip mask
  param_.attn_param.set_projection_weight(in_tensors_.at(idx++)->device_data());
  param_.attn_param.set_projection_bias(in_tensors_.at(idx++)->device_data());
  param_.norm2.set_eps(encoder_op->get_eps_layernorm2());
  param_.norm2.set_gamma(in_tensors_.at(idx++)->device_data());
  param_.norm2.set_beta(in_tensors_.at(idx++)->device_data());
  param_.is_moe_ = encoder_op->get_moe();
  if (param_.is_moe_) {
    param_.moe.set_expert_number(encoder_op->get_expert_num());
    param_.E = encoder_op->get_expert_num();
  } else {
    param_.ffn_param.set_mapping_weight(in_tensors_.at(idx++)->device_data());
    param_.ffn_param.set_mapping_bias(in_tensors_.at(idx++)->device_data());
    param_.ffn_param.set_projection_weight(in_tensors_.at(idx++)->device_data());
    param_.ffn_param.set_projection_bias(in_tensors_.at(idx++)->device_data());
  }
  param_.ffn_param.set_ffn_hidden_size(encoder_op->get_ffn_hidden_size());
  param_.HFFN = encoder_op->get_ffn_hidden_size();
  // setup normalization 3 parameters - if exist
  param_.is_last_norm_ = encoder_op->get_layer_norm();
  if (param_.is_last_norm_) {
    param_.norm3.set_eps(encoder_op->get_eps_layernorm3());
    param_.norm3.set_gamma(in_tensors_.at(idx++)->device_data());
    param_.norm3.set_beta(in_tensors_.at(idx++)->device_data());
  }
  // setup query layer
  if (param_.is_query_) {
    auto t = in_tensors_.at(idx++);
    param_.embedding_param.set_vcobalary_size(t->shape().at(C0NUM));
    param_.embedding_param.set_word_embedding(t->device_data());
    param_.embedding_param.set_top_query_embedding(in_tensors_.at(idx++)->device_data());
  }
  param_.is_embedding_ = encoder_op->get_embedding_layer();
  // setup embedding
  if (param_.is_embedding_) {
    auto t = in_tensors_.at(idx++);
    param_.embedding_param.set_vcobalary_size(t->shape().at(C0NUM));
    param_.embedding_param.set_word_embedding(t->device_data());
    param_.embedding_param.set_position_embedding(in_tensors_.at(idx++)->device_data());
  }
  param_.capacity = 0;  // capacity of tokens per expert
  return kSuccess;
}

int AscendNativeEncoderKernel::Prepare() {
  auto ret = InitEncoderParam();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Ascend native encoder kernel InitEncoderParam failed.";
    return kLiteError;
  }
  if (param_.is_query_) {
    encoder_driver_ = std::make_shared<ascend_native::AscendNativeQuery>();
  } else if (param_.is_last_norm_) {
    encoder_driver_ = std::make_shared<ascend_native::AscendNativeEncoderFuseLastNorm>();
  } else {
    encoder_driver_ = std::make_shared<ascend_native::AscendNativeEncoder>(param_.is_embedding_);
  }
  build_driver_input_const_tensors();
  return kSuccess;
}

void AscendNativeEncoderKernel::build_driver_input_const_tensors() {
  driver_input_tensors_.at(ENCODER_LN1_GAMMA_IDX) = param_.norm1.get_gamma();
  driver_input_tensors_.at(ENCODER_LN1_BETA_IDX) = param_.norm1.get_beta();
  if (param_.is_query_) {
    driver_input_tensors_.at(ENCODER_DENSE_Q_IDX) = param_.attn_param.get_qkv_weight();
    driver_input_tensors_.at(ENCODER_DENSE_KV_CONCAT_IDX) = param_.attn_param.get_kv_weight();
  } else {
    driver_input_tensors_.at(ENCODER_DENSE_CONCAT_IDX) = param_.attn_param.get_qkv_weight();
  }
  driver_input_tensors_.at(ENCODER_K_CACHE_IDX) = param_.attn_param.get_k_cache();
  driver_input_tensors_.at(ENCODER_V_CACHE_IDX) = param_.attn_param.get_v_cache();
  driver_input_tensors_.at(ENCODER_DENSE_BIAS_IDX) = param_.attn_param.get_qkv_bias();
  driver_input_tensors_.at(ENCODER_PROJECTION_IDX) = param_.attn_param.get_projection_weight();
  driver_input_tensors_.at(ENCODER_PROJECTION_BIAS_IDX) = param_.attn_param.get_projection_bias();
  driver_input_tensors_.at(ENCODER_LN2_GAMMA_IDX) = param_.norm2.get_gamma();
  driver_input_tensors_.at(ENCODER_LN2_BETA_IDX) = param_.norm2.get_beta();
  driver_input_tensors_.at(ENCODER_FFN_OUT_IDX) = param_.ffn_param.get_mapping_weight();
  driver_input_tensors_.at(ENCODER_FFN_OUT_BIAS_IDX) = param_.ffn_param.get_mapping_bias();
  driver_input_tensors_.at(ENCODER_FFN_PROJ_IDX) = param_.ffn_param.get_projection_weight();
  driver_input_tensors_.at(ENCODER_FFN_PROJ_BIAS_IDX) = param_.ffn_param.get_projection_bias();

  driver_input_tensors_.at(ENCODER_V_EMBEDDING_IDX) = param_.embedding_param.get_word_embedding();
  driver_input_tensors_.at(ENCODER_P_EMBEDDING_IDX) = param_.embedding_param.get_position_embedding();
  driver_input_tensors_.at(ENCODER_QUERY_EMBEDDING_IDX) = param_.embedding_param.get_top_query_embedding();

  driver_input_tensors_.at(ENCODER_LN3_GAMMA_IDX) = param_.norm3.get_gamma();
  driver_input_tensors_.at(ENCODER_LN3_BETA_IDX) = param_.norm3.get_beta();
}

int AscendNativeEncoderKernel::Run() {
  const std::vector<InferTensor *> &inputs = in_tensors();

  if (param_.is_embedding_) {
    driver_input_tensors_.at(ENCODER_INPUT_IDS_IDX) = inputs.at(0)->device_data();
  } else {
    driver_input_tensors_.at(ENCODER_INPUT_IDX) = inputs.at(0)->device_data();
  }
  if (param_.is_query_) {
    driver_output_tensors_.at(HEAD_OUTPUT_IDX) = out_tensors_.at(0)->device_data();
  } else if (param_.is_last_norm_) {
    driver_output_tensors_.at(NORM_OUTPUT_IDX) = out_tensors_.at(0)->device_data();
  } else {
    driver_output_tensors_.at(ENCODER_OUTPUT_IDX) = out_tensors_.at(0)->device_data();
  }
  driver_input_tensors_.at(ENCODER_BATCH_VALID_LENGTH_IDX) = inputs.at(inputs.size() - C1NUM)->device_data();
  driver_input_tensors_.at(ENCODER_POS_IDS_IDX) = inputs.at(inputs.size() - C2NUM)->device_data();
  int token = 0;
  if (param_.is_embedding_)
    ascend_native::PreapreVSL(driver_input_tensors_.at(ENCODER_BATCH_VALID_LENGTH_IDX), param_.B, &token,
                              const_cast<void *>(stream_));
  else
    ascend_native::GetTokenNum(driver_input_tensors_.at(ENCODER_BATCH_VALID_LENGTH_IDX), param_.B, &token,
                               const_cast<void *>(stream_));
  param_.num_of_tokens = token;
  static bool inc = false;
  if (inc) {
    param_.num_of_tokens = 1 * param_.B;
  }
  if (param_.is_query_ && !inc) inc = true;
  void *ws = get_workspace();

  encoder_driver_->Forward(&driver_input_tensors_, &driver_output_tensors_, ws, &param_, const_cast<void *>(stream_));
  return kSuccess;
}

size_t AscendNativeEncoderKernel::get_workspace_size() const { return encoder_driver_->GetWorkspaceSize(param_); }

REGISTER_ASCEND_NATIVE_CREATOR(kNameEncoderLayer, AscendNativeEncoderKernel)
}  // namespace mindspore::kernel
