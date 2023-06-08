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

#include "frontend/parallel/ops_info/flash_attention_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status FlashAttentionPrimitiveInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

Status FlashAttentionPrimitiveInfo::GetAttrs() {
  MS_LOG(INFO) << name_ << ": The size of flash attention attrs:" << attrs_.size();

  // infer inputs dimension size
  if (inputs_shape_.size() == FLASH_ATTENTION_INPUTS_SIZE) {
    MS_LOG(INFO) << name_ << ": Inputs shape size is 7, alibi mask not none.";
    alibi_mask_valid = true;
  }
  auto input_sz = inputs_shape_.size();
  bool input_sz_ok = input_sz == FLASH_ATTENTION_INPUTS_SIZE || input_sz == FLASH_ATTENTION_INPUTS_SIZE - 1;
  bool output_sz_ok = outputs_shape_.size() == FLASH_ATTENTION_OUTPUTS_SIZE;
  if (!input_sz_ok || !output_sz_ok) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size: " << input_sz << " or outputs shape size is wrong.";
    return FAILED;
  }

  size_t q_dim_sz = inputs_shape_.at(0).size();
  size_t k_dim_sz = inputs_shape_.at(1).size();
  size_t v_dim_sz = inputs_shape_.at(2).size();
  size_t dim_mask_d_sz = inputs_shape_.at(3).size();
  size_t attn_mask_d_sz = inputs_shape_.at(4).size();
  vector<int64_t> &dropout_mask_shape = inputs_shape_.at(DROPOUT_MASK_INPUT_INDEX);
  size_t dropout_mask_d_sz = dropout_mask_shape.size();

  MS_LOG(DEBUG) << name_ << " q_dim_sz: " << q_dim_sz;
  MS_LOG(DEBUG) << name_ << " k_dim_sz: " << k_dim_sz;
  MS_LOG(DEBUG) << name_ << " v_dim_sz: " << v_dim_sz;
  MS_LOG(DEBUG) << name_ << " dim_mask_dim_sz: " << dim_mask_d_sz;
  MS_LOG(DEBUG) << name_ << " attn_mask_dim_sz: " << attn_mask_d_sz;
  MS_LOG(DEBUG) << name_ << " dropout_mask_dim_sz: " << dropout_mask_d_sz;
  MS_LOG(DEBUG) << name_ << " dropout_mask_dim:" << ShapeToString(dropout_mask_shape);

  bool drop_mask_shape_ok = false;
  if (IsDynamic(dropout_mask_shape)) {
    drop_mask_shape_ok = true;
  }

  if (!drop_mask_shape_ok && dropout_mask_d_sz != dropout_mask_dim_sz) {
    MS_LOG(ERROR) << name_ << ": The dim dropout mask dim should be equal to 4, ";
    return FAILED;
  }

  if (!(q_dim_sz == qkv_dim_sz && k_dim_sz == qkv_dim_sz && v_dim_sz == qkv_dim_sz &&
        dim_mask_d_sz == dim_mask_dim_sz && attn_mask_d_sz == attn_mask_dim_sz)) {
    MS_LOG(ERROR) << name_
                  << ": The dim of Q, K, V, attn_mask, dropout mask dim should be equal to 4, "
                     "and dim mask dim size should be equal to 1.";
    return FAILED;
  }
  if (alibi_mask_valid) {
    size_t alibi_mask_d_sz = inputs_shape_.at(6).size();
    if (alibi_mask_d_sz != alibi_mask_dim_sz) {
      MS_LOG(ERROR) << name_ << ": The dim alibi mask dim should be equal to " << alibi_mask_dim_sz;
      return FAILED;
    }
  }

  return SUCCESS;
}

Status FlashAttentionPrimitiveInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map is empty";
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

Status FlashAttentionPrimitiveInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": The strategy is empty, 1111 mirror_ops.size() = " << mirror_ops_.size();
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }
  if (!alibi_mask_valid) {
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  MS_LOG(INFO) << name_ << ": The strategy is empty, 2222 mirror_ops.size() = " << mirror_ops_.size();
  return SUCCESS;
}

Status FlashAttentionPrimitiveInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  Dimensions stgy_q_dim = stra.at(0);
  Dimensions stgy_k_dim = stra.at(1);
  Dimensions stgy_v_dim = stra.at(2);
  Dimensions stgy_dim_mask_dim = stra.at(3);
  Dimensions stgy_attn_mask_dim = stra.at(4);
  Dimensions stgy_dropout_mask_dim = stra.at(5);

  size_t stgy_q_dim_sz = stgy_q_dim.size();
  size_t stgy_k_dim_sz = stgy_k_dim.size();
  size_t stgy_v_dim_sz = stgy_v_dim.size();
  size_t stgy_dim_msk_dim_sz = stgy_dim_mask_dim.size();
  size_t stgy_attn_msk_dim_sz = stgy_attn_mask_dim.size();
  size_t stgy_dropout_msk_dim_sz = stgy_dropout_mask_dim.size();

  MS_LOG(INFO) << "stgy_q_dim:" << ShapeToString(stgy_q_dim);
  MS_LOG(INFO) << "stgy_k_dim:" << ShapeToString(stgy_k_dim);
  MS_LOG(INFO) << "stgy_v_dim:" << ShapeToString(stgy_v_dim);
  MS_LOG(INFO) << "stgy_dim_mask_dim:" << ShapeToString(stgy_dim_mask_dim);
  MS_LOG(INFO) << "stgy_attn_msk_dim:" << ShapeToString(stgy_attn_mask_dim);
  MS_LOG(INFO) << "stgy_dropout_msk_dim:" << ShapeToString(stgy_dropout_mask_dim);

  if (!(stgy_q_dim_sz == qkv_dim_sz && stgy_k_dim_sz == qkv_dim_sz && stgy_v_dim_sz == qkv_dim_sz &&
        stgy_dim_msk_dim_sz == dim_mask_dim_sz && stgy_attn_msk_dim_sz == attn_mask_dim_sz &&
        stgy_dropout_msk_dim_sz == dropout_mask_dim_sz)) {
    MS_LOG(ERROR) << name_ << ": The dimensions of Q or K or V or mask strategy is wrong.";
    return FAILED;
  }
  // stgy_q_dim:[dp, mp, 1, 1]
  // stgy_k_dim:[dp, mp, 1, 1]
  // stgy_v_dim:[dp, mp, 1, 1]
  // stgy_dim_mask_dim:[1]
  // stgy_attn_msk_dim:[1/dp mp, 1, 1]
  // stgy_dropout_msk_dim:[dp, mp, 1, 1]
  auto dp = stgy_q_dim[0];
  auto mp = stgy_q_dim[1];
  MS_LOG(INFO) << name_ << " dp: " << dp;
  MS_LOG(INFO) << name_ << " mp: " << mp;

  auto dp_mp_shard_cfg = std::vector<int64>({dp, mp, 1, 1});  // Q K V shard cfg.
  auto attn_mask_shard_cfg = std::vector<int64>({dp, 1, 1});  // Q K V attn_mask dropout_mask shard cfg.
  uint64_t attn_mask_input_bs = inputs_shape_[4][0];
  if (attn_mask_input_bs == 1) {
    attn_mask_shard_cfg[0] = 1;
  }

  bool q_stgy_shard_ok = std::equal(stgy_q_dim.begin(), stgy_q_dim.end(), dp_mp_shard_cfg.begin());
  bool k_stgy_shard_ok = std::equal(stgy_k_dim.begin(), stgy_k_dim.end(), dp_mp_shard_cfg.begin());
  bool v_stgy_shard_ok = std::equal(stgy_v_dim.begin(), stgy_v_dim.end(), dp_mp_shard_cfg.begin());
  bool attn_msk_stgy_shard_ok =
    std::equal(stgy_attn_mask_dim.begin(), stgy_attn_mask_dim.end(), attn_mask_shard_cfg.begin());
  bool dropout_msk_stgy_shard_ok =
    std::equal(stgy_dropout_mask_dim.begin(), stgy_dropout_mask_dim.end(), dp_mp_shard_cfg.begin());
  if (!(q_stgy_shard_ok && k_stgy_shard_ok && v_stgy_shard_ok && attn_msk_stgy_shard_ok && dropout_msk_stgy_shard_ok &&
        stgy_dim_mask_dim[0] == 1)) {
    MS_LOG(ERROR) << name_ << ": sharding strategy configuration wrong.";
    return FAILED;
  }

  if (alibi_mask_valid) {
    Dimensions stgy_alibi_mask_dim = stra.at(6);
    MS_LOG(INFO) << "stgy_alibi_msk_dim:" << ShapeToString(stgy_alibi_mask_dim);
    size_t stgy_alibi_msk_dim_sz = stgy_alibi_mask_dim.size();
    if (stgy_alibi_msk_dim_sz != alibi_mask_dim_sz) {
      MS_LOG(ERROR) << name_ << ": The dimensions alibi mask strategy is wrong.";
      return FAILED;
    }
    bool alibi_msk_stgy_shard_ok =
      std::equal(stgy_alibi_mask_dim.begin(), stgy_alibi_mask_dim.end(), dp_mp_shard_cfg.begin());
    if (!alibi_msk_stgy_shard_ok) {
      MS_LOG(ERROR) << name_ << ": alibi_mask sharding strategy configuration wrong.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FlashAttentionPrimitiveInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions q_stgy_dim = stra.at(0);
  size_t dp, mp;
  dp = q_stgy_dim[0];
  mp = q_stgy_dim[1];
  dev_matrix_shape_.push_back(dp);
  dev_matrix_shape_.push_back(mp);
  dev_matrix_shape_.push_back(1);
  dev_matrix_shape_.push_back(1);

  return SUCCESS;
}

Status FlashAttentionPrimitiveInfo::InferTensorMap() {
  inputs_tensor_map_.push_back({3, 2, 1, 0});  // Q
  inputs_tensor_map_.push_back({3, 2, 1, 0});  // K
  inputs_tensor_map_.push_back({3, 2, 1, 0});  // V
  inputs_tensor_map_.push_back({-1});          // dim_mask

  uint64_t attn_mask_input_bs = inputs_shape_[4][0];
  if (attn_mask_input_bs == 1) {
    inputs_tensor_map_.push_back({-1, 1, 0});  // attn_mask
  } else {
    inputs_tensor_map_.push_back({3, 1, 0});  // attn_mask
  }
  inputs_tensor_map_.push_back({3, 2, 1, 0});  // dropout_mask

  if (alibi_mask_valid) {
    inputs_tensor_map_.push_back({3, 2, 1, 0});  // alibi_mask
  }
  outputs_tensor_map_.push_back({3, 2, 1, 0});  // O
  outputs_tensor_map_.push_back({3, 2, 1});     // L
  outputs_tensor_map_.push_back({3, 2, 1});     // M

  return SUCCESS;
}

std::vector<StrategyPtr> FlashAttentionPrimitiveInfo::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp_vector;

  return sp_vector;
}

REGISTER(FlashAttentionPrimitiveInfo);
}  // namespace parallel
}  // namespace mindspore
