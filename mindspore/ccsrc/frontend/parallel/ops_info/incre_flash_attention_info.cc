/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/core/ops/incre_flash_attention.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/parallel/ops_info/incre_flash_attention_info.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr size_t kInputQueryBatchDimBSH = 0;
constexpr size_t kInputQuerySeqDimBSH = 1;
constexpr size_t kInputQueryHiddenDimBSH = 2;
constexpr size_t kInputQueryBatchDimBNSD = 0;
constexpr size_t kInputQueryNDimBNSD = 1;
constexpr size_t kInputQuerySeqDimBNSD = 2;
constexpr size_t kInputQueryHiddenDimBNSD = 3;
constexpr char kAttrHeadNum[] = "num_heads";
constexpr char kAttrKVHeadNum[] = "num_key_value_heads";
constexpr char kAttrInputLayout[] = "input_layout";
constexpr char kAttrInputLayoutBSH[] = "BSH";
constexpr char kAttrInputLayoutBNSD[] = "BNSD";
constexpr size_t kRank4 = 4;
constexpr size_t kRank3 = 3;
}  // namespace

bool IncreFlashAttentionInfo::CheckStrategyOnIndex(int64_t strategy, int64_t true_value, const std::string &dim_name,
                                                   const std::string &input_name) {
  if (strategy != true_value) {
    MS_LOG(ERROR) << "For " << name_ << ": The " << dim_name << " of input " << input_name << " should be "
                  << true_value << ", but got strategy: " << strategy;
    return false;
  }
  return true;
}
void IncreFlashAttentionInfo::SetOptinalInputs() {
  optinal_inputs_.resize(ops::kIncreFlashAttentionInputsNum, true);
  size_t valid_input_index = 0;
  for (size_t index = 0; index < input_value_.size(); index++) {
    auto optinal_input_ptr = input_value_[index];
    if (optinal_input_ptr == nullptr) {
      if (index == ops::kIncreFlashAttentionInputAttnMaskIndex && valid_input_index < inputs_shape_.size()) {
        atten_mask_rank_ = inputs_shape_[valid_input_index].size();
      }
      if (index == ops::kIncreFlashAttentionInputPaddingMaskIndex && valid_input_index < inputs_shape_.size()) {
        padding_mask_rank_ = inputs_shape_[valid_input_index].size();
      }
      valid_input_index++;
    } else {
      if (optinal_input_ptr->isa<None>()) {
        optinal_inputs_[index] = False;
      }
    }
  }
  if (atten_mask_rank_ > kRank4 || padding_mask_rank_ > kRank4) {
    MS_LOG(EXCEPTION) << "atten_mask or padding_mask got unexpected ranks: " << atten_mask_rank_ << " and "
                      << padding_mask_rank_ << ". Expecting ranks not greater than 4.";
  }

  Shape atten_mask_tensor_map(atten_mask_rank_, -1);
  Shape atten_mask_strategy_map(atten_mask_rank_, 0);
  Shape padding_mask_tensor_map(padding_mask_rank_, -1);
  Shape padding_mask_strategy_map(padding_mask_rank_, 0);
  if (atten_mask_rank_ >= kRank3) {
    atten_mask_tensor_map[kInputQueryBatchDimBNSD] = 1;
    atten_mask_strategy_map[kInputQueryBatchDimBNSD] = 1;
  }
  if (padding_mask_rank_ >= kRank3) {
    padding_mask_tensor_map[kInputQueryBatchDimBNSD] = 1;
    padding_mask_strategy_map[kInputQueryBatchDimBNSD] = 1;
  }
  optinal_tensor_map_[ops::kIncreFlashAttentionInputAttnMaskIndex] = atten_mask_tensor_map;
  optinal_tensor_map_[ops::kIncreFlashAttentionInputPaddingMaskIndex] = padding_mask_tensor_map;
  optinal_op_strategies_[ops::kIncreFlashAttentionInputAttnMaskIndex] = atten_mask_strategy_map;
  optinal_op_strategies_[ops::kIncreFlashAttentionInputPaddingMaskIndex] = padding_mask_strategy_map;
}

Status IncreFlashAttentionInfo::GetAttrs() {
  head_num_ = GetIntAttr(kAttrHeadNum);
  kv_head_num = GetIntAttr(kAttrKVHeadNum);
  input_layout_ = GetStringAttr(kAttrInputLayout);
  SetOptinalInputs();
  return SUCCESS;
}

Status IncreFlashAttentionInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto strategies = strategy->GetInputDim();
  auto query_strategy = strategies[ops::kIncreFlashAttentionInputQueryIndex];
  auto key_strategy = strategies[ops::kIncreFlashAttentionInputKeyIndex];
  auto value_strategy = strategies[ops::kIncreFlashAttentionInputValueIndex];

  if (key_strategy != value_strategy) {
    MS_LOG(ERROR) << "For " << name_ << " : The in_strategy among  'key' and 'value' must be same.";
    return FAILED;
  }

  if (input_layout_ == kAttrInputLayoutBSH) {
    if (!CheckStrategyOnIndex(query_strategy[kInputQuerySeqDimBSH], 1, "S-Dimention", "query")) {
      return FAILED;
    }
    if (head_num_ % query_strategy[kInputQueryHiddenDimBSH] != 0) {
      MS_LOG(ERROR) << "For " << name_ << ": head_num % query_strategy[2] must be 0, but got " << head_num_
                    << "(head_num) and " << query_strategy[kInputQueryHiddenDimBSH] << "(query_strategy[2])";
      return FAILED;
    }
    dp_ = query_strategy[kInputQueryBatchDimBSH];
    mp_ = query_strategy[kInputQueryHiddenDimBSH];
  } else if (input_layout_ == kAttrInputLayoutBNSD) {
    if (!CheckStrategyOnIndex(query_strategy[kInputQuerySeqDimBNSD], 1, "S-Dimention", "query") ||
        !CheckStrategyOnIndex(query_strategy[kInputQueryHiddenDimBNSD], 1, "D-Dimention", "query")) {
      return FAILED;
    }
    dp_ = query_strategy[kInputQueryBatchDimBNSD];
    mp_ = query_strategy[kInputQueryNDimBNSD];
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }
  if (optinal_inputs_.empty()) {
    SetOptinalInputs();
  }
  return SUCCESS;
}

Status IncreFlashAttentionInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = {dp_, mp_};
  return SUCCESS;
}

Status IncreFlashAttentionInfo::InferTensorMap() {
  if (input_layout_ == kAttrInputLayoutBSH) {
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});  // query
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});  // key
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});  // value
    outputs_tensor_map_.push_back({1, -1, 0});               // attention_out
  } else if (input_layout_ == kAttrInputLayoutBNSD) {
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});  // query
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});  // key
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});  // value
    outputs_tensor_map_.push_back({1, 0, -1, -1});               // attention_out
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }
  for (auto index = static_cast<size_t>(ops::kIncreFlashAttentionInputAttnMaskIndex); index < optinal_inputs_.size();
       index++) {
    if (optinal_inputs_[index]) {
      (void)inputs_tensor_map_.emplace_back(optinal_tensor_map_[index]);
    }
  }
  return SUCCESS;
}

std::vector<StrategyPtr> IncreFlashAttentionInfo::GenerateOpStrategies(int64_t stage_id) {
  Shapes splitable_inputs;
  if (input_layout_ == kAttrInputLayoutBSH) {
    Shape splitable_query{1, 0, 2};
    Shape splitable_key{1, 0, 2};
    Shape splitable_value{1, 0, 2};
    splitable_inputs = {splitable_query, splitable_key, splitable_value};
  } else if (input_layout_ == kAttrInputLayoutBNSD) {
    Shape splitable_query{1, 2, 0, 0};
    Shape splitable_key{1, 2, 0, 0};
    Shape splitable_value{1, 2, 0, 0};
    splitable_inputs = {splitable_query, splitable_key, splitable_value};
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
  }
  if (optinal_inputs_.empty()) {
    SetOptinalInputs();
  }
  for (auto index = static_cast<size_t>(ops::kIncreFlashAttentionInputAttnMaskIndex); index < optinal_inputs_.size();
       index++) {
    if (optinal_inputs_[index]) {
      (void)splitable_inputs.emplace_back(optinal_op_strategies_[index]);
    }
  }

  std::vector<StrategyPtr> strategy_vector;
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, splitable_inputs, &strategy_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for dependent inputs() failed.";
  }
  if (strategy_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No valid strategy.";
  }
  return strategy_vector;
}

void IncreFlashAttentionInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[ops::kIncreFlashAttentionInputQueryIndex] = true;
  split_flag_list_[ops::kIncreFlashAttentionInputKeyIndex] = true;
  split_flag_list_[ops::kIncreFlashAttentionInputValueIndex] = true;
  split_flag_list_[ops::kIncreFlashAttentionInputAttnMaskIndex] = true;
  split_flag_list_[ops::kIncreFlashAttentionInputPaddingMaskIndex] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputActualSeqLengths] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputDequantScale1] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputQuantScale1] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputDequantScale2] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputQuantScale2] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputQuantOffset2] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputAntiquantScale] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputAntiquantOffset] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputBlockTable] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputsNum] = false;
}

Status IncreFlashAttentionInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  if (mirror_ops_.empty()) {
    // No need to insert mirror ops
    return SUCCESS;
  }
  for (size_t i = mirror_ops_.size(); i < ops::kIncreFlashAttentionInputsNum; ++i) {
    // Push empty mirror op for optional input
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  return SUCCESS;
}

void IncreFlashAttentionInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    auto clone_prim = prim->Clone();
    clone_prim->set_attr(kAttrHeadNum, MakeValue(head_num_ / mp_));
    clone_prim->set_attr(kAttrKVHeadNum, MakeValue(kv_head_num / mp_));
    cnode->set_input(0, NewValueNode(clone_prim)->cast<AnfNodePtr>());
  }
}

REGISTER(IncreFlashAttentionInfo);
}  // namespace parallel
}  // namespace mindspore
