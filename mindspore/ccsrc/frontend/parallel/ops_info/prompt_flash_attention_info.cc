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
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/ops_info/prompt_flash_attention_info.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/core/ops/prompt_flash_attention.h"
#include "mindspore/core/ops/array_ops.h"

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
constexpr char kAttrInputLayout[] = "input_layout";
constexpr size_t rank_2 = 2;
constexpr size_t rank_3 = 3;
}  // namespace

bool PromptFlashAttentionInfo::CheckStrategy(int64_t strategy, int64_t true_value, const std::string &dim_name,
                                             const std::string &input_name) {
  if (strategy != true_value) {
    MS_LOG(ERROR) << "For " << name_ << ": The " << dim_name << " of input " << input_name << " should be "
                  << true_value << ", but got strategy: " << strategy;
    return false;
  }
  return true;
}

void PromptFlashAttentionInfo::SetOptinalInputs() {
  optinal_inputs.resize(ops::kPromptFlashAttentionInputsNum, true);
  size_t valid_input_index = 0;
  for (size_t index = 0; index < input_value_.size(); index++) {
    auto optinal_input_ptr = input_value_[index];
    if (optinal_input_ptr == nullptr) {
      if (index == ops::kPromptFlashAttentionInputAttnMaskIndex && valid_input_index < inputs_shape_.size()) {
        atten_mask_rank = inputs_shape_[valid_input_index].size();
      }
      if (index == ops::kPromptFlashAttentionInputPaddingMaskIndex && valid_input_index < inputs_shape_.size()) {
        padding_mask_rank = inputs_shape_[valid_input_index].size();
      }
      valid_input_index++;
    } else {
      if (optinal_input_ptr->isa<None>()) {
        optinal_inputs[index] = False;
      }
    }
  }
  if (atten_mask_rank == rank_2) {
    optinal_tensor_map[ops::kPromptFlashAttentionInputAttnMaskIndex] = {-1, -1};
    optinal_op_strategies[ops::kPromptFlashAttentionInputAttnMaskIndex] = {0, 0};
  }
  if (atten_mask_rank == rank_3) {
    optinal_tensor_map[ops::kPromptFlashAttentionInputAttnMaskIndex] = {1, -1, -1};
    optinal_op_strategies[ops::kPromptFlashAttentionInputAttnMaskIndex] = {1, 0, 0};
  }
  if (padding_mask_rank == rank_2) {
    optinal_tensor_map[ops::kPromptFlashAttentionInputPaddingMaskIndex] = {-1, -1};
    optinal_op_strategies[ops::kPromptFlashAttentionInputPaddingMaskIndex] = {0, 0};
  }
  if (padding_mask_rank == rank_3) {
    optinal_tensor_map[ops::kPromptFlashAttentionInputPaddingMaskIndex] = {1, -1, -1};
    optinal_op_strategies[ops::kPromptFlashAttentionInputPaddingMaskIndex] = {1, 0, 0};
  }
}

void PromptFlashAttentionInfo::GenerateExpectStrategies() {
  expect_strategies = {{}, {}, {}, {dp_, 1, 1, 1}, {dp_, 1, 1, 1}, {dp_}, {dp_}, {}, {}, {}, {}, {}};
  if (atten_mask_rank == rank_2) {
    expect_strategies[ops::kPromptFlashAttentionInputAttnMaskIndex] = {1, 1};
  }
  if (atten_mask_rank == rank_3) {
    expect_strategies[ops::kPromptFlashAttentionInputAttnMaskIndex] = {dp_, 1, 1};
  }
  if (padding_mask_rank == rank_2) {
    expect_strategies[ops::kPromptFlashAttentionInputPaddingMaskIndex] = {1, 1};
  }
  if (padding_mask_rank == rank_3) {
    expect_strategies[ops::kPromptFlashAttentionInputPaddingMaskIndex] = {dp_, 1, 1};
  }
}

Status PromptFlashAttentionInfo::GetAttrs() {
  head_num_ = GetIntAttr(kAttrHeadNum);
  input_layout_ = GetStringAttr(kAttrInputLayout);
  SetOptinalInputs();
  return SUCCESS;
}

Status PromptFlashAttentionInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto strategies = strategy->GetInputDim();
  auto query_strategy = strategies[ops::kPromptFlashAttentionInputQueryIndex];
  auto key_strategy = strategies[ops::kPromptFlashAttentionInputKeyIndex];
  auto value_strategy = strategies[ops::kPromptFlashAttentionInputValueIndex];

  if (query_strategy != key_strategy || query_strategy != value_strategy) {
    MS_LOG(ERROR) << "For " << name_ << " : The in_strategy among 'query' 'key' and 'value' must be same.";
    return FAILED;
  }

  if (input_layout_ == "BSH") {
    if (!CheckStrategy(query_strategy[kInputQuerySeqDimBSH], 1, "S-Dimention", "query")) {
      return FAILED;
    }
    if (head_num_ % query_strategy[kInputQueryHiddenDimBSH] != 0) {
      MS_LOG(ERROR) << "For " << name_ << ": head_num % query_strategy[2] must be 0, but got " << head_num_
                    << "(head_num) and " << query_strategy[kInputQueryHiddenDimBSH] << "(query_strategy[2])";
      return FAILED;
    }
    dp_ = query_strategy[kInputQueryBatchDimBSH];
    mp_ = query_strategy[kInputQueryHiddenDimBSH];
  } else if (input_layout_ == "BNSD") {
    if (!CheckStrategy(query_strategy[kInputQuerySeqDimBNSD], 1, "S-Dimention", "query") ||
        !CheckStrategy(query_strategy[kInputQueryHiddenDimBNSD], 1, "D-Dimention", "query")) {
      return FAILED;
    }
    dp_ = query_strategy[kInputQueryBatchDimBNSD];
    mp_ = query_strategy[kInputQueryNDimBNSD];
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }
  if (optinal_inputs.empty()) {
    SetOptinalInputs();
  }

  GenerateExpectStrategies();
  size_t s_index = ops::kPromptFlashAttentionInputAttnMaskIndex;
  for (size_t index = s_index; index < optinal_inputs.size(); index++) {
    if (optinal_inputs[index]) {
      Shape expect_strategy = expect_strategies[index];
      auto actual_strategy = strategies[s_index++];
      if (expect_strategy != actual_strategy) {
        MS_LOG(ERROR) << "For" << name_ << ": The in_strategy for " << index << " must be " << expect_strategy
                      << ", but got " << actual_strategy;
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status PromptFlashAttentionInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = {dp_, mp_};
  return SUCCESS;
}

Status PromptFlashAttentionInfo::InferTensorMap() {
  if (optinal_inputs.empty()) {
    SetOptinalInputs();
  }
  if (input_layout_ == "BSH") {
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});  // query
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});  // key
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});  // value
    outputs_tensor_map_.push_back({1, -1, 0});               // attention_out
  } else if (input_layout_ == "BNSD") {
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});  // query
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});  // key
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});  // value
    outputs_tensor_map_.push_back({1, 0, -1, -1});               // attention_out
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }
  for (auto index = static_cast<size_t>(ops::kPromptFlashAttentionInputAttnMaskIndex); index < optinal_inputs.size();
       index++) {
    if (optinal_inputs[index]) {
      (void)inputs_tensor_map_.emplace_back(optinal_tensor_map[index]);
    }
  }
  return SUCCESS;
}

std::vector<StrategyPtr> PromptFlashAttentionInfo::GenerateOpStrategies(int64_t stage_id) {
  if (optinal_inputs.empty()) {
    SetOptinalInputs();
  }
  Shapes splitable_inputs;
  if (input_layout_ == "BSH") {
    Shape splitable_query{1, 0, 2};
    Shape splitable_key{1, 0, 2};
    Shape splitable_value{1, 0, 2};
    splitable_inputs = {splitable_query, splitable_key, splitable_value};

  } else if (input_layout_ == "BNSD") {
    Shape splitable_query{1, 2, 0, 0};
    Shape splitable_key{1, 2, 0, 0};
    Shape splitable_value{1, 2, 0, 0};
    splitable_inputs = {splitable_query, splitable_key, splitable_value};
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
  }
  for (auto index = static_cast<size_t>(ops::kPromptFlashAttentionInputAttnMaskIndex); index < optinal_inputs.size();
       index++) {
    if (optinal_inputs[index]) {
      (void)splitable_inputs.emplace_back(optinal_op_strategies[index]);
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

void PromptFlashAttentionInfo::ReComputeBatchSplitFlagList() {
  if (optinal_inputs.empty()) {
    SetOptinalInputs();
  }
  split_flag_list_[ops::kPromptFlashAttentionInputQueryIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputKeyIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputValueIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputAttnMaskIndex] =
    (optinal_inputs[ops::kPromptFlashAttentionInputAttnMaskIndex] && atten_mask_rank > rank_2);
  split_flag_list_[ops::kPromptFlashAttentionInputPaddingMaskIndex] =
    (optinal_inputs[ops::kPromptFlashAttentionInputPaddingMaskIndex] && padding_mask_rank > rank_2);
  split_flag_list_[ops::kPromptFlashAttentionInputActualSeqLengthsIndex] =
    optinal_inputs[ops::kPromptFlashAttentionInputActualSeqLengthsIndex];
  split_flag_list_[ops::kPromptFlashAttentionInputActualSeqLengthsKvIndex] =
    optinal_inputs[ops::kPromptFlashAttentionInputActualSeqLengthsKvIndex];
  split_flag_list_[ops::kPromptFlashAttentionInputDeqScale1Index] = false;
  split_flag_list_[ops::kPromptFlashAttentionInputQuantScale1Index] = false;
  split_flag_list_[ops::kPromptFlashAttentionInputDeqScale2Index] = false;
  split_flag_list_[ops::kPromptFlashAttentionInputQuantScale2Index] = false;
  split_flag_list_[ops::kPromptFlashAttentionInputQuantOffset2Index] = false;
}

Status PromptFlashAttentionInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  if (mirror_ops_.empty()) {
    // No need to insert mirror ops
    return SUCCESS;
  }
  for (size_t i = mirror_ops_.size(); i < ops::kPromptFlashAttentionInputsNum; ++i) {
    // Push empty mirror op for optional input
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  return SUCCESS;
}

void PromptFlashAttentionInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    prim->set_attr(kAttrHeadNum, MakeValue(head_num_ / mp_));
  }
}

REGISTER(PromptFlashAttentionInfo);
}  // namespace parallel
}  // namespace mindspore
