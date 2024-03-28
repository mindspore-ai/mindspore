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
constexpr size_t kInputQuerySeqDimBSH = 1;
constexpr size_t kInputQueryHiddenDimBSH = 2;
constexpr size_t kInputBatchDim = 0;
constexpr size_t kInputQueryNDimBNSD = 1;
constexpr size_t kInputQuerySeqDimBNSD = 2;
constexpr size_t kInputQueryHiddenDimBNSD = 3;
constexpr char kAttrHeadNum[] = "num_heads";
constexpr char kAttrSparseMode[] = "sparse_mode";
constexpr char kAttrKVHeadNum[] = "num_key_value_heads";
constexpr char kAttrInputLayout[] = "input_layout";
constexpr size_t kRank2 = 2;
constexpr size_t kRank3 = 3;
constexpr size_t kSparseMode0 = 0;
enum class SparseMode { SPARSE_MODE_0, SPARSE_MODE_1, SPARSE_MODE_2 };
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
  optinal_inputs_.resize(ops::kPromptFlashAttentionInputsNum, true);
  optinal_tensor_map_.resize(ops::kPromptFlashAttentionInputsNum, {-1, -1});
  optinal_op_strategies_.resize(ops::kPromptFlashAttentionInputsNum, {0});
  size_t valid_input_index = 0;
  for (size_t index = 0; index < input_value_.size(); index++) {
    auto optinal_input_ptr = input_value_[index];
    if (optinal_input_ptr == nullptr || optinal_input_ptr->isa<tensor::Tensor>()) {
      if (index == ops::kPromptFlashAttentionInputAttnMaskIndex && valid_input_index < inputs_shape_.size()) {
        atten_mask_rank_ = inputs_shape_[valid_input_index].size();
      }
      if (index == ops::kPromptFlashAttentionInputPaddingMaskIndex && valid_input_index < inputs_shape_.size()) {
        padding_mask_rank_ = inputs_shape_[valid_input_index].size();
      }
      valid_input_index++;
    } else if (optinal_input_ptr->isa<None>()) {
      optinal_inputs_[index] = False;
    } else {
      TypePtr input_type = optinal_input_ptr->type();
      MS_EXCEPTION_IF_NULL(input_type);
      MS_EXCEPTION(TypeError) << "The given input at index: " << index
                              << "has an invalid data type: " << input_type->ReprString()
                              << ". The expected types are: Tensor or None.";
    }
  }

  Shape atten_mask_tensor_map(atten_mask_rank_, -1);
  Shape atten_mask_strategy_map(atten_mask_rank_, 0);
  Shape padding_mask_tensor_map(padding_mask_rank_, -1);
  Shape padding_mask_strategy_map(padding_mask_rank_, 0);
  if (atten_mask_rank_ >= kRank3 && sparse_mode_ == kSparseMode0) {
    atten_mask_tensor_map[0] = 1;
    atten_mask_strategy_map[0] = 1;
  }
  if (padding_mask_rank_ >= kRank3) {
    padding_mask_tensor_map[0] = 1;
    padding_mask_strategy_map[0] = 1;
  }
  optinal_tensor_map_[ops::kPromptFlashAttentionInputAttnMaskIndex] = atten_mask_tensor_map;
  optinal_tensor_map_[ops::kPromptFlashAttentionInputPaddingMaskIndex] = padding_mask_tensor_map;
  optinal_tensor_map_[ops::kPromptFlashAttentionInputActualSeqLengthsIndex] = {1};
  optinal_tensor_map_[ops::kPromptFlashAttentionInputActualSeqLengthsKvIndex] = {1};

  optinal_op_strategies_[ops::kPromptFlashAttentionInputAttnMaskIndex] = atten_mask_strategy_map;
  optinal_op_strategies_[ops::kPromptFlashAttentionInputPaddingMaskIndex] = padding_mask_strategy_map;
  optinal_op_strategies_[ops::kPromptFlashAttentionInputActualSeqLengthsIndex] = {1};
  optinal_op_strategies_[ops::kPromptFlashAttentionInputActualSeqLengthsKvIndex] = {1};
}

Status PromptFlashAttentionInfo::GetAttrs() {
  head_num_ = GetIntAttr(kAttrHeadNum);
  kv_head_num_ = GetIntAttr(kAttrKVHeadNum);
  input_layout_ = GetStringAttr(kAttrInputLayout);
  sparse_mode_ = GetIntAttr(kAttrSparseMode);
  SetOptinalInputs();
  return SUCCESS;
}

int PromptFlashAttentionInfo::GetSqueezedIndex(size_t original_index) {
  if (original_index >= optinal_inputs_.size()) {
    MS_LOG(WARNING) << "provided index [" << original_index << "] is out of range [" << optinal_inputs_.size() << "]";
    return -1;
  }
  int id_counter = 0;
  for (size_t index = 1; index <= original_index; index++) {
    if (optinal_inputs_[index]) {
      id_counter++;
    }
  }
  return id_counter;
}

Status PromptFlashAttentionInfo::CheckAttenMaskStrategy(const StrategyPtr &strategy, size_t input_index) {
  auto strategies = strategy->GetInputDim();
  if (!optinal_inputs_[input_index]) {
    return SUCCESS;
  }
  auto atten_mask_idx = GetSqueezedIndex(input_index);
  auto atten_mask_strategy = strategies[atten_mask_idx];
  auto query_strategy = strategies[ops::kPromptFlashAttentionInputAttnMaskIndex];
  if (atten_mask_idx >= 0) {
    if (atten_mask_strategy[kInputBatchDim] != query_strategy[kInputBatchDim]) {
      MS_LOG(ERROR) << "atten_mask strategy batch dim should be same.";
      return FAILED;
    }
    for (size_t index = 1; index < atten_mask_strategy.size(); index++) {
      if (!CheckStrategy(atten_mask_strategy[index], 1, "dims except batch", "atten_mask")) {
        return FAILED;
      }
    }
  }
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
    dp_ = query_strategy[kInputBatchDim];
    mp_ = query_strategy[kInputQueryHiddenDimBSH];
  } else if (input_layout_ == "BNSD") {
    if (!CheckStrategy(query_strategy[kInputQuerySeqDimBNSD], 1, "S-Dimention", "query") ||
        !CheckStrategy(query_strategy[kInputQueryHiddenDimBNSD], 1, "D-Dimention", "query")) {
      return FAILED;
    }
    dp_ = query_strategy[kInputBatchDim];
    mp_ = query_strategy[kInputQueryNDimBNSD];
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }
  if (optinal_inputs_.empty()) {
    SetOptinalInputs();
  }

  if (atten_mask_rank_ == kRank2 || sparse_mode_ != kSparseMode0) {
    if (!CheckStrategy(strategies[ops::kPromptFlashAttentionInputAttnMaskIndex][kInputBatchDim], 1, "B-Dimention",
                       "atten_mask")) {
      return FAILED;
    }
  }
  if (padding_mask_rank_ == kRank2) {
    if (!CheckStrategy(strategies[ops::kPromptFlashAttentionInputPaddingMaskIndex][kInputBatchDim], 1, "B-Dimention",
                       "padding_mask")) {
      return FAILED;
    }
  }
  if (CheckAttenMaskStrategy(strategy, ops::kPromptFlashAttentionInputAttnMaskIndex) != SUCCESS) {
    MS_LOG(ERROR) << "Check strategy for atten mask failed";
    return FAILED;
  }
  return SUCCESS;
}

Status PromptFlashAttentionInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = {dp_, mp_};
  return SUCCESS;
}

Status PromptFlashAttentionInfo::InferTensorMap() {
  if (optinal_inputs_.empty()) {
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
  for (auto index = static_cast<size_t>(ops::kPromptFlashAttentionInputAttnMaskIndex); index < optinal_inputs_.size();
       index++) {
    if (optinal_inputs_[index]) {
      (void)inputs_tensor_map_.emplace_back(optinal_tensor_map_[index]);
    }
  }
  return SUCCESS;
}

std::vector<StrategyPtr> PromptFlashAttentionInfo::GenerateOpStrategies(int64_t stage_id) {
  if (optinal_inputs_.empty()) {
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
  for (auto index = static_cast<size_t>(ops::kPromptFlashAttentionInputAttnMaskIndex); index < optinal_inputs_.size();
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

void PromptFlashAttentionInfo::ReComputeBatchSplitFlagList() {
  if (optinal_inputs_.empty()) {
    SetOptinalInputs();
  }
  split_flag_list_[ops::kPromptFlashAttentionInputQueryIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputKeyIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputValueIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputAttnMaskIndex] =
    (optinal_inputs_[ops::kPromptFlashAttentionInputAttnMaskIndex] && atten_mask_rank_ > kRank2 &&
     sparse_mode_ == kSparseMode0);
  split_flag_list_[ops::kPromptFlashAttentionInputPaddingMaskIndex] =
    (optinal_inputs_[ops::kPromptFlashAttentionInputPaddingMaskIndex] && padding_mask_rank_ > kRank2);
  split_flag_list_[ops::kPromptFlashAttentionInputActualSeqLengthsIndex] =
    optinal_inputs_[ops::kPromptFlashAttentionInputActualSeqLengthsIndex];
  split_flag_list_[ops::kPromptFlashAttentionInputActualSeqLengthsKvIndex] =
    optinal_inputs_[ops::kPromptFlashAttentionInputActualSeqLengthsKvIndex];
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
    MS_EXCEPTION_IF_NULL(prim);
    auto clone_prim = prim->Clone();
    clone_prim->set_attr(kAttrHeadNum, MakeValue(head_num_ / mp_));
    clone_prim->set_attr(kAttrKVHeadNum, MakeValue(kv_head_num_ / mp_));
    cnode->set_input(0, NewValueNode(clone_prim)->cast<AnfNodePtr>());
  }
}

REGISTER(PromptFlashAttentionInfo);
}  // namespace parallel
}  // namespace mindspore
