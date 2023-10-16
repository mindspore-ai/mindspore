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

Status PromptFlashAttentionInfo::GetAttrs() {
  head_num_ = GetIntAttr(kAttrHeadNum);
  input_layout_ = GetStringAttr(kAttrInputLayout);
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
  Shape expect_atten_mask_strategy{dp_, 1, 1, 1};
  auto atten_mask_strategy = strategies[ops::kPromptFlashAttentionInputAttnMaskIndex];
  if (atten_mask_strategy != expect_atten_mask_strategy) {
    MS_LOG(ERROR) << "For" << name_ << ": The in_strategy for 'atten_mask' must be " << expect_atten_mask_strategy
                  << ", but got " << atten_mask_strategy;
    return FAILED;
  }
  return SUCCESS;
}

Status PromptFlashAttentionInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = {dp_, mp_};
  return SUCCESS;
}

Status PromptFlashAttentionInfo::InferTensorMap() {
  if (input_layout_ == "BSH") {
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});       // query
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});       // key
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});       // value
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, -1, -1});  // attn_mask
    outputs_tensor_map_.push_back({1, -1, 0});                    // attention_out
  } else if (input_layout_ == "BNSD") {
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});   // query
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});   // key
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});   // value
    (void)inputs_tensor_map_.emplace_back(Shape{1, -1, -1, -1});  // attn_mask
    outputs_tensor_map_.push_back({1, 0, -1, -1});                // attention_out
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }
  return SUCCESS;
}

std::vector<StrategyPtr> PromptFlashAttentionInfo::GenerateOpStrategies(int64_t stage_id) {
  Shapes splitable_inputs;
  if (input_layout_ == "BSH") {
    Shape splitable_query{1, 0, 2};
    Shape splitable_key{1, 0, 2};
    Shape splitable_value{1, 0, 2};
    Shape splitable_attn_mask{1, 0, 0, 0};
    splitable_inputs = {splitable_query, splitable_key, splitable_value, splitable_attn_mask};
  } else if (input_layout_ == "BNSD") {
    Shape splitable_query{1, 2, 0, 0};
    Shape splitable_key{1, 2, 0, 0};
    Shape splitable_value{1, 2, 0, 0};
    Shape splitable_attn_mask{1, 0, 0, 0};
    splitable_inputs = {splitable_query, splitable_key, splitable_value, splitable_attn_mask};
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
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

// 只针对batch 是否可以切分
void PromptFlashAttentionInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[ops::kPromptFlashAttentionInputQueryIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputKeyIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputValueIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputAttnMaskIndex] = true;
  split_flag_list_[ops::kPromptFlashAttentionInputPaddingMaskIndex] = false;
  split_flag_list_[ops::kPromptFlashAttentionInputActualSeqLengthsIndex] = false;
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
