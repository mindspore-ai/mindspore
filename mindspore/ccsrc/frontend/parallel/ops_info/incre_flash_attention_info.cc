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
#include "ops/ops_func_impl/incre_flash_attention.h"
#include "ops/op_enum.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/parallel/ops_info/incre_flash_attention_info.h"
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr size_t kInputBatchDim = 0;
constexpr size_t kInputQuerySeqDimBSH = 1;
constexpr size_t kInputQueryHiddenDimBSH = 2;
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
constexpr size_t kRank2 = 2;
constexpr int64_t kAntiquantStratDimBSHLayout = 1;
constexpr int64_t kAntiquantStratDimBNSDLayout = 1;

class FASInputLayoutMode {
 public:
  static std::string ConvertEnumToString(int64_t id) {
    static const std::vector<std::string> input_layout_modes = {"BSH", "BNSD", "SBH", "BSND", "TND"};
    if (id < 0 || id >= static_cast<int64_t>(input_layout_modes.size())) {
      MS_LOG(EXCEPTION) << "Invalid input layout mode " << id;
      return "";
    }
    return input_layout_modes[id];
  }
};

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
      if (index == ops::kIncreFlashAttentionInputPseShiftIndex && valid_input_index < inputs_shape_.size()) {
        pse_shift_rank_ = inputs_shape_[valid_input_index].size();
      }
      valid_input_index++;
    } else {
      if (optinal_input_ptr->isa<None>() || optinal_input_ptr->isa<StringImm>() || optinal_input_ptr->isa<Int64Imm>() ||
          optinal_input_ptr->isa<FP32Imm>()) {
        optinal_inputs_[index] = False;
      }
    }
  }
  if (atten_mask_rank_ > kRank4 || pse_shift_rank_ > kRank4) {
    MS_LOG(EXCEPTION) << "atten_mask or pse_shift got unexpected ranks: " << atten_mask_rank_ << " and "
                      << pse_shift_rank_ << ". Expecting ranks not greater than 4.";
  }

  Shape atten_mask_tensor_map(atten_mask_rank_, -1);
  Shape atten_mask_strategy_map(atten_mask_rank_, 0);
  Shape pse_shift_tensor_map(pse_shift_rank_, -1);
  Shape pse_shift_strategy_map(pse_shift_rank_, 0);
  if (optinal_inputs_[ops::kIncreFlashAttentionInputAttnMaskIndex]) {
    atten_mask_tensor_map[kInputBatchDim] = 1;
    atten_mask_strategy_map[kInputBatchDim] = 1;
  }
  if (pse_shift_rank_ >= kRank3) {
    pse_shift_tensor_map[kInputBatchDim] = 1;
    pse_shift_strategy_map[kInputBatchDim] = 1;
  }
  optinal_tensor_map_[ops::kIncreFlashAttentionInputAttnMaskIndex] = atten_mask_tensor_map;
  optinal_tensor_map_[ops::kIncreFlashAttentionInputPseShiftIndex] = pse_shift_tensor_map;
  optinal_op_strategies_[ops::kIncreFlashAttentionInputAttnMaskIndex] = atten_mask_strategy_map;
  optinal_op_strategies_[ops::kIncreFlashAttentionInputPseShiftIndex] = pse_shift_strategy_map;
}

Status IncreFlashAttentionInfo::GetAttrs() {
  auto head_num_opt = GetScalarValueFromInputsWithCheck<int64_t>(input_value_, name_, kAttrHeadNum);
  if (!head_num_opt.has_value()) {
    return FAILED;
  }
  head_num_ = head_num_opt.value();
  auto kv_head_num_opt = GetScalarValueFromInputsWithCheck<int64_t>(input_value_, name_, kAttrKVHeadNum);
  if (!kv_head_num_opt.has_value()) {
    return FAILED;
  }
  kv_head_num = kv_head_num_opt.value();
  auto input_layout_opt = GetScalarValueFromInputsWithCheck<int64_t>(input_value_, name_, kAttrInputLayout);
  if (!input_layout_opt.has_value()) {
    return FAILED;
  }
  auto input_layout_enum = input_layout_opt.value();
  input_layout_ = FASInputLayoutMode::ConvertEnumToString(input_layout_enum);
  SetOptinalInputs();
  return SUCCESS;
}

// The purpose of this function is to get the squeezed index of the optional inputs, e.g.
// input_0   input_1   input_2   (opt_input_3)   (opt_input_4)   opt_input_5
// 0         1         2          None               None              3
// when input 3 and 4 are not provided, the squeezed index of input 5 is 3
size_t IncreFlashAttentionInfo::GetSqueezedIndex(size_t original_index) {
  if (original_index >= optinal_inputs_.size()) {
    MS_LOG(WARNING) << "provided index [" << original_index << "] is out of range [" << optinal_inputs_.size() << "]";
    return -1;
  }
  size_t id_counter = 0;
  for (size_t index = kIndex1; index <= original_index; index++) {
    if (optinal_inputs_[index]) {
      id_counter++;
    }
  }
  return id_counter;
}

Status IncreFlashAttentionInfo::CheckAntiquantStrategy(const StrategyPtr &strategy, size_t input_index) {
  auto strategies = strategy->GetInputDim();
  if (!optinal_inputs_[input_index]) {
    return SUCCESS;
  }
  auto antiquant_idx = GetSqueezedIndex(input_index);
  if (antiquant_idx >= 0) {
    if (input_layout_ == kAttrInputLayoutBSH) {
      auto antiquant_strategy = strategies[antiquant_idx];
      if (antiquant_strategy.size() != kRank2) {
        MS_LOG(ERROR) << "antiquant strategy length should be strictly 2 in BSH layout.";
        return FAILED;
      }
      if (antiquant_strategy[kIndex0] != kAntiquantStratDimBSHLayout) {
        MS_LOG(ERROR) << "antiquant strategy first dim should be strictly 1 in BSH layout.";
        return FAILED;
      }
      if (antiquant_strategy[kIndex1] != mp_) {
        MS_LOG(ERROR) << "antiquant strategy second dim should be strictly equal to the strategy value of the third "
                         "dim of Query in BSH layout.";
        return FAILED;
      }
    } else if (input_layout_ == kAttrInputLayoutBNSD) {
      auto antiquant_strategy = strategies[antiquant_idx];
      if (antiquant_strategy.size() != kRank4) {
        MS_LOG(ERROR) << "antiquant strategy length should be strictly 4 in BNSD layout.";
        return FAILED;
      }
      if ((antiquant_strategy[kIndex0] != kAntiquantStratDimBNSDLayout) ||
          (antiquant_strategy[kIndex2] != kAntiquantStratDimBNSDLayout) ||
          (antiquant_strategy[kIndex3] != kAntiquantStratDimBNSDLayout)) {
        MS_LOG(ERROR) << "antiquant strategy first, third, and forth dim should be strictly 1 in BNSD layout.";
        return FAILED;
      }
      if (antiquant_strategy[kIndex1] != mp_) {
        MS_LOG(ERROR) << "antiquant strategy second dim should be strictly to the strategy value of the third dim of "
                         "Query in BNSD layout.";
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status IncreFlashAttentionInfo::CheckAttenMaskStrategy(const StrategyPtr &strategy, size_t input_index) {
  auto strategies = strategy->GetInputDim();
  if (!optinal_inputs_[input_index]) {
    return SUCCESS;
  }
  auto atten_mask_idx = GetSqueezedIndex(input_index);
  auto atten_mask_strategy = strategies[atten_mask_idx];
  auto query_strategy = strategies[ops::kIncreFlashAttentionInputQueryIndex];
  if (atten_mask_idx >= kIndex0) {
    if (atten_mask_strategy[kInputBatchDim] != query_strategy[kInputBatchDim]) {
      MS_LOG(ERROR) << "atten_mask strategy batch dim should be same.";
      return FAILED;
    }
    for (size_t index = kIndex1; index < atten_mask_strategy.size(); index++) {
      if (!CheckStrategyOnIndex(atten_mask_strategy[index], 1, "dims except batch", "atten_mask")) {
        return FAILED;
      }
    }
  }
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

  if (query_strategy != key_strategy || query_strategy != value_strategy) {
    MS_LOG(ERROR) << "For " << name_ << " : The in_strategy among 'query' , 'key' and 'value' must be same.";
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
    dp_ = query_strategy[kInputBatchDim];
    mp_ = query_strategy[kInputQueryHiddenDimBSH];
  } else if (input_layout_ == kAttrInputLayoutBNSD) {
    if (!CheckStrategyOnIndex(query_strategy[kInputQuerySeqDimBNSD], 1, "S-Dimention", "query") ||
        !CheckStrategyOnIndex(query_strategy[kInputQueryHiddenDimBNSD], 1, "D-Dimention", "query")) {
      return FAILED;
    }
    dp_ = query_strategy[kInputBatchDim];
    mp_ = query_strategy[kInputQueryNDimBNSD];
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }
  if (CheckAntiquantStrategy(strategy, ops::kIncreFlashAttentionInputAntiquantScale) != SUCCESS) {
    MS_LOG(ERROR) << "Check strategy for antiquant_scale failed";
    return FAILED;
  }
  if (CheckAntiquantStrategy(strategy, ops::kIncreFlashAttentionInputAntiquantOffset) != SUCCESS) {
    MS_LOG(ERROR) << "Check strategy for antiquant_offset failed";
    return FAILED;
  }
  if (CheckAttenMaskStrategy(strategy, ops::kIncreFlashAttentionInputAttnMaskIndex) != SUCCESS) {
    MS_LOG(ERROR) << "Check strategy for atten mask failed";
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
      if (index == ops::kIncreFlashAttentionInputAntiquantScale ||
          index == ops::kIncreFlashAttentionInputAntiquantOffset) {
        if (input_layout_ == kAttrInputLayoutBSH) {
          (void)inputs_tensor_map_.emplace_back(Shape{-1, 0});
          continue;
        } else if (input_layout_ == kAttrInputLayoutBNSD) {
          (void)inputs_tensor_map_.emplace_back(Shape{-1, 0, -1, -1});
          continue;
        }
      }
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
  split_flag_list_[ops::kIncreFlashAttentionInputPseShiftIndex] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputActualSeqLengths] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputDequantScale1] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputQuantScale1] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputDequantScale2] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputQuantScale2] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputQuantOffset2] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputAntiquantScale] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputAntiquantOffset] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputBlockTable] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputKvPaddingSize] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputNumHeads] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputInputLayout] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputScaleValue] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputNumKeyValueHeads] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputBlockSize] = false;
  split_flag_list_[ops::kIncreFlashAttentionInputInnerPrecise] = false;
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
    SetValueInputToCNode<int64_t>(cnode, ops::kIncreFlashAttentionInputNumHeads + 1, head_num_ / mp_);
    SetValueInputToCNode<int64_t>(cnode, ops::kIncreFlashAttentionInputNumKeyValueHeads + 1, kv_head_num / mp_);
    cnode->set_input(0, NewValueNode(clone_prim)->cast<AnfNodePtr>());
  }
}

REGISTER(IncreFlashAttentionInfo);
}  // namespace parallel
}  // namespace mindspore
