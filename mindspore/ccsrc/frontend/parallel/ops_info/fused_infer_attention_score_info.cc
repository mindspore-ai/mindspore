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

#include "frontend/parallel/ops_info/fused_infer_attention_score_info.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/core/ops/ops_func_impl/fused_infer_attention_score.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_enum.h"

namespace mindspore {
using mindspore::ops::FASInputLayoutMode;
using mindspore::ops::FusedInferAttentionScoreInputIndex;
namespace parallel {
namespace {
constexpr size_t kInputQueryBatchDimBSH = 0;
constexpr size_t kInputQuerySeqDimBSH = 1;
constexpr size_t kInputQueryHiddenDimBSH = 2;
constexpr size_t kInputQueryBatchDimBNSD = 0;
constexpr size_t kInputQueryNDimBNSD = 1;
constexpr size_t kInputQuerySeqDimBNSD = 2;
constexpr size_t kInputQueryHiddenDimBNSD = 3;
constexpr size_t kInputQueryBatchDimBSND = 0;
constexpr size_t kInputQuerySeqDimBSND = 1;
constexpr size_t kInputQueryNDimBSND = 2;
constexpr size_t kInputQueryHiddenDimBSND = 3;
constexpr size_t rank_2 = 2;
constexpr size_t rank_3 = 3;
}  // namespace

template <typename T>
void SetValueInputToCNode(const CNodePtr &cnode, size_t index, T value) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto inputs = cnode->inputs();
  if (index >= inputs.size()) {
    MS_LOG(EXCEPTION) << "The input index (" << index << ") is exceed of inputs size (" << inputs.size() << ").";
  }
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  auto value_node = NewValueNode(MakeValue(value));
  MS_EXCEPTION_IF_NULL(value_node);
  manager->SetEdge(cnode, index, value_node);
}

bool FusedInferAttentionScoreInfo::CheckStrategyOnIndex(int64_t strategy, int64_t true_value,
                                                        const std::string &dim_name, const std::string &input_name) {
  if (strategy != true_value) {
    MS_LOG(ERROR) << "For " << name_ << ": The " << dim_name << " of input " << input_name << " should be "
                  << true_value << ", but got strategy: " << strategy;
    return false;
  }
  return true;
}

void FusedInferAttentionScoreInfo::SetOptinalInputs() {
  optinal_inputs.resize(ops::kFusedInferAttentionScoreInputKvPaddingSizeIndex + 1, true);
  size_t valid_input_index = 3;
  for (size_t index = ops::kFusedInferAttentionScoreInputPseShiftIndex; index < input_value_.size(); index++) {
    auto optinal_input_ptr = input_value_[index];
    if (optinal_input_ptr == nullptr) {
      if (index == ops::kFusedInferAttentionScoreInputPseShiftIndex && valid_input_index < inputs_shape_new_.size()) {
        auto padding_mask_shape = inputs_shape_new_[valid_input_index]->GetAllElements();
        padding_mask_rank = padding_mask_shape[0].size();
      }
      if (index == ops::kFusedInferAttentionScoreInputAttnMaskIndex && valid_input_index < inputs_shape_new_.size()) {
        auto atten_mask_shape = inputs_shape_new_[valid_input_index]->GetAllElements();
        atten_mask_rank = atten_mask_shape[0].size();
      }
      valid_input_index++;
    } else {
      if (optinal_input_ptr->isa<None>()) {
        optinal_inputs[index] = False;
      }
    }
  }
  if (atten_mask_rank == rank_2) {
    optinal_tensor_map[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {-1, -1};
    optinal_op_strategies[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {0, 0};
  }
  if (atten_mask_rank == rank_3) {
    optinal_tensor_map[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {1, -1, -1};
    optinal_op_strategies[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {1, 0, 0};
  }
  if (padding_mask_rank == rank_2) {
    optinal_tensor_map[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {-1, -1};
    optinal_op_strategies[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {0, 0};
  }
  if (padding_mask_rank == rank_3) {
    optinal_tensor_map[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {1, -1, -1};
    optinal_op_strategies[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {1, 0, 0};
  }
}

void FusedInferAttentionScoreInfo::GenerateExpectStrategies() {
  expect_strategies = {{}, {}, {}, {dp_, 1, 1, 1}, {dp_, 1, 1}, {dp_}, {dp_}, {}, {}, {}, {}, {}, {}, {}, {}};
  if (atten_mask_rank == rank_2) {
    expect_strategies[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {1, 1};
  }
  if (atten_mask_rank == rank_3) {
    expect_strategies[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {dp_, 1, 1};
  }
  if (padding_mask_rank == rank_2) {
    expect_strategies[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {1, 1};
  }
  if (padding_mask_rank == rank_3) {
    expect_strategies[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {dp_, 1, 1};
  }
}

Status FusedInferAttentionScoreInfo::GetAttrs() {
  MS_EXCEPTION_IF_NULL(cnode_);
  auto inputs = cnode_->inputs();

  auto head_num_node = inputs[ops::kFusedInferAttentionScoreInputNumHeadsIndex + 1];
  MS_EXCEPTION_IF_NULL(head_num_node);
  if (!head_num_node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The head_num input is not a value node.";
  }
  auto head_num_value = head_num_node->cast<ValueNodePtr>()->value();
  MS_EXCEPTION_IF_NULL(head_num_value);
  head_num_ = GetValue<int64_t>(head_num_value);

  auto kv_head_node = inputs[ops::kFusedInferAttentionScoreInputNumKeyValueHeadsIndex + 1];
  MS_EXCEPTION_IF_NULL(kv_head_node);
  if (!kv_head_node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The NumKeyValueHeads input is not a value node.";
  }
  auto kv_head_value = kv_head_node->cast<ValueNodePtr>()->value();
  MS_EXCEPTION_IF_NULL(kv_head_value);
  kv_head_num = GetValue<int64_t>(kv_head_value);

  auto input_layout_node = inputs[ops::kFusedInferAttentionScoreInputLayoutIndex + 1];
  MS_EXCEPTION_IF_NULL(input_layout_node);
  if (!input_layout_node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The NumKeyValueHeads input is not a value node.";
  }
  auto input_layout_value = input_layout_node->cast<ValueNodePtr>()->value();
  MS_EXCEPTION_IF_NULL(input_layout_value);
  input_layout_ = GetValue<int64_t>(input_layout_value);
  SetOptinalInputs();
  return SUCCESS;
}

Status FusedInferAttentionScoreInfo::CheckStrategy(const StrategyPtr &strategy) {
  std::vector<std::vector<int64_t>> squashed_stra;
  std::vector<std::vector<int64_t>> squashed_shape;
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null.";
    return FAILED;
  }

  if (!strategy->HasTupleInTupleStrategy()) {
    MS_LOG(ERROR) << name_ << "for key and value, The strategy must be tuple in tuple.";
    return FAILED;
  }
  NewStrategies stra = strategy->GetInputNewDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": Strategy size must be larger than 1.";
    return FAILED;
  }
  auto query_strategys = stra[ops::kFusedInferAttentionScoreInputQueryIndex]->GetAllElements();
  auto query_strategy = query_strategys[0];
  auto key_strategys = stra[ops::kFusedInferAttentionScoreInputKeyIndex]->GetAllElements();
  auto value_strategys = stra[ops::kFusedInferAttentionScoreInputValueIndex]->GetAllElements();

  if (key_strategys.size() != value_strategys.size()) {
    MS_LOG(ERROR) << "For " << name_ << " : The num of in_strategy among 'key' and 'value' must be same.";
    return FAILED;
  }
  if (!std::equal(key_strategys.begin(), key_strategys.end(), value_strategys.begin())) {
    MS_LOG(ERROR) << "For " << name_ << " : The in_strategy among 'key' and 'value' must be same.";
    return FAILED;
  }
  // shapevalue and shapelist into one vector
  for (size_t i = 0; i < stra.size(); ++i) {
    if (stra[i]->is_list() != inputs_shape_new_[i]->is_list()) {
      MS_LOG(ERROR) << name_ << ": The strategy and shape must be both list or both value.";
      return FAILED;
    }
    auto shape_element = inputs_shape_new_[i]->GetAllElements();
    auto stra_element = stra[i]->GetAllElements();
    squashed_stra.insert(squashed_stra.end(), stra_element.begin(), stra_element.end());
    squashed_shape.insert(squashed_shape.end(), shape_element.begin(), shape_element.end());
  }
  if (CheckStrategyByVector(squashed_stra, squashed_shape) != SUCCESS) {
    return FAILED;
  }

  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      if (head_num_ % query_strategy[kInputQueryHiddenDimBSH] != 0) {
        MS_LOG(ERROR) << "For " << name_ << ": head_num % query_strategy[2] must be 0, but got " << head_num_
                      << "(head_num) and " << query_strategy[kInputQueryHiddenDimBSH] << "(query_strategy[2])";
        return FAILED;
      }
      dp_ = query_strategy[kInputQueryBatchDimBSH];
      mp_ = query_strategy[kInputQueryHiddenDimBSH];
      break;
    case FASInputLayoutMode::BNSD:
      if (!CheckStrategyOnIndex(query_strategy[kInputQueryHiddenDimBNSD], 1, "D-Dimention", "query")) {
        return FAILED;
      }
      dp_ = query_strategy[kInputQueryBatchDimBNSD];
      mp_ = query_strategy[kInputQueryNDimBNSD];
      break;
    case FASInputLayoutMode::BSND:
      if (!CheckStrategyOnIndex(query_strategy[kInputQueryHiddenDimBSND], 1, "D-Dimention", "query")) {
        return FAILED;
      }
      dp_ = query_strategy[kInputQueryBatchDimBSND];
      mp_ = query_strategy[kInputQueryNDimBSND];
      break;
    default:
      MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
      return FAILED;
  }

  if (optinal_inputs.empty()) {
    SetOptinalInputs();
  }
  return SUCCESS;
}

Status FusedInferAttentionScoreInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = {dp_, mp_};
  return SUCCESS;
}

Status FusedInferAttentionScoreInfo::InferTensorMap() {
  if (optinal_inputs.empty()) {
    SetOptinalInputs();
  }
  Shape validMap;
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      // (b, h) -> (dp_, mp_) -> (1, 0)
      validMap = Shape{1, -1, 0};  // query
      break;
    case FASInputLayoutMode::BNSD:
      // (b, n) -> (dp_, mp_) -> (1, 0)  BNSD -> (1 ,0, , )
      validMap = Shape{1, 0, -1, -1};
      break;
    case FASInputLayoutMode::BSND:
      // (b, n) -> (dp_, mp_) -> (1, 0)  BSND -> (1 , ,0, )
      validMap = Shape{1, -1, 0, -1};
      break;
    default:
      MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
      return FAILED;
  }
  std::vector<ShapeBasePtr> key_value_tensorist_map_idx;
  for (size_t i = 0; i < inputs_shape_new_[ops::kFusedInferAttentionScoreInputKeyIndex]->size(); i++) {
    key_value_tensorist_map_idx.emplace_back(std::make_shared<ShapeValue>(validMap));
  }
  inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(validMap));                    // query
  inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeList>(key_value_tensorist_map_idx));  // key
  inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeList>(key_value_tensorist_map_idx));  // value
  outputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(validMap));                   // attention_out
  outputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(Shape{1, 0, -1, -1}));        // softmax_lse

  for (auto index = static_cast<size_t>(ops::kFusedInferAttentionScoreInputPseShiftIndex);
       index < optinal_inputs.size(); index++) {
    if (optinal_inputs[index]) {
      if (index == ops::kFusedInferAttentionScoreInputAntiquantScaleIndex ||
          index == ops::kFusedInferAttentionScoreInputAntiquantOffsetIndex) {
        if (input_layout_ == FASInputLayoutMode::BSH) {  // (2, D) D=H/N
          (void)inputs_tensor_map_.emplace_back(Shape{-1, 0});
          continue;
        } else if (input_layout_ == FASInputLayoutMode::BNSD ||  // (2, N, 1, D)
                   input_layout_ == FASInputLayoutMode::BSND) {
          (void)inputs_tensor_map_.emplace_back(Shape{-1, 0, -1, -1});
          continue;
        }
      }
      (void)inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(optinal_tensor_map[index]));
    }
  }
  return SUCCESS;
}

Status FusedInferAttentionScoreInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_map_new_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor map is empty.";
    return FAILED;
  }

  if (outputs_tensor_map_new_[0]->empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  if (out_dev_matrix_shape_.empty()) {
    out_dev_matrix_shape_ = dev_matrix_shape_;
  }
  as_loss_divisor_ =
    ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape_, outputs_tensor_map_new_[0]->GetAllElements()[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_new_[0]->GetAllElements()[0])
               << ", loss divisor is " << as_loss_divisor_;
  return SUCCESS;
}

std::vector<StrategyPtr> FusedInferAttentionScoreInfo::GenerateOpStrategies(int64_t stage_id) { return {}; }

void FusedInferAttentionScoreInfo::ReComputeBatchSplitFlagList() {
  if (optinal_inputs.empty()) {
    SetOptinalInputs();
  }
  split_flag_list_[ops::kFusedInferAttentionScoreInputQueryIndex] = true;
  split_flag_list_[ops::kFusedInferAttentionScoreInputKeyIndex] = true;
  split_flag_list_[ops::kFusedInferAttentionScoreInputValueIndex] = true;
  split_flag_list_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] =
    (optinal_inputs[ops::kFusedInferAttentionScoreInputAttnMaskIndex] && atten_mask_rank > rank_2);
  split_flag_list_[ops::kFusedInferAttentionScoreInputPseShiftIndex] =
    (optinal_inputs[ops::kFusedInferAttentionScoreInputPseShiftIndex] && padding_mask_rank > rank_2);
  split_flag_list_[ops::kFusedInferAttentionScoreInputActualSeqLengthsIndex] =
    optinal_inputs[ops::kFusedInferAttentionScoreInputActualSeqLengthsIndex];
  split_flag_list_[ops::kFusedInferAttentionScoreInputActualSeqLengthsKvIndex] =
    optinal_inputs[ops::kFusedInferAttentionScoreInputActualSeqLengthsKvIndex];
  split_flag_list_[ops::kFusedInferAttentionScoreInputDequantScale1Index] = false;
  split_flag_list_[ops::kFusedInferAttentionScoreInputQuantScale1Index] = false;
  split_flag_list_[ops::kFusedInferAttentionScoreInputDequantScale2Index] = false;
  split_flag_list_[ops::kFusedInferAttentionScoreInputQuantScale2Index] = false;
  split_flag_list_[ops::kFusedInferAttentionScoreInputQuantOffset2Index] = false;
  split_flag_list_[ops::kFusedInferAttentionScoreInputAntiquantScaleIndex] = false;
  split_flag_list_[ops::kFusedInferAttentionScoreInputAntiquantOffsetIndex] = false;
  split_flag_list_[ops::kFusedInferAttentionScoreInputBlockTableIndex] = false;
}

Status FusedInferAttentionScoreInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  if (mirror_ops_.empty()) {
    // No need to insert mirror ops
    return SUCCESS;
  }
  for (size_t i = mirror_ops_.size(); i < ops::kFusedInferAttentionScoreInputKvPaddingSizeIndex + 1; ++i) {
    // Push empty mirror op for optional input
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  return SUCCESS;
}

void FusedInferAttentionScoreInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    auto clone_prim = prim->Clone();
    SetValueInputToCNode<int64_t>(cnode, ops::kFusedInferAttentionScoreInputNumHeadsIndex + 1, head_num_ / mp_);
    SetValueInputToCNode<int64_t>(cnode, ops::kFusedInferAttentionScoreInputNumKeyValueHeadsIndex + 1,
                                  kv_head_num / mp_);
    cnode->set_input(0, NewValueNode(clone_prim)->cast<AnfNodePtr>());
  }
}

REGISTER(FusedInferAttentionScoreInfo);
}  // namespace parallel
}  // namespace mindspore
