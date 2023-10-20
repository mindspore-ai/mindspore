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

#include "frontend/parallel/ops_info/flash_attention_score_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/core/ops/flash_attention_score.h"
#include "mindspore/core/ops/array_ops.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr size_t kInputQueryBatchDim = 0;
constexpr size_t kInputQuerySeqDim = 1;
constexpr size_t kInputQueryHiddenDim = 2;
constexpr char kAttrHeadNum[] = "head_num";
constexpr char kAttrKeepProb[] = "keep_prob";
int64_t SEED_NUM = 1;

size_t GetNonMonadInputSize(const CNodePtr &cnode) {
  size_t cnode_non_monad_size = cnode->size();
  for (auto &input : cnode->inputs()) {
    if (HasAbstractMonad(input)) {
      cnode_non_monad_size--;
    }
  }
  return cnode_non_monad_size;
}

void ReplaceOneOp(const Operator &replace_op, const CNodePtr &reshape_node) {
  FuncGraphPtr func_graph = reshape_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since manager is nullptr";
  }
  std::string instance_name = CreateInstanceName(reshape_node, 0);
  std::vector<AnfNodePtr> replace_input;
  replace_input = ReplaceOpInput(replace_op, instance_name, reshape_node);
  if (reshape_node->inputs().size() == RESHAPE_INPUT_SIZE) {
    replace_input.push_back(reshape_node->input(kIndex2));
  }
  CNodePtr replace_node = func_graph->NewCNode(replace_input);
  MS_EXCEPTION_IF_NULL(replace_node);
  ScopePtr scope = reshape_node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  replace_node->set_scope(scope);
  replace_node->set_in_forward_flag(true);
  replace_input[0]->set_scope(scope);
  auto prim = GetValueNode<PrimitivePtr>(replace_node->input(0));
  auto origin_prim = GetValueNode<PrimitivePtr>(reshape_node->input(0));
  SetUserAttrs(origin_prim->attrs(), prim);
  (void)manager->Replace(reshape_node, replace_node);
}

PrimitivePtr GetDropoutGenMaskPrim(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() != RESHAPE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of Reshape cnode's inputs must be " << RESHAPE_INPUT_SIZE;
  }

  AnfNodePtr dropout_gen_mask = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  if (!dropout_gen_mask->isa<CNode>()) {
    MS_LOG(INFO) << "Input is not a CNode, no need to replace";
    return nullptr;
  }

  auto dropout_gen_mask_cnode = dropout_gen_mask->cast<CNodePtr>();
  size_t cnode_non_monad_size = GetNonMonadInputSize(dropout_gen_mask_cnode);
  if (cnode_non_monad_size != DROPOUT_GEN_MASK_CNODE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of dropout gen mask cnode's inputs must be " << DROPOUT_GEN_MASK_CNODE_INPUT_SIZE;
  }
  if (!IsValueNode<Primitive>(dropout_gen_mask_cnode->input(0))) {
    MS_LOG(EXCEPTION) << "The input[0] of dropout gen mask cnode is not primitive";
  }

  auto value_node = dropout_gen_mask_cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto prim = value_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() != DROPOUT_GEN_MASK) {
    MS_LOG(EXCEPTION) << "The primitive name is not DropoutGenMask";
  }
  return prim;
}

void SetGenMaskShape(const CNodePtr &cnode, const Shape &input_slice_shape) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() != RESHAPE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of reshape cnode's inputs must be " << RESHAPE_INPUT_SIZE;
  }

  AnfNodePtr dropout_gen_mask = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  if (!dropout_gen_mask->isa<CNode>()) {
    MS_LOG(WARNING) << "The dropout do mask cnode's input[" << ops::kFlashAttentionScoreInputDropMaskIndex + 1
                    << "] is not a cnode.";
    return;
  }

  auto dropout_gen_mask_cnode = dropout_gen_mask->cast<CNodePtr>();
  size_t cnode_non_monad_size = GetNonMonadInputSize(dropout_gen_mask_cnode);
  if (cnode_non_monad_size != DROPOUT_GEN_MASK_CNODE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of dropout gen mask cnode's inputs must be " << DROPOUT_GEN_MASK_CNODE_INPUT_SIZE;
  }

  if (!IsValueNode<ValueTuple>(dropout_gen_mask_cnode->input(1))) {
    MS_LOG(EXCEPTION) << "The input[1] of dropout gen mask cnode is not ValueTuple.";
  }

  FuncGraphPtr func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure: AddNode error since manager is nullptr.";
  }
  ValuePtr new_shape = MakeValue(input_slice_shape);
  AnfNodePtr val = NewValueNode(new_shape);
  dropout_gen_mask_cnode->set_input(kIndex1, val);
}
}  // namespace

std::vector<Operator> FlashAttentionScoreInfo::GetDropoutGenMaskReplaceOp(const CNodePtr &cnode) {
  std::vector<Operator> replace_ops;
  MS_EXCEPTION_IF_NULL(cnode);
  PrimitivePtr prim = GetDropoutGenMaskPrim(cnode);
  if (prim == nullptr) {
    return replace_ops;
  }

  if (inputs_tensor_info_.empty()) {
    MS_LOG(EXCEPTION) << "The tensor info of FlashAttentionScore is empty";
  }

  if (cnode->inputs().size() != RESHAPE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of reshape cnode's inputs must be " << RESHAPE_INPUT_SIZE;
  }

  auto attr = prim->attrs();
  if ((attr.find(SEED0) == attr.end()) || (attr.find(SEED1) == attr.end())) {
    MS_LOG(EXCEPTION) << "The attrs of dropout gen mask must be have seed0 and seed1";
  }

  Shape input_slice_shape = inputs_tensor_info_[ops::kFlashAttentionScoreInputDropMaskIndex].slice_shape();
  input_slice_shape[input_slice_shape.size() - 1] *= 8;  // Restores the shape of DropoutGenMask input
  auto seed_0 = GetValue<int64_t>(attr[SEED0]);
  auto seed_1 = GetValue<int64_t>(attr[SEED1]);
  if ((seed_0 == 0) && (seed_1 == 0) && (repeated_calc_num_ > 1)) {
    seed_0 = SEED_NUM;
    seed_1 = SEED_NUM;
    SEED_NUM++;
  } else {
    SetGenMaskShape(cnode, input_slice_shape);
    MS_LOG(DEBUG) << "The input slice shape dropout is " << ShapeToString(input_slice_shape);
    return replace_ops;
  }
  ValuePtr new_shape = MakeValue(input_slice_shape);
  Attr attr_0 = std::make_pair(SEED0, MakeValue(seed_0));
  Attr attr_1 = std::make_pair(SEED1, MakeValue(seed_1));
  OperatorAttrs attrs = {attr_0, attr_1};
  Attr param_0 = std::make_pair(SHAPE, new_shape);
  Attr param_1 = std::make_pair(KEEP_PROB, MakeValue(keep_prob_));
  OperatorParams params = {std::make_pair(param_0, 1), std::make_pair(param_1, 2)};
  OperatorArgs args = std::make_pair(attrs, params);
  Operator replace_op = {std::make_pair(DROPOUT_GEN_MASK, args)};
  replace_ops.push_back(replace_op);
  return replace_ops;
}

Status FlashAttentionScoreInfo::GetAttrs() {
  head_num_ = GetIntAttr(kAttrHeadNum);
  keep_prob_ = GetFloatAttr(kAttrKeepProb);
  has_drop_mask_input_ = !common::IsFloatEqual(keep_prob_, 1.0);
  return SUCCESS;
}

Status FlashAttentionScoreInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto strategies = strategy->GetInputDim();
  auto query_strategy = strategies[ops::kFlashAttentionScoreInputQueryIndex];
  auto key_strategy = strategies[ops::kFlashAttentionScoreInputKeyIndex];
  auto value_strategy = strategies[ops::kFlashAttentionScoreInputValueIndex];

  if (query_strategy != key_strategy || query_strategy != value_strategy) {
    MS_LOG(ERROR) << name_ << ": The in_strategy among of 'query'(" << query_strategy << "), 'key'( " << key_strategy
                  << ") and 'value'" << value_strategy << ") must be same.";
    return FAILED;
  }
  if (query_strategy[kInputQuerySeqDim] != 1) {
    MS_LOG(ERROR) << name_
                  << ": The S-Dimention of input 'query' cannot be split, but got strategy: " << query_strategy;
    return FAILED;
  }
  if (head_num_ % query_strategy[kInputQueryHiddenDim] != 0) {
    MS_LOG(ERROR) << name_ << ": head_num % query_strategy[2] must be 0, but got " << head_num_ << "(head_num) and "
                  << query_strategy[kInputQueryHiddenDim] << "(query_strategy[2])";
    return FAILED;
  }
  dp_ = query_strategy[kInputQueryBatchDim];
  mp_ = query_strategy[kInputQueryHiddenDim];
  if (has_drop_mask_input_) {
    Shape expect_drop_mask_strategy{dp_, mp_, 1, 1};
    auto drop_mask_strategy = strategies[ops::kFlashAttentionScoreInputDropMaskIndex];
    if (drop_mask_strategy != expect_drop_mask_strategy) {
      MS_LOG(ERROR) << name_ << ": The in_strategy for 'drop_mask' must be " << expect_drop_mask_strategy
                    << ", but got " << drop_mask_strategy;
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = {dp_, mp_};
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferTensorMap() {
  (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});       // query
  (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});       // key
  (void)inputs_tensor_map_.emplace_back(Shape{1, -1, 0});       // value
  (void)inputs_tensor_map_.emplace_back(Shape{1, -1, -1, -1});  // attn_mask
  // drop_mask
  if (has_drop_mask_input_) {
    (void)inputs_tensor_map_.emplace_back(Shape{1, 0, -1, -1});
  }

  outputs_tensor_map_.push_back({1, -1, 0});      // attention_out
  outputs_tensor_map_.push_back({1, 0, -1, -1});  // softmax_max
  outputs_tensor_map_.push_back({1, 0, -1, -1});  // softmax_sum
  return SUCCESS;
}

void FlashAttentionScoreInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    auto clone_prim = prim->Clone();
    MS_EXCEPTION_IF_NULL(prim);
    clone_prim->set_attr(kAttrHeadNum, MakeValue(head_num_ / mp_));
    cnode->set_input(0, NewValueNode(clone_prim)->cast<AnfNodePtr>());

    // If DropoutGenMask -> Reshape -> FlashAttentionScore, replace its.
    auto reshape_node = cnode->input(ops::kFlashAttentionScoreInputDropMaskIndex + 1);
    MS_EXCEPTION_IF_NULL(reshape_node);
    if (!IsPrimitiveCNode(reshape_node, prim::kPrimReshape)) {
      continue;
    }
    auto reshape_cnode = reshape_node->cast<CNodePtr>();
    // replace slice_shape for ReShape
    Shape input_slice_shape = inputs_tensor_info_[ops::kFlashAttentionScoreInputDropMaskIndex].slice_shape();
    ValuePtr new_shape = MakeValue(input_slice_shape);
    AnfNodePtr val = NewValueNode(new_shape);
    reshape_cnode->set_input(kIndex2, val);

    std::vector<Operator> replace_op = GetDropoutGenMaskReplaceOp(reshape_cnode);
    if (replace_op.empty()) {
      MS_LOG(DEBUG) << name_ << ": No need to replace dropout_gen_mask";
      continue;
    }
    ReplaceOneOp(replace_op[0], reshape_cnode->input(kIndex1)->cast<CNodePtr>());
  }
}

Status FlashAttentionScoreInfo::InferAsLossDivisor() {
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

std::vector<StrategyPtr> FlashAttentionScoreInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape splitable_query{1, 0, 2};
  Shape splitable_key{1, 0, 2};
  Shape splitable_value{1, 0, 2};
  Shape splitable_attn_mask{1, 0, 0, 0};
  Shape splitable_real_shift{};
  Shape splitable_padding_mask{};
  Shapes splitable_inputs = {splitable_query, splitable_key, splitable_value, splitable_attn_mask};
  if (has_drop_mask_input_) {
    (void)splitable_inputs.emplace_back(Shape{1, 2, 0, 0});
  }

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, splitable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for dependent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No valid strategy.";
  }
  return sp_vector;
}

void FlashAttentionScoreInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[ops::kFlashAttentionScoreInputQueryIndex] = true;
  split_flag_list_[ops::kFlashAttentionScoreInputKeyIndex] = true;
  split_flag_list_[ops::kFlashAttentionScoreInputValueIndex] = true;
  split_flag_list_[ops::kFlashAttentionScoreInputAttnMaskIndex] = true;
  if (has_drop_mask_input_) {
    split_flag_list_[ops::kFlashAttentionScoreInputDropMaskIndex] = has_drop_mask_input_;
  }
}

Status FlashAttentionScoreInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }
  for (size_t i = mirror_ops_.size(); i < ops::kFlashAttentionScoreInputsNum; ++i) {
    // Push empty mirror op for nums
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  return SUCCESS;
}

REGISTER(FlashAttentionScoreInfo);
}  // namespace parallel
}  // namespace mindspore
