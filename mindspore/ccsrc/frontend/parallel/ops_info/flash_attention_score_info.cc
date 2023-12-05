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
constexpr size_t kInputQKVBatchDimBSH = 0;
constexpr size_t kInputQKVSeqDimBSH = 1;
constexpr size_t kInputQKVHiddenDimBSH = 2;
constexpr size_t kInputQKVBatchDimBNSD = 0;
constexpr size_t kInputQKVHeadNumDimBNSD = 1;
constexpr size_t kInputQKVSeqDimBNSD = 2;
constexpr size_t kInputQKVHeadSizeDimBNSD = 3;
constexpr char kInputLayoutBSH[] = "BSH";
constexpr char kInputLayoutBNSD[] = "BNSD";
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

void FlashAttentionScoreInfo::InitIsInputPassed() {
  is_input_passed_.resize(input_value_.size());
  for (size_t i = 0; i < input_value_.size(); ++i) {
    is_input_passed_[i] = (input_value_[i] == nullptr || !input_value_[i]->isa<None>());
  }
}

size_t FlashAttentionScoreInfo::GetStrategyRealIndex(size_t index) {
  if (index >= is_input_passed_.size() || !is_input_passed_[index]) {
    MS_LOG(INTERNAL_EXCEPTION) << name_ << ": GetStrategyRealIndex failed, index is " << index;
  }
  auto real_index = -1;
  for (size_t i = 0; i <= index; ++i) {
    if (is_input_passed_[i]) {
      ++real_index;
    }
  }
  return real_index;
}

void FlashAttentionScoreInfo::InitExpectedStrategies() {
  expect_strategies_ = Strategies(ops::kFlashAttentionScoreInputsNum);
  if (input_layout_ == kInputLayoutBSH) {
    expect_strategies_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_split_num_, s1_split_num_, n1_split_num_};
    expect_strategies_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_split_num_, 1, n2_split_num_};
    expect_strategies_[ops::kFlashAttentionScoreInputValueIndex] = {batch_split_num_, 1, n2_split_num_};
  } else {
    expect_strategies_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_split_num_, n1_split_num_, s1_split_num_, 1};
    expect_strategies_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_split_num_, n2_split_num_, 1, 1};
    expect_strategies_[ops::kFlashAttentionScoreInputValueIndex] = {batch_split_num_, n2_split_num_, 1, 1};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    int64_t real_shift_s1_split_num = real_shift_have_s1_dim_ ? s1_split_num_ : 1;
    expect_strategies_[ops::kFlashAttentionScoreInputRealShiftIndex] = {batch_split_num_, n1_split_num_,
                                                                        real_shift_s1_split_num, 1};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex]) {
    expect_strategies_[ops::kFlashAttentionScoreInputDropMaskIndex] = {batch_split_num_, n1_split_num_, s1_split_num_,
                                                                       1};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputPaddingMaskIndex]) {
    expect_strategies_[ops::kFlashAttentionScoreInputPaddingMaskIndex] = {};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    auto attn_mask_shape = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
    if (attn_mask_shape.size() == kSizeTwo) {
      // attn_mask_shape: (S1, S2)
      expect_strategies_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {s1_split_num_, 1};
    } else if (attn_mask_shape.size() == kSizeFour) {
      // attn_mask_shape: (B, N1, S1, S2) or (B, 1, S1, S2)
      auto attn_mask_n1_split_num = attn_mask_shape[kIndex1] == 1 ? 1 : n1_split_num_;
      expect_strategies_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {batch_split_num_, attn_mask_n1_split_num,
                                                                         s1_split_num_, 1};
    }
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputPrefixIndex]) {
    expect_strategies_[ops::kFlashAttentionScoreInputPrefixIndex] = {batch_split_num_};
  }
  expect_strategies_.erase(std::remove(expect_strategies_.begin(), expect_strategies_.end(), Shape{}),
                           expect_strategies_.end());
}

void FlashAttentionScoreInfo::InitInputsTensorMap() {
  inputs_tensor_map_ = std::vector<Shape>(ops::kFlashAttentionScoreInputsNum);
  int64_t kv_head_num_map = kv_split_ ? dev_matrix_n1_dim_ : -1;
  if (input_layout_ == kInputLayoutBSH) {
    inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex] = {dev_matrix_batch_dim_, dev_matrix_s1_dim_,
                                                                    dev_matrix_n1_dim_};
    inputs_tensor_map_[ops::kFlashAttentionScoreInputKeyIndex] = {dev_matrix_batch_dim_, -1, kv_head_num_map};
    inputs_tensor_map_[ops::kFlashAttentionScoreInputValueIndex] = {dev_matrix_batch_dim_, -1, kv_head_num_map};

  } else {
    inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex] = {dev_matrix_batch_dim_, dev_matrix_n1_dim_,
                                                                    dev_matrix_s1_dim_, -1};
    inputs_tensor_map_[ops::kFlashAttentionScoreInputKeyIndex] = {dev_matrix_batch_dim_, kv_head_num_map, -1, -1};
    inputs_tensor_map_[ops::kFlashAttentionScoreInputValueIndex] = {dev_matrix_batch_dim_, kv_head_num_map, -1, -1};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    auto real_shift_s1_map = real_shift_have_s1_dim_ ? dev_matrix_s1_dim_ : -1;
    inputs_tensor_map_[ops::kFlashAttentionScoreInputRealShiftIndex] = {dev_matrix_batch_dim_, dev_matrix_n1_dim_,
                                                                        real_shift_s1_map, -1};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex]) {
    inputs_tensor_map_[ops::kFlashAttentionScoreInputDropMaskIndex] = {dev_matrix_batch_dim_, dev_matrix_n1_dim_,
                                                                       dev_matrix_s1_dim_, -1};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputPaddingMaskIndex]) {
    inputs_tensor_map_[ops::kFlashAttentionScoreInputPaddingMaskIndex] = {};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    auto attn_mask_shape = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
    if (attn_mask_shape.size() == kSizeTwo) {
      // attn_mask_shape: (S1, S2)
      inputs_tensor_map_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {dev_matrix_s1_dim_, -1};
    } else if (attn_mask_shape.size() == kSizeFour) {
      // attn_mask_shape: (B, N1, S1, S2) or (B, 1, S1, S2)
      auto attn_mask_n1_map = attn_mask_shape[kIndex1] == 1 ? -1 : dev_matrix_n1_dim_;
      inputs_tensor_map_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {dev_matrix_batch_dim_, attn_mask_n1_map,
                                                                         dev_matrix_s1_dim_, -1};
    }
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputPrefixIndex]) {
    inputs_tensor_map_[ops::kFlashAttentionScoreInputPrefixIndex] = {dev_matrix_batch_dim_};
  }
  inputs_tensor_map_.erase(std::remove(inputs_tensor_map_.begin(), inputs_tensor_map_.end(), Shape{}),
                           inputs_tensor_map_.end());
}

void FlashAttentionScoreInfo::InitSplittableInputs() {
  splittable_inputs_ = std::vector<Shape>(ops::kFlashAttentionScoreInputsNum);
  int64_t batch_group = 3;
  int64_t s1_group = 2;
  int64_t n1_group = 1;
  int64_t n2_group = kv_split_ ? n1_group : 0;
  if (input_layout_ == kInputLayoutBSH) {
    splittable_inputs_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_group, s1_group, n1_group};
    splittable_inputs_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_group, 0, n2_group};
    splittable_inputs_[ops::kFlashAttentionScoreInputValueIndex] = {batch_group, 0, n2_group};

  } else {
    splittable_inputs_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_group, n1_group, s1_group, 0};
    splittable_inputs_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_group, n2_group, 0, 0};
    splittable_inputs_[ops::kFlashAttentionScoreInputValueIndex] = {batch_group, n2_group, 0, 0};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    auto real_shift_s1_group = real_shift_have_s1_dim_ ? s1_group : 0;
    splittable_inputs_[ops::kFlashAttentionScoreInputRealShiftIndex] = {batch_group, n1_group, real_shift_s1_group, 0};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex]) {
    splittable_inputs_[ops::kFlashAttentionScoreInputDropMaskIndex] = {batch_group, n1_group, s1_group, 0};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputPaddingMaskIndex]) {
    splittable_inputs_[ops::kFlashAttentionScoreInputPaddingMaskIndex] = {};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    auto attn_mask_shape = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
    if (attn_mask_shape.size() == kSizeTwo) {
      // attn_mask_shape: (S1, S2)
      splittable_inputs_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {s1_group, -1};
    } else if (attn_mask_shape.size() == kSizeFour) {
      // attn_mask_shape: (B, N1, S1, S2) or (B, 1, S1, S2)
      auto attn_mask_n1_group = attn_mask_shape[kIndex1] == 1 ? 0 : n1_group;
      splittable_inputs_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {batch_group, attn_mask_n1_group, s1_group, 1};
    }
    splittable_inputs_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {1, 0, 0, 0};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputPrefixIndex]) {
    splittable_inputs_[ops::kFlashAttentionScoreInputPrefixIndex] = {batch_group};
  }
  splittable_inputs_.erase(std::remove(splittable_inputs_.begin(), splittable_inputs_.end(), Shape{}),
                           splittable_inputs_.end());
}

Status FlashAttentionScoreInfo::GetAttrs() {
  InitIsInputPassed();
  head_num_ = GetIntAttr(kAttrHeadNum);
  keep_prob_ = GetFloatAttr(kAttrKeepProb);
  has_drop_mask_input_ = is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex];
  input_layout_ = GetStringAttr(kAttrInputLayout);
  if (input_layout_ == kInputLayoutBSH) {
    auto q_hidden_size = inputs_shape_[ops::kFlashAttentionScoreInputQueryIndex][kInputQKVHiddenDimBSH];
    auto k_hidden_size = inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex][kInputQKVHiddenDimBSH];
    kv_split_ = q_hidden_size != k_hidden_size * head_num_;
  } else if (input_layout_ == kInputLayoutBNSD) {
    auto k_head_num = inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex][kInputQKVHeadNumDimBNSD];
    kv_split_ = k_head_num != 1;
  } else {
    MS_LOG(ERROR) << name_ << ": The attribute 'input_layout' must be either " << kInputLayoutBSH << " or "
                  << kInputLayoutBNSD << ", but got " << input_layout_;
    return FAILED;
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    auto real_shift_s1_dim =
      inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputRealShiftIndex)).at(kIndex3);
    real_shift_have_s1_dim_ = real_shift_s1_dim > 1;
  }
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
  if (key_strategy != value_strategy) {
    MS_LOG(ERROR) << name_ << ": The in_strategy both of 'key'( " << key_strategy << ") and 'value'" << value_strategy
                  << ") must be same.";
    return FAILED;
  }
  int64_t s2_split_num;
  if (input_layout_ == kInputLayoutBSH) {
    if (head_num_ % query_strategy[kInputQKVHiddenDimBSH] != 0) {
      MS_LOG(ERROR) << name_ << ": head_num % query_strategy[" << kInputQKVHiddenDimBSH << "] must be 0, but got "
                    << head_num_ << "(head_num) and " << query_strategy[kInputQKVHiddenDimBSH] << "(query_strategy["
                    << kInputQKVHiddenDimBSH << "])";
      return FAILED;
    }
    if (!kv_split_ && key_strategy[kInputQKVHiddenDimBSH] != 1) {
      MS_LOG(ERROR) << name_ << ": Under the MQA，the hidden-dim of input 'key' cannot be split.";
      return FAILED;
    }
    batch_split_num_ = query_strategy[kInputQKVBatchDimBSH];
    n1_split_num_ = query_strategy[kInputQKVHiddenDimBSH];
    s1_split_num_ = query_strategy[kInputQKVSeqDimBSH];
    n2_split_num_ = key_strategy[kInputQKVHiddenDimBSH];
    s2_split_num = key_strategy[kInputQKVSeqDimBSH];
  } else {
    if (head_num_ % query_strategy[kInputQKVHeadNumDimBNSD] != 0) {
      MS_LOG(ERROR) << name_ << ": head_num % query_strategy[" << kInputQKVHeadNumDimBNSD << "] must be 0, but got "
                    << head_num_ << "(head_num) and " << query_strategy[kInputQKVHeadNumDimBNSD] << "(query_strategy["
                    << kInputQKVHeadNumDimBNSD << "])";
      return FAILED;
    }
    batch_split_num_ = query_strategy[kInputQKVBatchDimBNSD];
    n1_split_num_ = query_strategy[kInputQKVHeadNumDimBNSD];
    s1_split_num_ = query_strategy[kInputQKVSeqDimBNSD];
    n2_split_num_ = key_strategy[kInputQKVHeadNumDimBNSD];
    s2_split_num = key_strategy[kInputQKVSeqDimBNSD];
  }
  if (s2_split_num != 1) {
    MS_LOG(ERROR) << name_ << ": The S-Dimention of input 'key' cannot be split, but got the strategy of key is "
                  << key_strategy;
    return FAILED;
  }
  if (kv_split_ && n1_split_num_ != n2_split_num_) {
    MS_LOG(ERROR) << name_ << ": The split num of N1-dim and N2-dim must be equal if N2 > 1, but got " << n1_split_num_
                  << " and " << n2_split_num_;
  }

  InitExpectedStrategies();
  if (strategies != expect_strategies_) {
    MS_LOG(ERROR) << name_ << ": The input strategy must be " << expect_strategies_ << ", but got " << strategies;
    return FAILED;
  }

  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferDevMatrixShape() {
  if (input_layout_ == kInputLayoutBSH) {
    dev_matrix_shape_ = {batch_split_num_, s1_split_num_, n1_split_num_};
    dev_matrix_batch_dim_ = 2;
    dev_matrix_s1_dim_ = 1;
    dev_matrix_n1_dim_ = 0;
  } else {
    dev_matrix_shape_ = {batch_split_num_, n1_split_num_, s1_split_num_};
    dev_matrix_batch_dim_ = 2;
    dev_matrix_n1_dim_ = 1;
    dev_matrix_s1_dim_ = 0;
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferTensorMap() {
  InitInputsTensorMap();
  outputs_tensor_map_.push_back({dev_matrix_batch_dim_, dev_matrix_n1_dim_, dev_matrix_s1_dim_, -1});  // softmax_max
  outputs_tensor_map_.push_back({dev_matrix_batch_dim_, dev_matrix_n1_dim_, dev_matrix_s1_dim_, -1});  // softmax_sum
  outputs_tensor_map_.push_back({-1});                                                                 // softmax_out
  outputs_tensor_map_.push_back(inputs_tensor_map_[0]);                                                // attention_out
  return SUCCESS;
}

void FlashAttentionScoreInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    auto clone_prim = prim->Clone();
    MS_EXCEPTION_IF_NULL(prim);
    clone_prim->set_attr(kAttrHeadNum, MakeValue(head_num_ / n1_split_num_));
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
  InitSplittableInputs();
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, splittable_inputs_, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for dependent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No valid strategy.";
  }
  return sp_vector;
}

void FlashAttentionScoreInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_ = std::vector<bool>(inputs_shape_.size(), true);
}

Status FlashAttentionScoreInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }
  // Insert empty OperatorInfo for optional input
  size_t cur_index = 0;
  std::vector<OperatorVector> real_mirror_ops(input_value_.size(), OperatorVector());
  for (size_t i = 0; i < input_value_.size(); ++i) {
    if (is_input_passed_[i]) {
      real_mirror_ops[i] = mirror_ops_[cur_index++];
    }
    mirror_ops_ = real_mirror_ops;
  }
  return SUCCESS;
}

REGISTER(FlashAttentionScoreInfo);
}  // namespace parallel
}  // namespace mindspore
