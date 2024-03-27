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
#include <tuple>
#include <map>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "mindspore/core/ops/flash_attention_score.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/nn_ops.h"

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
constexpr int64_t kLoadBalanceSplitNum = 2;
enum OpAttrUpdateMode : int64_t {
  kLeftUpToLeftUp = 0,
  kLeftUpToRightDown = 1,
  kRightDownToRightDown = 2,
};
const std::vector<int64_t> needCompressAttnMask = {ops::kSparseLeftUpCausal, ops::kSparseRightDownCausal,
                                                   ops::kSparseBand};
const std::map<int64_t, int64_t> opAttrUpdateMap = {{ops::kSparseDefaultMask, kLeftUpToLeftUp},
                                                    {ops::kSparseLeftUpCausal, kLeftUpToRightDown},
                                                    {ops::kSparseRightDownCausal, kRightDownToRightDown},
                                                    {ops::kSparseBand, kRightDownToRightDown}};

size_t GetNonMonadInputSize(const CNodePtr &cnode) {
  size_t cnode_non_monad_size = cnode->size();
  for (auto &input : cnode->inputs()) {
    if (HasAbstractMonad(input)) {
      cnode_non_monad_size--;
    }
  }
  return cnode_non_monad_size;
}

int64_t NewSeedGeneration() {
  static int64_t seed_generation = 0;
  ++seed_generation;
  return seed_generation;
}

int64_t LongAdd(int64_t base, int64_t shift) {
  int64_t result;
  if (shift > 0) {
    if (base > INT_MAX - shift) {
      result = INT_MAX;
    } else {
      result = base + shift;
    }
  } else {
    if (base < INT_MIN - shift) {
      result = INT_MIN;
    } else {
      result = base + shift;
    }
  }
  return result;
}
}  // namespace

void FlashAttentionScoreInfo::UpdateDropoutGenMaskSliceShapeAndSeed(const CNodePtr &dropout_gen_mask_cnode) {
  if (!IsPrimitiveCNode(dropout_gen_mask_cnode, prim::kPrimDropoutGenMask)) {
    return;
  }

  // Update seed according rank_id for DropoutGenMask
  PrimitivePtr prim = GetCNodePrimitive(dropout_gen_mask_cnode);
  auto seed_0 = GetValue<int64_t>(prim->GetAttr(SEED0));
  auto seed_1 = GetValue<int64_t>(prim->GetAttr(SEED1));
  int64_t rank_id = g_device_manager->rank_index_in_stage();
  int64_t seed_bias = 0;
  // When seed and seed2 are both 0, ensure that the 0th card in each group has the same result
  if (seed_0 == 0 && seed_1 == 0) {
    seed_bias = NewSeedGeneration();
  }
  MS_EXCEPTION_IF_ZERO("repeated_calc_num_", repeated_calc_num_);
  if (repeated_num_in_dev_matrix_right_) {
    seed_bias += rank_id / repeated_calc_num_;
  } else {
    int64_t device_num = stage_device_size_;
    MS_EXCEPTION_IF_ZERO("device_num", device_num);
    seed_bias += rank_id % (device_num / repeated_calc_num_);
  }
  auto clone_prim = prim->Clone();
  clone_prim->set_attr(SEED0, MakeValue<int64_t>(seed_0 + seed_bias));
  clone_prim->set_attr(SEED1, MakeValue<int64_t>(seed_1 + seed_bias));
  auto func_graph = dropout_gen_mask_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->SetEdge(dropout_gen_mask_cnode, 0, NewValueNode(clone_prim)->cast<AnfNodePtr>());

  // Update slice shape for DropoutGenMask and Reshape
  Shape input_slice_shape = inputs_tensor_info_.at(ops::kFlashAttentionScoreInputDropMaskIndex).slice_shape();
  constexpr int64_t BITS_NUM_PER_BYTE = 8;
  input_slice_shape[input_slice_shape.size() - 1] *= BITS_NUM_PER_BYTE;  // Restores the shape of DropoutGenMask input
  size_t cnode_non_monad_size = GetNonMonadInputSize(dropout_gen_mask_cnode);
  if (cnode_non_monad_size != DROPOUT_GEN_MASK_CNODE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of dropout gen mask cnode's inputs must be " << DROPOUT_GEN_MASK_CNODE_INPUT_SIZE;
  }
  if (!IsValueNode<ValueTuple>(dropout_gen_mask_cnode->input(kIndex1))) {
    MS_LOG(EXCEPTION) << "The input[1] of dropout gen mask cnode is not ValueTuple.";
  }
  ValuePtr new_shape = MakeValue(input_slice_shape);
  AnfNodePtr val = NewValueNode(new_shape);
  manager->SetEdge(dropout_gen_mask_cnode, kIndex1, val);
  MS_LOG(DEBUG) << "The input slice shape dropout is " << ShapeToString(input_slice_shape);
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
    auto real_shift_batch_split_num = real_shift_have_batch_dim_ ? batch_split_num_ : 1;
    expect_strategies_[ops::kFlashAttentionScoreInputRealShiftIndex] = {real_shift_batch_split_num, n1_split_num_,
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
    int64_t s1_split_num_attn_mask = is_attn_mask_compressed_ ? 1 : s1_split_num_;
    if (attn_mask_shape.size() == kSizeTwo) {
      // attn_mask_shape: (S1, S2)
      expect_strategies_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {s1_split_num_attn_mask, 1};
    } else if (attn_mask_shape.size() == kSizeFour) {
      // attn_mask_shape: (B, N1, S1, S2) or (B, 1, S1, S2)
      auto attn_mask_n1_split_num = attn_mask_have_n1_dim_ ? n1_split_num_ : 1;
      auto attn_batch_split_num = attn_mask_have_batch_dim_ ? batch_split_num_ : 1;
      expect_strategies_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {attn_batch_split_num, attn_mask_n1_split_num,
                                                                         s1_split_num_attn_mask, 1};
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
    auto real_shift_batch_map = real_shift_have_batch_dim_ ? dev_matrix_batch_dim_ : -1;
    inputs_tensor_map_[ops::kFlashAttentionScoreInputRealShiftIndex] = {real_shift_batch_map, dev_matrix_n1_dim_,
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
    int64_t dev_matrix_s1_dim_attn_mask = is_attn_mask_compressed_ ? -1 : dev_matrix_s1_dim_;
    if (attn_mask_shape.size() == kSizeTwo) {
      // attn_mask_shape: (S1, S2)
      inputs_tensor_map_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {dev_matrix_s1_dim_attn_mask, -1};
    } else if (attn_mask_shape.size() == kSizeFour) {
      // attn_mask_shape: (B, N1, S1, S2) or (B, 1, S1, S2)
      auto attn_mask_batch_map = attn_mask_have_batch_dim_ ? dev_matrix_batch_dim_ : -1;
      auto attn_mask_n1_map = attn_mask_have_n1_dim_ ? dev_matrix_n1_dim_ : -1;
      inputs_tensor_map_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {attn_mask_batch_map, attn_mask_n1_map,
                                                                         dev_matrix_s1_dim_attn_mask, -1};
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
  scale_value_ = GetFloatAttr(kAttrScaleValue);
  pre_tokens_ = GetIntAttr(kAttrPreTokens);
  next_tokens_ = GetIntAttr(kAttrNextTokens);
  input_layout_ = GetStringAttr(kAttrInputLayout);
  sparse_mode_ = GetIntAttr(kAttrSparseMode);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  enable_load_balance_ = ms_context->get_param<bool>(MS_CTX_ENABLE_FLASH_ATTENTION_LOAD_BALANCE);
  is_attn_mask_compressed_ =
    std::find(needCompressAttnMask.begin(), needCompressAttnMask.end(), sparse_mode_) != needCompressAttnMask.end();
  need_update_op_attrs_mode_ = sparse_mode_ != ops::kSparseAllMask;
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
    auto real_shift_batch_dim =
      inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputRealShiftIndex)).at(kIndex0);
    real_shift_have_batch_dim_ = real_shift_batch_dim > 1;
  }

  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    auto attn_mask_shape = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
    if (attn_mask_shape.size() == kSizeFour) {
      attn_mask_have_batch_dim_ = attn_mask_shape.at(kIndex0) > 1;
      attn_mask_have_n1_dim_ = attn_mask_shape.at(kIndex1) > 1;
    }
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
    return FAILED;
  }

  InitExpectedStrategies();
  if (strategies != expect_strategies_) {
    MS_LOG(ERROR) << name_ << ": The input strategy must be " << expect_strategies_ << ", but got " << strategies;
    return FAILED;
  }

  return SUCCESS;
}

Status FlashAttentionScoreInfo::CheckStrategyForDynamicShape(const StrategyPtr &) {
  for (auto &cnode : cnodes_) {
    // If DropoutGenMask -> Reshape -> FlashAttentionScore
    auto reshape_node = cnode->input(ops::kFlashAttentionScoreInputDropMaskIndex + 1);
    MS_EXCEPTION_IF_NULL(reshape_node);
    if (!IsPrimitiveCNode(reshape_node, prim::kPrimReshape)) {
      continue;
    }

    MS_LOG(ERROR)
      << name_ << ": it does not support dynamic shape if it need to replace dst-shape for reshape, the inputs' shape: "
      << ShapesToString(inputs_shape_);
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

std::vector<int64_t> FlashAttentionScoreInfo::GetSplitIdAndRank() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  int64_t seq_dim = SizeToLong(dev_matrix_shape_.size()) - dev_matrix_s1_dim_ - 1;
  if (dev_matrix.GetDevicesAlongDim(seq_dim, &group_devices) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " get group devices along dim " << seq_dim << " failed.";
  }
  auto iter = std::find(group_devices.begin(), group_devices.end(), rank);
  if (iter == group_devices.end()) {
    MS_LOG(EXCEPTION) << "FlashAttentionScore S1 sequence parallel get split id failed. "
                      << "rank " << rank << " not in group " << group_devices;
  }
  int64_t split_id = iter - group_devices.begin();
  int64_t target_split_id = s1_split_num_ - split_id - 1;
  int64_t target_rank_id = group_devices[target_split_id];
  return std::vector<int64_t>({rank, target_rank_id, split_id, target_split_id});
}

std::tuple<int64_t, int64_t> FlashAttentionScoreInfo::GetAttentionMaskAttrs(const int64_t split_id,
                                                                            const int64_t split_num) {
  int64_t kv_seq_length;
  int64_t q_seq_length;
  if (input_layout_ == kInputLayoutBSH) {
    kv_seq_length = inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex][kInputQKVSeqDimBSH];
    q_seq_length = inputs_shape_[ops::kFlashAttentionScoreInputQueryIndex][kInputQKVSeqDimBSH];
  } else {
    kv_seq_length = inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex][kInputQKVSeqDimBNSD];
    q_seq_length = inputs_shape_[ops::kFlashAttentionScoreInputQueryIndex][kInputQKVSeqDimBNSD];
  }
  int64_t q_len_each_split = q_seq_length / split_num;
  int64_t new_pre_tokens;
  if (sparse_mode_ == ops::kSparseDefaultMask || sparse_mode_ == ops::kSparseBand) {
    new_pre_tokens = pre_tokens_;
  } else if (sparse_mode_ == ops::kSparseLeftUpCausal) {
    new_pre_tokens = q_seq_length;
  } else {
    new_pre_tokens = kv_seq_length;
  }
  int64_t new_next_tokens =
    (sparse_mode_ == ops::kSparseDefaultMask || sparse_mode_ == ops::kSparseBand) ? next_tokens_ : 0;
  switch (opAttrUpdateMap.at(sparse_mode_)) {
    case kLeftUpToLeftUp:
      new_pre_tokens = LongAdd(new_pre_tokens, -split_id * q_len_each_split);
      new_next_tokens = LongAdd(new_next_tokens, split_id * q_len_each_split);
      break;
    case kLeftUpToRightDown:
      new_pre_tokens = LongAdd(new_pre_tokens, (kv_seq_length - (split_id + 1) * q_len_each_split));
      new_next_tokens = LongAdd(new_next_tokens, -(kv_seq_length - (split_id + 1) * q_len_each_split));
      break;
    case kRightDownToRightDown:
      new_pre_tokens = LongAdd(new_pre_tokens, (split_num - split_id - 1) * (q_seq_length / split_num));
      new_next_tokens = LongAdd(new_next_tokens, -(split_num - split_id - 1) * (q_seq_length / split_num));
      break;
    default:
      MS_LOG(EXCEPTION) << "Invalid sparse mode " << sparse_mode_ << ", sparse mode should be one of [0, 2, 3, 4].";
  }
  return std::make_tuple(new_pre_tokens, new_next_tokens);
}

void FlashAttentionScoreInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    auto clone_prim = prim->Clone();
    MS_EXCEPTION_IF_NULL(clone_prim);
    clone_prim->set_attr(kAttrHeadNum, MakeValue(head_num_ / n1_split_num_));
    if (s1_split_num_ > 1 && !enable_load_balance_ && need_update_op_attrs_mode_) {
      std::vector<int64_t> split_info = GetSplitIdAndRank();
      int64_t split_id = split_info[kIndex2];
      int64_t new_pre_tokens, new_next_tokens;
      std::tie(new_pre_tokens, new_next_tokens) = GetAttentionMaskAttrs(split_id, s1_split_num_);
      int64_t new_sparse_mode = is_attn_mask_compressed_ ? ops::kSparseBand : sparse_mode_;
      clone_prim->set_attr(kAttrSparseMode, MakeValue(new_sparse_mode));
      clone_prim->set_attr(kAttrPreTokens, MakeValue(new_pre_tokens));
      clone_prim->set_attr(kAttrNextTokens, MakeValue(new_next_tokens));
    }
    auto manager = cnode->func_graph()->manager();
    manager->SetEdge(cnode, 0, NewValueNode(clone_prim)->cast<AnfNodePtr>());

    // If DropoutGenMask -> Reshape -> FlashAttentionScore, replace its.
    auto reshape_node = cnode->input(ops::kFlashAttentionScoreInputDropMaskIndex + 1);
    MS_EXCEPTION_IF_NULL(reshape_node);
    if (!IsPrimitiveCNode(reshape_node, prim::kPrimReshape)) {
      continue;
    }
    auto reshape_cnode = reshape_node->cast<CNodePtr>();
    if (!IsPrimitiveCNode(reshape_cnode->input(kIndex1), prim::kPrimDropoutGenMask)) {
      continue;
    }
    auto dropout_gen_mask_cnode = reshape_cnode->input(kIndex1)->cast<CNodePtr>();
    // Update slice_shape for ReShape
    Shape input_slice_shape = inputs_tensor_info_.at(ops::kFlashAttentionScoreInputDropMaskIndex).slice_shape();
    ValuePtr new_shape = MakeValue(input_slice_shape);
    AnfNodePtr val = NewValueNode(new_shape);
    manager->SetEdge(reshape_cnode, kIndex2, val);
    // Update slice shape and seed for DropoutGenMask
    UpdateDropoutGenMaskSliceShapeAndSeed(dropout_gen_mask_cnode);
  }
}

void FlashAttentionScoreInfo::LoadBalanceSplitAlongSeqDim(size_t input_index, GenerateGraph *gen_g,
                                                          AnfNodePtr *split_node, AnfNodePtr *keep_node,
                                                          AnfNodePtr *exchange_node) {
  OperatorAttrs split_attrs;
  int64_t q_split_axis;
  switch (input_index) {
    case ops::kFlashAttentionScoreInputQueryIndex:
      q_split_axis = (input_layout_ == kInputLayoutBSH) ? kInputQKVSeqDimBSH : kInputQKVSeqDimBNSD;
      split_attrs = {std::make_pair(AXIS, MakeValue(q_split_axis)),
                     std::make_pair(OUTPUT_NUM, MakeValue(kLoadBalanceSplitNum))};
      *split_node = gen_g->PushBack({gen_g->NewOpInst(SPLIT, split_attrs), gen_g->virtual_input_node()});
      *keep_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(0)});
      *exchange_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(1)});
      break;
    case ops::kFlashAttentionScoreInputRealShiftIndex:
    case ops::kFlashAttentionScoreInputDropMaskIndex:
      if (is_input_passed_[input_index]) {
        split_attrs = {std::make_pair(AXIS, MakeValue<int64_t>(kInputQKVSeqDimBNSD)),
                       std::make_pair(OUTPUT_NUM, MakeValue(kLoadBalanceSplitNum))};
        *split_node = gen_g->PushBack({gen_g->NewOpInst(SPLIT, split_attrs), gen_g->virtual_input_node()});
        *keep_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(0)});
        *exchange_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(1)});
      } else {
        *keep_node = gen_g->virtual_input_node();
        *exchange_node = gen_g->virtual_input_node();
      }
      break;
    case ops::kFlashAttentionScoreInputAttnMaskIndex:
      if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex] && !is_attn_mask_compressed_) {
        auto attn_mask_shape = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
        if (attn_mask_shape.size() == kSizeTwo) {
          split_attrs = {std::make_pair(AXIS, MakeValue<int64_t>(0)),
                         std::make_pair(OUTPUT_NUM, MakeValue(kLoadBalanceSplitNum))};
        } else {
          split_attrs = {std::make_pair(AXIS, MakeValue<int64_t>(2)),
                         std::make_pair(OUTPUT_NUM, MakeValue(kLoadBalanceSplitNum))};
        }
        *split_node = gen_g->PushBack({gen_g->NewOpInst(SPLIT, split_attrs), gen_g->virtual_input_node()});
        *keep_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(0)});
        *exchange_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(1)});
      } else {
        *keep_node = gen_g->virtual_input_node();
        *exchange_node = gen_g->virtual_input_node();
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "Invalid input index. Only 0(query), 3(real_shift), 4(drop_mask) and 6(attn_mask)"
                        << "support sequence dim parallel, but got " << input_index;
  }
}

void FlashAttentionScoreInfo::LoadBalanceExchange(const int64_t all_gather_idx, const Group &group,
                                                  const AnfNodePtr &input_node, AnfNodePtr *exchange_node,
                                                  GenerateGraph *gen_g) {
  OperatorAttrs all_gather_attrs = {std::make_pair(GROUP, MakeValue(group.name()))};
  OperatorAttrs all_gather_split_attrs = {std::make_pair(AXIS, MakeValue<int64_t>(0)),
                                          std::make_pair(OUTPUT_NUM, MakeValue(kLoadBalanceSplitNum))};
  auto all_gather_node = gen_g->PushBack({gen_g->NewOpInst(ALL_GATHER, all_gather_attrs), input_node});
  auto split_node = gen_g->PushBack({gen_g->NewOpInst(SPLIT, all_gather_split_attrs), all_gather_node});
  *exchange_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), split_node, CreatInt64Imm(all_gather_idx)});
}

void FlashAttentionScoreInfo::GetFlashAttentionScoreOpNode(int64_t split_id, int64_t split_num, const AnfNodePtr &q,
                                                           const AnfNodePtr &real_shift, const AnfNodePtr &drop_mask,
                                                           const AnfNodePtr &attn_mask, AnfNodePtr *fa_op,
                                                           GenerateGraph *gen_g) {
  int64_t new_sparse_mode = is_attn_mask_compressed_ ? ops::kSparseBand : sparse_mode_;
  int64_t new_pre_tokens, new_next_tokens;
  if (!need_update_op_attrs_mode_) {
    new_pre_tokens = pre_tokens_;
    new_next_tokens = next_tokens_;
  } else {
    std::tie(new_pre_tokens, new_next_tokens) = GetAttentionMaskAttrs(split_id, split_num);
  }
  OperatorAttrs fa_attrs = {std::make_pair(HEAD_NUM, MakeValue(head_num_ / n1_split_num_)),
                            std::make_pair(KEEP_PROB, MakeValue(keep_prob_)),
                            std::make_pair(SCALE_VALUE, MakeValue(scale_value_)),
                            std::make_pair(PRE_TOKENS, MakeValue(new_pre_tokens)),
                            std::make_pair(NEXT_TOKENS, MakeValue(new_next_tokens)),
                            std::make_pair(INNER_PRECISE, MakeValue<int64_t>(0)),
                            std::make_pair(INPUT_LAYOUT, MakeValue(input_layout_)),
                            std::make_pair(SPARSE_MODE, MakeValue<int64_t>(new_sparse_mode))};
  *fa_op = gen_g->PushBack({gen_g->NewOpInst(FLASH_ATTENTION_SCORE, fa_attrs), q, gen_g->virtual_input_node(),
                            gen_g->virtual_input_node(), real_shift, drop_mask, gen_g->virtual_input_node(), attn_mask,
                            gen_g->virtual_input_node()});
}

std::vector<std::pair<AnfNodePtr, int64_t>> FlashAttentionScoreInfo::ReplaceGraphGetInputNodes(
  const AnfNodePtr &q_split, const AnfNodePtr &real_shift_split, const AnfNodePtr &drop_mask_split,
  const AnfNodePtr &attn_mask_split, const AnfNodePtr &flash_attention_score_keep,
  const AnfNodePtr &flash_attention_score_target) {
  std::pair<AnfNodePtr, int64_t> real_shift_input;
  if (is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    real_shift_input = std::make_pair(real_shift_split, kIndex4);
  } else {
    real_shift_input = std::make_pair(flash_attention_score_keep, kIndex4);
  }
  std::pair<AnfNodePtr, int64_t> drop_mask_input;
  if (is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex]) {
    drop_mask_input = std::make_pair(drop_mask_split, kIndex5);
  } else {
    drop_mask_input = std::make_pair(flash_attention_score_keep, kIndex5);
  }
  std::pair<AnfNodePtr, int64_t> attn_mask_input;
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex] && !is_attn_mask_compressed_) {
    attn_mask_input = std::make_pair(attn_mask_split, kIndex7);
  } else {
    attn_mask_input = std::make_pair(flash_attention_score_keep, kIndex7);
  }

  std::vector<std::pair<AnfNodePtr, int64_t>> inputs_nodes = {std::make_pair(q_split, kIndex1),
                                                              std::make_pair(flash_attention_score_keep, kIndex2),
                                                              std::make_pair(flash_attention_score_keep, kIndex3),
                                                              real_shift_input,
                                                              drop_mask_input,
                                                              std::make_pair(flash_attention_score_keep, kIndex6),
                                                              attn_mask_input,
                                                              std::make_pair(flash_attention_score_keep, kIndex8),
                                                              std::make_pair(flash_attention_score_target, kIndex2),
                                                              std::make_pair(flash_attention_score_target, kIndex3)};
  if (!is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    inputs_nodes.push_back(std::make_pair(flash_attention_score_target, kIndex4));
  }
  if (!is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex]) {
    inputs_nodes.push_back(std::make_pair(flash_attention_score_target, kIndex5));
  }
  inputs_nodes.push_back(std::make_pair(flash_attention_score_target, kIndex6));
  if (!is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex] || is_attn_mask_compressed_) {
    inputs_nodes.push_back(std::make_pair(flash_attention_score_target, kIndex7));
  }
  inputs_nodes.push_back(std::make_pair(flash_attention_score_target, kIndex8));
  return inputs_nodes;
}

Status FlashAttentionScoreInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    return FAILED;
  }
  CheckGlobalDeviceManager();
  std::vector<int64_t> split_info = GetSplitIdAndRank();
  int64_t rank_id = split_info[kIndex0];
  int64_t target_rank_id = split_info[kIndex1];
  int64_t split_id = split_info[kIndex2];
  int64_t target_split_id = split_info[kIndex3];
  Group group;
  RankList swap_group_devices = {rank_id, target_rank_id};
  if (g_device_manager->CreateGroup(swap_group_devices, &group) != SUCCESS) {
    MS_LOG(ERROR) << "Create communication group for " << swap_group_devices << " failed";
    return FAILED;
  }

  AnfNodePtr q_split, q_keep, q_exchange;
  LoadBalanceSplitAlongSeqDim(ops::kFlashAttentionScoreInputQueryIndex, &gen_g, &q_split, &q_keep, &q_exchange);
  AnfNodePtr real_shift_split, real_shift_keep, real_shift_exchange;
  LoadBalanceSplitAlongSeqDim(ops::kFlashAttentionScoreInputRealShiftIndex, &gen_g, &real_shift_split, &real_shift_keep,
                              &real_shift_exchange);
  AnfNodePtr drop_mask_split, drop_mask_keep, drop_mask_exchange;
  LoadBalanceSplitAlongSeqDim(ops::kFlashAttentionScoreInputDropMaskIndex, &gen_g, &drop_mask_split, &drop_mask_keep,
                              &drop_mask_exchange);
  AnfNodePtr attn_mask_split, attn_mask_keep, attn_mask_exchange;
  LoadBalanceSplitAlongSeqDim(ops::kFlashAttentionScoreInputAttnMaskIndex, &gen_g, &attn_mask_split, &attn_mask_keep,
                              &attn_mask_exchange);

  AnfNodePtr flash_attention_score_keep;
  GetFlashAttentionScoreOpNode(split_id * kLoadBalanceSplitNum, s1_split_num_ * kLoadBalanceSplitNum, q_keep,
                               real_shift_keep, drop_mask_keep, attn_mask_keep, &flash_attention_score_keep, &gen_g);
  auto softmax_max_keep = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_keep,
                                          CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxMaxIndex)});
  auto softmax_sum_keep = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_keep,
                                          CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxSumIndex)});
  auto softmax_out_keep = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_keep,
                                          CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxOutIndex)});
  auto attention_out_keep = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_keep,
                                            CreatInt64Imm(ops::kFlashAttentionScoreOutputAttentionOutIndex)});

  const int64_t all_gather_idx = (split_id < target_split_id) ? 1 : 0;
  AnfNodePtr q_target;
  LoadBalanceExchange(all_gather_idx, group, q_exchange, &q_target, &gen_g);
  AnfNodePtr real_shift_target;
  if (is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    LoadBalanceExchange(all_gather_idx, group, real_shift_exchange, &real_shift_target, &gen_g);
  } else {
    real_shift_target = gen_g.virtual_input_node();
  }
  AnfNodePtr drop_mask_target;
  if (is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex]) {
    LoadBalanceExchange(all_gather_idx, group, drop_mask_exchange, &drop_mask_target, &gen_g);
  } else {
    drop_mask_target = gen_g.virtual_input_node();
  }
  AnfNodePtr attn_mask_target;
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex] && !is_attn_mask_compressed_) {
    LoadBalanceExchange(all_gather_idx, group, attn_mask_exchange, &attn_mask_target, &gen_g);
  } else {
    attn_mask_target = gen_g.virtual_input_node();
  }

  AnfNodePtr flash_attention_score_target;
  GetFlashAttentionScoreOpNode(target_split_id * kLoadBalanceSplitNum + 1, s1_split_num_ * kLoadBalanceSplitNum,
                               q_target, real_shift_target, drop_mask_target, attn_mask_target,
                               &flash_attention_score_target, &gen_g);
  auto softmax_max_target = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_target,
                                            CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxMaxIndex)});
  auto softmax_sum_target = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_target,
                                            CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxSumIndex)});
  auto attention_out_target = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_target,
                                              CreatInt64Imm(ops::kFlashAttentionScoreOutputAttentionOutIndex)});

  AnfNodePtr attention_out_exchange;
  LoadBalanceExchange(all_gather_idx, group, attention_out_target, &attention_out_exchange, &gen_g);

  int64_t softmax_concat_axis = kInputQKVSeqDimBNSD;
  auto softmax_max_maketuple =
    gen_g.PushBack({NewValueNode(prim::kPrimMakeTuple), softmax_max_keep, softmax_max_target});
  auto softmax_max =
    gen_g.PushBack({gen_g.NewOpInst(CONCAT), softmax_max_maketuple, CreatInt64Imm(softmax_concat_axis)});
  auto softmax_sum_maketuple =
    gen_g.PushBack({NewValueNode(prim::kPrimMakeTuple), softmax_sum_keep, softmax_sum_target});
  auto softmax_sum =
    gen_g.PushBack({gen_g.NewOpInst(CONCAT), softmax_sum_maketuple, CreatInt64Imm(softmax_concat_axis)});
  int64_t attention_out_concat_axis = (input_layout_ == kInputLayoutBSH) ? kInputQKVSeqDimBSH : kInputQKVSeqDimBNSD;
  auto attention_out_maketuple =
    gen_g.PushBack({NewValueNode(prim::kPrimMakeTuple), attention_out_keep, attention_out_exchange});
  auto attention_out =
    gen_g.PushBack({gen_g.NewOpInst(CONCAT), attention_out_maketuple, CreatInt64Imm(attention_out_concat_axis)});
  auto output_maketuple =
    gen_g.PushBack({NewValueNode(prim::kPrimMakeTuple), softmax_max, softmax_sum, softmax_out_keep, attention_out});

  std::vector<std::pair<AnfNodePtr, int64_t>> inputs_nodes =
    ReplaceGraphGetInputNodes(q_split, real_shift_split, drop_mask_split, attn_mask_split, flash_attention_score_keep,
                              flash_attention_score_target);

  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(inputs_nodes, output_maketuple));
  return SUCCESS;
}

ReplaceGraphPtr FlashAttentionScoreInfo::replace_graph(const CNodePtr &cnode) {
  if (s1_split_num_ > 1 && enable_load_balance_) {
    if (ComputeReplaceGraph(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << "FlashAttentionScore S1 sequence parallel with load balance get replace graph failed";
    }
  }
  return replace_graph_;
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
