/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "ir/value.h"
#include "ir/core_ops_primitive.h"
#include "ir/tensor_new.h"
#include "include/frontend/parallel/graph_costmodel.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "primitive/nn_ops.h"
#include "primitive/array_ops.h"
#include "mindspore/ops/infer/ops_func_impl/flash_attention_score.h"
#include "include/op_enum.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
using mindspore::ops::FASInputLayoutMode;
namespace parallel {
namespace {
constexpr size_t kInputRealShiftSeqDim = 2;
constexpr size_t kInputDropMaskSeqDim = 2;
constexpr size_t kOutputSoftmaxSeqDim = 2;
constexpr int64_t kLoadBalanceSplitNum = 2;
constexpr int64_t kDefaultInt64Value = -10000;
const std::vector<int64_t> needCompressAttnMask = {ops::kSparseLeftUpCausal, ops::kSparseRightDownCausal,
                                                   ops::kSparseBand, ops::kSparseBlockLocal};

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

int64_t GetSplitNumByMapId(const Shape &dev_matrix, int64_t map_id) {
  if (map_id == MAP_NONE) {
    return NO_SPLIT_STRATEGY;
  }
  auto axis = dev_matrix.size() - 1 - LongToSize(map_id);
  if (axis >= dev_matrix.size()) {
    MS_LOG(EXCEPTION) << "The tensor map id (" << map_id
                      << ") is out of device matrix's range. device_matrix: " << dev_matrix;
  }
  return dev_matrix[axis];
}

int64_t GetSplitNumByTensorMap(const Shape &dev_matrix, const Shape &tensor_map) {
  auto split_num = std::accumulate(tensor_map.begin(), tensor_map.end(), 1, [&dev_matrix](int64_t a, int64_t map_id) {
    return a * GetSplitNumByMapId(dev_matrix, map_id);
  });
  return split_num;
}
}  // namespace

Status ComputeSparseInfoForFlashAttentionScore(const int64_t sparse_mode, const int64_t pre_tokens,
                                               const int64_t next_tokens, const int64_t seq_split_id,
                                               const int64_t seq_split_num, int64_t q_seq_len, int64_t kv_seq_len,
                                               int64_t *new_sparse_mode, int64_t *new_pre_tokens,
                                               int64_t *new_next_tokens) {
  *new_pre_tokens = (sparse_mode == ops::kSparseDefaultMask || sparse_mode == ops::kSparseBand)
                      ? pre_tokens
                      : q_seq_len * seq_split_num;
  *new_next_tokens = (sparse_mode == ops::kSparseDefaultMask || sparse_mode == ops::kSparseBand) ? next_tokens : 0;
  *new_sparse_mode = sparse_mode != ops::kSparseDefaultMask ? ops::kSparseBand : sparse_mode;
  switch (opAttrUpdateMap.at(sparse_mode)) {
    case kLeftUpToLeftUp:
      *new_pre_tokens = LongAdd(*new_pre_tokens, -seq_split_id * q_seq_len);
      *new_next_tokens = LongAdd(*new_next_tokens, seq_split_id * q_seq_len);
      break;
    case kLeftUpToRightDown:
      *new_pre_tokens = LongAdd(*new_pre_tokens, (kv_seq_len - (seq_split_id + 1) * q_seq_len));
      *new_next_tokens = LongAdd(*new_next_tokens, -(kv_seq_len - (seq_split_id + 1) * q_seq_len));
      break;
    case kRightDownToRightDown:
      *new_pre_tokens = LongAdd(*new_pre_tokens, (seq_split_num - seq_split_id - 1) * (q_seq_len));
      *new_next_tokens = LongAdd(*new_next_tokens, -(seq_split_num - seq_split_id - 1) * (q_seq_len));
      break;
    default:
      MS_LOG(ERROR) << "Invalid sparse mode " << sparse_mode << ", sparse mode should be one of [0, 2, 3, 4].";
      return FAILED;
  }
  return SUCCESS;
}

void FlashAttentionScoreInfo::UpdateDropoutGenMaskSliceShapeAndSeed(const CNodePtr &dropout_gen_mask_cnode) {
  if (!IsPrimitiveCNode(dropout_gen_mask_cnode, prim::kPrimDropoutGenMask)) {
    return;
  }

  // Update seed according rank_id for DropoutGenMask
  PrimitivePtr prim = GetCNodePrimitive(dropout_gen_mask_cnode);
  MS_EXCEPTION_IF_NULL(prim);
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
    MS_LOG_WITH_NODE(EXCEPTION, dropout_gen_mask_cnode)
      << "The size of dropout gen mask cnode's inputs must be " << DROPOUT_GEN_MASK_CNODE_INPUT_SIZE;
  }
  if (!IsValueNode<ValueTuple>(dropout_gen_mask_cnode->input(kIndex1))) {
    MS_LOG_WITH_NODE(EXCEPTION, dropout_gen_mask_cnode) << "The input[1] of dropout gen mask cnode is not ValueTuple.";
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

void FlashAttentionScoreInfo::ResetInputsShape() {
  size_t real_input_num = 0;
  for (size_t i = 0; i <= ops::kFlashAttentionScoreInputActualSeqKVlenIndex; ++i) {
    if (is_input_passed_[i]) {
      real_input_num += 1;
    }
  }
  inputs_shape_.resize(real_input_num);
}

size_t FlashAttentionScoreInfo::GetStrategyRealIndex(size_t index) {
  if (index >= is_input_passed_.size() || !is_input_passed_[index]) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode_) << name_ << ": GetStrategyRealIndex failed, index is " << index;
  }
  auto real_index = -1;
  for (size_t i = 0; i <= index; ++i) {
    if (is_input_passed_[i]) {
      ++real_index;
    }
  }
  return real_index;
}

RankList FlashAttentionScoreInfo::GetSPRankList() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  int64_t dev_matrix_s1_dim_real = dev_matrix_s1_dim_;
  if (inputs_tensor_map_.size() > ops::kFlashAttentionScoreInputQueryIndex) {
    switch (input_layout_) {
      case FASInputLayoutMode::BSH:
        dev_matrix_s1_dim_real = inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex][kIndex1];
        break;
      case FASInputLayoutMode::SBH:
        dev_matrix_s1_dim_real = inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex][kIndex0];
        break;
      case FASInputLayoutMode::BNSD:
        dev_matrix_s1_dim_real = inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex][kIndex2];
        break;
      case FASInputLayoutMode::BSND:
        dev_matrix_s1_dim_real = inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex][kIndex1];
        break;
      default:
        break;
    }
  }
  int64_t seq_dim = SizeToLong(dev_matrix_shape_.size()) - dev_matrix_s1_dim_real - 1;
  if (dev_matrix.GetDevicesAlongDim(seq_dim, &group_devices) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " get group devices along dim " << seq_dim << " failed.";
  }
  return group_devices;
}

int64_t FlashAttentionScoreInfo::GetActualSeqLengthSize() {
  if (is_input_passed_[ops::kFlashAttentionScoreInputActualSeqQlenIndex]) {
    auto actual_size = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputActualSeqQlenIndex));
    if (actual_size.size() != 1) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << "Shape of actual seq length should be 1.";
    }
    return actual_size[0] / batch_split_num_;
  }
  return 0;
}

Status FlashAttentionScoreInfo::InitAttnMaskStrategies() {
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    auto attn_mask_shape = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
    int64_t s1_split_num_attn_mask = is_attn_mask_compressed_ ? 1 : s1_split_num_;
    if (attn_mask_shape.size() == kSizeTwo) {
      // attn_mask_shape: (S1, 1)
      expect_strategies_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {s1_split_num_attn_mask, 1};
    } else if (attn_mask_shape.size() == kSizeFour) {
      // attn_mask_shape: (B, N1, S1, S2) or (B, 1, S1, S2)
      auto attn_mask_n1_split_num = attn_mask_have_n1_dim_ ? n1_split_num_ : 1;
      auto attn_batch_split_num = attn_mask_have_batch_dim_ ? batch_split_num_ : 1;
      expect_strategies_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {attn_batch_split_num, attn_mask_n1_split_num,
                                                                         s1_split_num_attn_mask, 1};
    }
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InitExpectedStrategies() {
  expect_strategies_ = Strategies(ops::kFlashAttentionScoreInputsNum);
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      expect_strategies_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_split_num_, s1_split_num_, n1_split_num_};
      expect_strategies_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_split_num_, s2_split_num_, n2_split_num_};
      expect_strategies_[ops::kFlashAttentionScoreInputValueIndex] = {batch_split_num_, s2_split_num_, n2_split_num_};
      break;
    case FASInputLayoutMode::SBH:
      expect_strategies_[ops::kFlashAttentionScoreInputQueryIndex] = {s1_split_num_, batch_split_num_, n1_split_num_};
      expect_strategies_[ops::kFlashAttentionScoreInputKeyIndex] = {s2_split_num_, batch_split_num_, n2_split_num_};
      expect_strategies_[ops::kFlashAttentionScoreInputValueIndex] = {s2_split_num_, batch_split_num_, n2_split_num_};
      break;
    case FASInputLayoutMode::BNSD:
      expect_strategies_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_split_num_, n1_split_num_, s1_split_num_,
                                                                      1};
      expect_strategies_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_split_num_, n2_split_num_, s2_split_num_, 1};
      expect_strategies_[ops::kFlashAttentionScoreInputValueIndex] = {batch_split_num_, n2_split_num_, s2_split_num_,
                                                                      1};
      break;
    case FASInputLayoutMode::BSND:
      expect_strategies_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_split_num_, s1_split_num_, n1_split_num_,
                                                                      1};
      expect_strategies_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_split_num_, s2_split_num_, n2_split_num_, 1};
      expect_strategies_[ops::kFlashAttentionScoreInputValueIndex] = {batch_split_num_, s2_split_num_, n2_split_num_,
                                                                      1};
      break;
    case FASInputLayoutMode::TND:
      expect_strategies_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_split_num_ * s1_split_num_, n1_split_num_,
                                                                      1};
      expect_strategies_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_split_num_, n2_split_num_, 1};
      expect_strategies_[ops::kFlashAttentionScoreInputValueIndex] = {batch_split_num_, n2_split_num_, 1};
      break;
    case FASInputLayoutMode::TH:
      expect_strategies_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_split_num_ * s1_split_num_, n1_split_num_};
      expect_strategies_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_split_num_, n2_split_num_};
      expect_strategies_[ops::kFlashAttentionScoreInputValueIndex] = {batch_split_num_, n2_split_num_};
      break;
    default:
      MS_LOG(ERROR) << name_ << "Not support layout: " << input_layout_;
      return FAILED;
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
  InitAttnMaskStrategies();

  // padding_mask is not support yet, skip it.

  if (is_input_passed_[ops::kFlashAttentionScoreInputPrefixIndex]) {
    expect_strategies_[ops::kFlashAttentionScoreInputPrefixIndex] = {batch_split_num_};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputActualSeqQlenIndex]) {
    expect_strategies_[ops::kFlashAttentionScoreInputActualSeqQlenIndex] = {batch_split_num_};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputActualSeqKVlenIndex]) {
    expect_strategies_[ops::kFlashAttentionScoreInputActualSeqKVlenIndex] = {batch_split_num_};
  }
  expect_strategies_.erase(std::remove(expect_strategies_.begin(), expect_strategies_.end(), Shape{}),
                           expect_strategies_.end());
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InitQKVTensorMap() {
  int64_t kv_head_num_map = kv_split_ ? dev_matrix_n1_dim_ : -1;
  auto dev_matrix_s2_dim = enable_ring_attention_ ? dev_matrix_s1_dim_ : -1;
  dev_matrix_s2_dim = enable_flash_sp_ ? dev_matrix_s1_dim_ : dev_matrix_s2_dim;
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex] = {dev_matrix_batch_dim_, dev_matrix_s1_dim_,
                                                                      dev_matrix_n1_dim_};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputKeyIndex] = {dev_matrix_batch_dim_, dev_matrix_s2_dim,
                                                                    kv_head_num_map};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputValueIndex] = {dev_matrix_batch_dim_, dev_matrix_s2_dim,
                                                                      kv_head_num_map};
      break;
    case FASInputLayoutMode::SBH:
      inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex] = {dev_matrix_s1_dim_, dev_matrix_batch_dim_,
                                                                      dev_matrix_n1_dim_};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputKeyIndex] = {dev_matrix_s2_dim, dev_matrix_batch_dim_,
                                                                    kv_head_num_map};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputValueIndex] = {dev_matrix_s2_dim, dev_matrix_batch_dim_,
                                                                      kv_head_num_map};
      break;
    case FASInputLayoutMode::BNSD:
      inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex] = {dev_matrix_batch_dim_, dev_matrix_n1_dim_,
                                                                      dev_matrix_s1_dim_, -1};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputKeyIndex] = {dev_matrix_batch_dim_, kv_head_num_map,
                                                                    dev_matrix_s2_dim, -1};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputValueIndex] = {dev_matrix_batch_dim_, kv_head_num_map,
                                                                      dev_matrix_s2_dim, -1};
      break;
    case FASInputLayoutMode::BSND:
      inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex] = {dev_matrix_batch_dim_, dev_matrix_s1_dim_,
                                                                      dev_matrix_n1_dim_, -1};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputKeyIndex] = {dev_matrix_batch_dim_, dev_matrix_s2_dim,
                                                                    kv_head_num_map, -1};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputValueIndex] = {dev_matrix_batch_dim_, dev_matrix_s2_dim,
                                                                      kv_head_num_map, -1};
      break;
    case FASInputLayoutMode::TND:
      inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex] = {dev_matrix_batch_dim_, dev_matrix_n1_dim_, -1};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputKeyIndex] = {dev_matrix_batch_dim_, kv_head_num_map, -1};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputValueIndex] = {dev_matrix_batch_dim_, kv_head_num_map, -1};
      break;
    case FASInputLayoutMode::TH:
      inputs_tensor_map_[ops::kFlashAttentionScoreInputQueryIndex] = {dev_matrix_batch_dim_, dev_matrix_n1_dim_};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputKeyIndex] = {dev_matrix_batch_dim_, kv_head_num_map};
      inputs_tensor_map_[ops::kFlashAttentionScoreInputValueIndex] = {dev_matrix_batch_dim_, kv_head_num_map};
      break;
    default:
      MS_LOG(ERROR) << name_ << "Not support layout: " << input_layout_;
      return FAILED;
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InitInputsTensorMap() {
  inputs_tensor_map_ = std::vector<Shape>(ops::kFlashAttentionScoreInputsNum);
  if (InitQKVTensorMap() != SUCCESS) {
    return FAILED;
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
  if (is_input_passed_[ops::kFlashAttentionScoreInputActualSeqQlenIndex]) {
    inputs_tensor_map_[ops::kFlashAttentionScoreInputActualSeqQlenIndex] = {dev_matrix_batch_dim_};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputActualSeqKVlenIndex]) {
    inputs_tensor_map_[ops::kFlashAttentionScoreInputActualSeqKVlenIndex] = {dev_matrix_batch_dim_};
  }
  inputs_tensor_map_.erase(std::remove(inputs_tensor_map_.begin(), inputs_tensor_map_.end(), Shape{}),
                           inputs_tensor_map_.end());
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InitAttnMaskSplittableInputs() {
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    int64_t s1_group = 2;
    auto attn_mask_shape = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
    int64_t attn_s1_group = is_attn_mask_compressed_ ? 0 : s1_group;
    int64_t attn_s2_group = enable_ring_attention_ ? attn_s1_group : 0;
    attn_s2_group = enable_flash_sp_ ? attn_s1_group : attn_s2_group;
    if (attn_mask_shape.size() == kSizeTwo) {
      // attn_mask_shape: (S1, S2)
      splittable_inputs_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {attn_s1_group, attn_s2_group};
    } else if (attn_mask_shape.size() == kSizeFour) {
      int64_t n1_group = 1;
      int64_t batch_group = 3;
      // attn_mask_shape: (B, N1, S1, S2) or (B, 1, S1, S2)
      auto attn_mask_n1_group = attn_mask_shape[kIndex1] == 1 ? 0 : n1_group;
      splittable_inputs_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {batch_group, attn_mask_n1_group, attn_s1_group,
                                                                         0};
    }
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InitSplittableInputs() {
  splittable_inputs_ = std::vector<Shape>(ops::kFlashAttentionScoreInputsNum);
  int64_t batch_group = 3;
  int64_t s1_group = 2;
  int64_t n1_group = 1;
  int64_t n2_group = kv_split_ ? n1_group : 0;
  int64_t s2_group = enable_ring_attention_ ? s1_group : 0;
  s2_group = enable_flash_sp_ ? s1_group : s2_group;
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      splittable_inputs_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_group, s1_group, n1_group};
      splittable_inputs_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_group, s2_group, n2_group};
      splittable_inputs_[ops::kFlashAttentionScoreInputValueIndex] = {batch_group, s2_group, n2_group};
      break;
    case FASInputLayoutMode::SBH:
      splittable_inputs_[ops::kFlashAttentionScoreInputQueryIndex] = {s1_group, batch_group, n1_group};
      splittable_inputs_[ops::kFlashAttentionScoreInputKeyIndex] = {s2_group, batch_group, n2_group};
      splittable_inputs_[ops::kFlashAttentionScoreInputValueIndex] = {s2_group, batch_group, n2_group};
      break;
    case FASInputLayoutMode::BNSD:
      splittable_inputs_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_group, n1_group, s1_group, 0};
      splittable_inputs_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_group, n2_group, s2_group, 0};
      splittable_inputs_[ops::kFlashAttentionScoreInputValueIndex] = {batch_group, n2_group, s2_group, 0};
      break;
    case FASInputLayoutMode::BSND:
      splittable_inputs_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_group, s1_group, n1_group, 0};
      splittable_inputs_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_group, s2_group, n2_group, 0};
      splittable_inputs_[ops::kFlashAttentionScoreInputValueIndex] = {batch_group, s2_group, n2_group, 0};
      break;
    case FASInputLayoutMode::TND:
      splittable_inputs_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_group, n1_group, 0};
      splittable_inputs_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_group, n2_group, 0};
      splittable_inputs_[ops::kFlashAttentionScoreInputValueIndex] = {batch_group, n2_group, 0};
      break;
    case FASInputLayoutMode::TH:
      splittable_inputs_[ops::kFlashAttentionScoreInputQueryIndex] = {batch_group, n1_group};
      splittable_inputs_[ops::kFlashAttentionScoreInputKeyIndex] = {batch_group, n2_group};
      splittable_inputs_[ops::kFlashAttentionScoreInputValueIndex] = {batch_group, n2_group};
      break;
    default:
      MS_LOG(ERROR) << name_ << "Not support layout: " << input_layout_;
      return FAILED;
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
  InitAttnMaskSplittableInputs();
  if (is_input_passed_[ops::kFlashAttentionScoreInputPrefixIndex]) {
    splittable_inputs_[ops::kFlashAttentionScoreInputPrefixIndex] = {batch_group};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputActualSeqQlenIndex]) {
    splittable_inputs_[ops::kFlashAttentionScoreInputActualSeqQlenIndex] = {batch_group};
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputActualSeqKVlenIndex]) {
    splittable_inputs_[ops::kFlashAttentionScoreInputActualSeqKVlenIndex] = {batch_group};
  }
  splittable_inputs_.erase(std::remove(splittable_inputs_.begin(), splittable_inputs_.end(), Shape{}),
                           splittable_inputs_.end());
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InitQKVHeadAndSeqDimFromInputLayout() {
  if (ops::layoutMap.find(input_layout_) == ops::layoutMap.end()) {
    MS_LOG(ERROR) << name_ << ": Not support layout " << input_layout_ << " in parallel currently.";
    return FAILED;
  }

  auto input_layout_str = ops::layoutMap.at(input_layout_);

  qkv_seq_dim_ = input_layout_str.find('S');
  if (qkv_seq_dim_ == std::string::npos) {
    qkv_seq_dim_ = input_layout_str.find('T');
  }

  qkv_batch_dim_ = input_layout_str.find('B');
  if (qkv_batch_dim_ == std::string::npos) {
    qkv_batch_dim_ = input_layout_str.find('T');
  }

  qkv_head_dim_ = input_layout_str.find('N');
  if (qkv_head_dim_ == std::string::npos) {
    qkv_head_dim_ = input_layout_str.find('H');
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InitAttrs() { return GetAttrs(); }

Status FlashAttentionScoreInfo::CheckInputLayout() {
  if (InferSplitNumAndDevMatrixShapeByLayout() != SUCCESS) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << ": Infer split num and dev matrix shape by layout failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Infer split num and dev matrix shape by layout failed.";
    }
    return FAILED;
  }

  auto query_shape = inputs_shape_[ops::kFlashAttentionScoreInputQueryIndex];
  auto key_shape = inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex];
  if (s1_split_num_ > 1 && is_flatten_batch_seq_ &&
      (sparse_mode_ != ops::kSparseRightDownCausal || query_shape[0] != key_shape[0])) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << ": When input_layout is TND, sparse_mode is 3, "
                   << "and the T-dimension of query and key are the same, "
                   << "the T-dimension of query can be sliced. query_shape: " << query_shape
                   << ", key_shape: " << key_shape << ", sparse_mode: " << sparse_mode_;
    } else {
      MS_LOG(ERROR) << name_ << ": When input_layout is TND, sparse_mode is 3, "
                    << "and the T-dimension of query and key are the same, "
                    << "the T-dimension of query can be sliced. query_shape: " << query_shape
                    << ", key_shape: " << key_shape << ", sparse_mode: " << sparse_mode_;
    }
    return FAILED;
  }

  // Check all device matrix should be the same
  if (ops::kFlashAttentionScoreInputQueryIndex >= inputs_tensor_info_.size()) {
    return FAILED;
  }
  auto query_tensor_info = inputs_tensor_info_[GetStrategyRealIndex(ops::kFlashAttentionScoreInputQueryIndex)];
  dev_matrix_shape_ = query_tensor_info.tensor_layout().device_arrangement_origin().array();
  return SUCCESS;
}

Status FlashAttentionScoreInfo::CheckOutputLayout() { return SUCCESS; }

Status FlashAttentionScoreInfo::InferOutputLayout() {
  auto query_layout = inputs_tensor_info_[ops::kFlashAttentionScoreInputQueryIndex].tensor_layout();
  const auto &query_device_matrix = query_layout.device_arrangement_origin().array();
  const auto &query_tensor_map = query_layout.tensor_map_before();

  // Construct layout for softmax_max and softmax_sum
  std::vector<Shape> softmax_max_sum_tensor_map;
  Shape softmax_max_sum_tensor_shape;
  if (is_flatten_batch_seq_) {
    softmax_max_sum_tensor_map.push_back(query_tensor_map[qkv_batch_dim_]);                              // T
    softmax_max_sum_tensor_shape.push_back(query_layout.tensor_shape_before().array()[qkv_batch_dim_]);  // T
    softmax_max_sum_tensor_map.push_back(query_tensor_map[qkv_head_dim_]);                               // N
    softmax_max_sum_tensor_shape.push_back(head_num_);                                                   // N
    softmax_max_sum_tensor_map.push_back({MAP_NONE});                                                    // 8
    softmax_max_sum_tensor_shape.push_back(8);                                                           // 8
  } else {
    softmax_max_sum_tensor_map.push_back(query_tensor_map[qkv_batch_dim_]);                              // B
    softmax_max_sum_tensor_shape.push_back(query_layout.tensor_shape_before().array()[qkv_batch_dim_]);  // B
    softmax_max_sum_tensor_map.push_back(query_tensor_map[qkv_head_dim_]);                               // N
    softmax_max_sum_tensor_shape.push_back(head_num_);                                                   // N
    softmax_max_sum_tensor_map.push_back(query_tensor_map[qkv_seq_dim_]);                                // S
    softmax_max_sum_tensor_shape.push_back(query_layout.tensor_shape_before().array()[qkv_seq_dim_]);    // S
    softmax_max_sum_tensor_map.push_back({MAP_NONE});                                                    // 8
    softmax_max_sum_tensor_shape.push_back(8);                                                           // 8
  }
  softmax_max_tensor_layout_.InitFromExtendVector(query_device_matrix, softmax_max_sum_tensor_map,
                                                  outputs_shape()[ops::kFlashAttentionScoreOutputSoftmaxMaxIndex]);
  softmax_sum_tensor_layout_.InitFromExtendVector(query_device_matrix, softmax_max_sum_tensor_map,
                                                  outputs_shape()[ops::kFlashAttentionScoreOutputSoftmaxSumIndex]);

  // Construct layout for softmax_out
  softmax_out_tensor_layout_.InitFromExtendVector(query_device_matrix, std::vector<Shape>{{MAP_NONE}},
                                                  outputs_shape()[ops::kFlashAttentionScoreOutputSoftmaxOutIndex]);
  attention_out_tensor_layout_.InitFromExtendVector(query_device_matrix, query_tensor_map,
                                                    outputs_shape()[ops::kFlashAttentionScoreOutputAttentionOutIndex]);
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferOutputTensorInfo() {
  auto status = InferOutputLayout();
  if (status != SUCCESS) {
    return status;
  }
  (void)outputs_tensor_info_.emplace_back(TensorInfo(softmax_max_tensor_layout_));
  (void)outputs_tensor_info_.emplace_back(TensorInfo(softmax_sum_tensor_layout_));
  (void)outputs_tensor_info_.emplace_back(TensorInfo(softmax_out_tensor_layout_));
  (void)outputs_tensor_info_.emplace_back(TensorInfo(attention_out_tensor_layout_));
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferAsLossDivisorByLayout() {
  if (outputs_tensor_info_.size() != ops::kFlashAttentionScoreOutputsNum) {
    MS_LOG(ERROR)
      << name_
      << ": The size of outputs tensor info must be equal to the size of FlashAttentionScore's output size, but got  "
      << outputs_tensor_info_.size() << " and " << ops::kFlashAttentionScoreOutputsNum;
    return FAILED;
  }

  auto attention_out_tensor_info = outputs_tensor_info_[ops::kFlashAttentionScoreOutputAttentionOutIndex];
  TensorMaps attention_out_tensor_map = attention_out_tensor_info.tensor_layout().tensor_map_before();
  if (attention_out_tensor_map.empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  auto out_dev_matrix_shape = attention_out_tensor_info.tensor_layout().device_arrangement_origin().array();
  if (out_dev_matrix_shape.empty()) {
    MS_LOG(INFO) << name_ << ": out_dev_matrix_shape is empty";
    out_dev_matrix_shape = dev_matrix_shape_;
  }
  Shape squashed_tensor_map;
  for (const auto &tensor_map : attention_out_tensor_map) {
    std::copy(tensor_map.begin(), tensor_map.end(), std::back_inserter(squashed_tensor_map));
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape, squashed_tensor_map);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape)
               << ", the output tensor map is " << ShapeToString(squashed_tensor_map) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferMirrorOpsByLayout() {
  mirror_ops_.clear();
  if (inputs_shape_.empty()) {
    MS_LOG(INFO) << name_ << ": The inputs size is empty";
    return SUCCESS;
  }

  bool group_is_empty = true;
  for (size_t i = 0; i < inputs_tensor_info_.size(); ++i) {
    if (inputs_tensor_info_[i] == TensorInfo()) {
      (void)mirror_ops_.emplace_back(OperatorVector());
      continue;
    }
    auto input_tensor_layout = inputs_tensor_info_[i].tensor_layout();
    auto repeated_rank_list = input_tensor_layout.InferRepeatedGroup();

    OperatorVector mirror_op;
    if (repeated_rank_list.size() == 1) {
      MS_LOG(INFO) << name_ << ": The mirror group is empty, the input index is " << i;
      mirror_ops_.push_back(mirror_op);
      continue;
    }

    Group mirror_group;
    if (g_device_manager->CreateGroup(repeated_rank_list, &mirror_group) != SUCCESS) {
      MS_LOG(ERROR) << name_
                    << ": Create communication group by tensor_map failed, the rank_list is: " << repeated_rank_list
                    << ", the full_name of node is: " << cnode_->fullname_with_scope();
      return FAILED;
    }
    group_is_empty = false;
    mirror_op = CreateMirrorOps(mirror_group.name(), mirror_group.GetDevNum());
    mirror_ops_.push_back(mirror_op);
  }

  if (group_is_empty) {
    mirror_ops_.clear();
    MS_LOG(INFO) << name_ << ": No need to insert mirror ops";
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::CheckInputInRingAttention() {
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel) {
    MS_LOG(WARNING) << "ring attention & flash sp only supports semi parallel mode";
    return SUCCESS;
  }
  if (input_layout_ != FASInputLayoutMode::BSH && input_layout_ != FASInputLayoutMode::BNSD) {
    MS_LOG(ERROR) << "Ring attention currently only supports BSH and BNSD layout";
    return FAILED;
  }
  if (sparse_mode_ != 0) {
    MS_LOG(ERROR) << "Ring attention currently only supports sparse mode 0";
    return FAILED;
  }
  if (!common::IsFloatEqual(keep_prob_, 1.0)) {
    MS_LOG(ERROR) << "Ring attention currently only supports keep prob 1.0";
  }
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex] &&
      (enable_flash_sp_ || enable_load_balance_ || enable_ra_cp_)) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "ring attention load balance don't supports user defined mask";
  }
  return SUCCESS;
}

void FlashAttentionScoreInfo::CheckDynamicShape() {
  if (is_dynamic_rank_) {
    MS_LOG(EXCEPTION) << name_ << ": Not support dynamic rank currently.";
  }
  if (is_dynamic_shape_ && enable_ra_cp_) {
    MS_LOG(EXCEPTION) << name_ << ": Not support dynamic shape currently when enable ring attention cp.";
  }
}

void FlashAttentionScoreInfo::InitFromInput() {
  head_num_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFlashAttentionScoreInputHeadNumIndex + 1);
  keep_prob_ = GetInputValueFromCNode<float>(cnode_, ops::kFlashAttentionScoreInputKeepProbIndex + 1);
  scale_value_ = GetInputValueFromCNode<float>(cnode_, ops::kFlashAttentionScoreInputScaleValueIndex + 1);
  pre_tokens_ = GetInputValueFromCNodeWithDefaultValue<int64_t>(
    cnode_, ops::kFlashAttentionScoreInputPreTokensIndex + 1, kDefaultInt64Value);
  next_tokens_ = GetInputValueFromCNodeWithDefaultValue<int64_t>(
    cnode_, ops::kFlashAttentionScoreInputNextTokensIndex + 1, kDefaultInt64Value);
  input_layout_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFlashAttentionScoreInputLayoutIndex + 1);
  sparse_mode_ = GetInputValueFromCNodeWithDefaultValue<int64_t>(
    cnode_, ops::kFlashAttentionScoreInputSparseModeIndex + 1, kDefaultInt64Value);
  is_flatten_batch_seq_ = (input_layout_ == FASInputLayoutMode::TND) || (input_layout_ == FASInputLayoutMode::TH);
}

Status FlashAttentionScoreInfo::GetAttrsForRA() {
  auto enable_ring_attention_iter = attrs_.find(ENABLE_RING_ATTENTION);
  if (enable_ring_attention_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(enable_ring_attention_iter->second);
    if (enable_ring_attention_iter->second->isa<BoolImm>()) {
      MS_EXCEPTION_IF_NULL(enable_ring_attention_iter->second->cast<BoolImmPtr>());
      enable_ring_attention_ = enable_ring_attention_iter->second->cast<BoolImmPtr>()->value();
      enable_load_balance_ = false;
    } else {
      MS_LOG(ERROR) << name_ << ": The type of primitive attribute 'enable_ring_attention' should be bool.";
      return FAILED;
    }
  }
  auto enable_ra_send_recv_iter = attrs_.find(ENABLE_RA_SEND_RECV);
  if (enable_ra_send_recv_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(enable_ra_send_recv_iter->second);
    if (enable_ra_send_recv_iter->second->isa<BoolImm>()) {
      MS_EXCEPTION_IF_NULL(enable_ra_send_recv_iter->second->cast<BoolImmPtr>());
      enable_ra_send_recv_ = enable_ra_send_recv_iter->second->cast<BoolImmPtr>()->value();
    }
  }
  auto enable_ra_cp_iter = attrs_.find(ENABLE_RA_CONTEXT_PARALLEL);
  if (enable_ra_cp_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(enable_ra_cp_iter->second);
    MS_EXCEPTION_IF_NULL(enable_ra_cp_iter->second->cast<BoolImmPtr>());
    if (enable_ra_cp_iter->second->isa<BoolImm>()) {
      enable_ra_cp_ = enable_ra_cp_iter->second->cast<BoolImmPtr>()->value();
    }
  }
  if (!enable_ring_attention_ && (enable_ra_send_recv_ || enable_ra_cp_)) {
    MS_LOG(ERROR) << "Enable ring attention should be true.";
    return FAILED;
  }
  auto enable_flash_sp_iter = attrs_.find("enable_flash_sp");
  if (enable_flash_sp_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(enable_flash_sp_iter->second);
    if (enable_flash_sp_iter->second->isa<BoolImm>()) {
      MS_EXCEPTION_IF_NULL(enable_flash_sp_iter->second->cast<BoolImmPtr>());
      enable_flash_sp_ = enable_flash_sp_iter->second->cast<BoolImmPtr>()->value();
      enable_load_balance_ = false;
    } else {
      MS_LOG(ERROR) << "enable_flash_sp should be bool";
      return FAILED;
    }
  }
  if (enable_ring_attention_ || enable_flash_sp_) {
    if (CheckInputInRingAttention() != Status::SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::GetAttrs() {
  InitIsInputPassed();
  ResetInputsShape();
  InitFromInput();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  enable_load_balance_ = ms_context->get_param<bool>(MS_CTX_ENABLE_FLASH_ATTENTION_LOAD_BALANCE);

  if (is_flatten_batch_seq_ && enable_load_balance_) {
    MS_LOG(WARNING) << name_ << ": Load balancing is not supported in the layout 'TND' and will be disabled.";
    enable_load_balance_ = false;
  }

  if (GetAttrsForRA() != Status::SUCCESS) {
    return FAILED;
  }

  is_attn_mask_compressed_ =
    std::find(needCompressAttnMask.begin(), needCompressAttnMask.end(), sparse_mode_) != needCompressAttnMask.end();
  need_update_op_attrs_mode_ = sparse_mode_ != ops::kSparseAllMask;
  if (InitQKVHeadAndSeqDimFromInputLayout() != Status::SUCCESS) {
    return FAILED;
  }

  CheckDynamicShape();

  kv_split_ = inputs_shape_[ops::kFlashAttentionScoreInputQueryIndex][qkv_head_dim_] !=
              inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex][qkv_head_dim_] * head_num_;

  if (is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    auto real_shift_s1_dim =
      inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputRealShiftIndex)).at(kIndex3);
    real_shift_have_s1_dim_ = real_shift_s1_dim != 1;
    auto real_shift_batch_dim =
      inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputRealShiftIndex)).at(kIndex0);
    real_shift_have_batch_dim_ = real_shift_batch_dim != 1;
  }

  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    auto attn_mask_shape = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
    if (attn_mask_shape.size() == kSizeFour) {
      attn_mask_have_batch_dim_ = attn_mask_shape.at(kIndex0) != 1;
      attn_mask_have_n1_dim_ = attn_mask_shape.at(kIndex1) != 1;
    }
  }
  q_seq_len_ = inputs_shape_[ops::kFlashAttentionScoreInputQueryIndex][qkv_seq_dim_];
  kv_seq_len_ = inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex][qkv_seq_dim_];
  dynamic_seq_flag_ = (q_seq_len_ == -1) || (kv_seq_len_ == -1);
  return SUCCESS;
}

Status CheckKVStrategy(const std::string &name, const Shape &key_strategy, const Shape &value_strategy) {
  if (key_strategy != value_strategy) {
    MS_LOG(ERROR) << name << ": The in_strategy both of 'key'( " << key_strategy << ") and 'value'" << value_strategy
                  << ") must be same.";
    return FAILED;
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
  if (head_num_ % query_strategy[qkv_head_dim_] != 0) {
    MS_LOG(ERROR) << name_ << ": head_num % query_strategy[" << qkv_head_dim_ << "] must be 0, but got " << head_num_
                  << "(head_num) and " << query_strategy[qkv_head_dim_] << "(query_strategy[" << qkv_head_dim_ << "])";
    return FAILED;
  }
  if (!kv_split_ && key_strategy[qkv_head_dim_] != 1) {
    MS_LOG(ERROR) << name_ << ": Under the MQA，the hidden-dim of input 'key' cannot be split.";
    return FAILED;
  }

  if (is_flatten_batch_seq_) {
    if (query_strategy[qkv_seq_dim_] != key_strategy[qkv_seq_dim_]) {
      MS_LOG(ERROR)
        << name_ << ": The split num of seq-dim between query and key must be the same when layout is 'TND'. But got "
        << query_strategy[qkv_seq_dim_] << " and " << key_strategy[qkv_seq_dim_];
      return FAILED;
    }
  } else {
    auto s2_split_num = key_strategy[qkv_seq_dim_];
    if (s2_split_num != 1 && !enable_ring_attention_ && !enable_flash_sp_) {
      MS_LOG(ERROR) << name_ << ": The S-Dimension of input 'key' cannot be split, but got the strategy of key is "
                    << key_strategy;
      return FAILED;
    }
  }

  if (is_flatten_batch_seq_) {
    batch_split_num_ = key_strategy[qkv_batch_dim_];
    s1_split_num_ = query_strategy[qkv_batch_dim_] / batch_split_num_;
  } else {
    batch_split_num_ = query_strategy[qkv_batch_dim_];
    s1_split_num_ = query_strategy[qkv_seq_dim_];
  }
  n1_split_num_ = query_strategy[qkv_head_dim_];

  s2_split_num_ = enable_ring_attention_ ? s1_split_num_ : 1;
  s2_split_num_ = enable_flash_sp_ ? s1_split_num_ : s2_split_num_;

  n2_split_num_ = key_strategy[qkv_head_dim_];

  if (kv_split_ && n1_split_num_ != n2_split_num_) {
    MS_LOG(ERROR) << name_ << ": The split num of N1-dim and N2-dim must be equal if N2 > 1, but got " << n1_split_num_
                  << " and " << n2_split_num_;
    return FAILED;
  }

  if (s1_split_num_ > 1 && is_flatten_batch_seq_) {
    MS_LOG(ERROR)
      << name_
      << ": Currently, input_layout is TND, and the seq dimension of query is segmented. Please use Layout to "
         "set the strategy.";
    return FAILED;
  }
  if (CheckStrategyExpected(strategy) != SUCCESS || CheckKVStrategy(name_, key_strategy, value_strategy) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::CheckStrategyExpected(const StrategyPtr &strategy) {
  auto strategies = strategy->GetInputDim();
  if (InitExpectedStrategies() != SUCCESS) {
    return FAILED;
  }

  if (strategies != expect_strategies_) {
    MS_LOG(ERROR) << name_ << ": The input strategy must be " << expect_strategies_ << ", but got " << strategies;
    if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
      Dimensions attn_stra = strategies.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
      if (attn_stra[attn_stra.size() - kIndex1] != kIndex1) {
        MS_LOG(ERROR) << name_ << ": The last dim of the attn_mask should not be split.";
      }
    }
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
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
    case FASInputLayoutMode::BSND:
    case FASInputLayoutMode::TND:
    case FASInputLayoutMode::TH:
      dev_matrix_shape_ = {batch_split_num_, s1_split_num_, n1_split_num_};
      dev_matrix_batch_dim_ = kIndex2;
      dev_matrix_s1_dim_ = kIndex1;
      dev_matrix_n1_dim_ = kIndex0;
      break;
    case FASInputLayoutMode::SBH:
      dev_matrix_shape_ = {s1_split_num_, batch_split_num_, n1_split_num_};
      dev_matrix_s1_dim_ = kIndex2;
      dev_matrix_batch_dim_ = kIndex1;
      dev_matrix_n1_dim_ = kIndex0;
      break;
    case FASInputLayoutMode::BNSD:
      dev_matrix_shape_ = {batch_split_num_, n1_split_num_, s1_split_num_};
      dev_matrix_batch_dim_ = kIndex2;
      dev_matrix_n1_dim_ = kIndex1;
      dev_matrix_s1_dim_ = kIndex0;
      break;
    default:
      MS_LOG(ERROR) << name_ << ": Not support layout: " << input_layout_;
      return FAILED;
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferSplitNumAndDevMatrixShapeByLayout() {
  dev_matrix_shape_ =
    inputs_tensor_info_[ops::kFlashAttentionScoreInputQueryIndex].tensor_layout().device_arrangement_origin().array();
  auto query_layout = inputs_tensor_info_[ops::kFlashAttentionScoreInputQueryIndex].tensor_layout();
  auto key_layout = inputs_tensor_info_[ops::kFlashAttentionScoreInputKeyIndex].tensor_layout();
  auto query_tensor_map = query_layout.tensor_map_before();
  auto query_batch_map = query_tensor_map.at(qkv_batch_dim_);
  auto query_seq_map = query_tensor_map.at(qkv_seq_dim_);
  auto query_head_map = query_tensor_map.at(qkv_head_dim_);
  auto key_seq_map = key_layout.tensor_map_before().at(qkv_seq_dim_);

  auto dev_matrix_shape = dev_matrix_shape_;
  if (is_flatten_batch_seq_) {
    if (query_batch_map.size() == kSizeOne) {
      dev_matrix_batch_dim_ = query_batch_map[0];
      dev_matrix_s1_dim_ = MAP_NONE;
    } else if (query_batch_map.size() == kSizeTwo) {
      dev_matrix_batch_dim_ = query_batch_map[0];
      dev_matrix_s1_dim_ = query_batch_map[1];
    } else {
      MS_LOG(ERROR) << name_
                    << ": The seq-dimension of query can only be mapped upto 2 device matrix dimension, but got "
                    << query_batch_map;
      return FAILED;
    }

    batch_split_num_ = GetSplitNumByMapId(dev_matrix_shape, dev_matrix_batch_dim_);
    s1_split_num_ = GetSplitNumByMapId(dev_matrix_shape, dev_matrix_s1_dim_);
    if (s1_split_num_ > 1 && GetSplitNumByTensorMap(dev_matrix_shape, query_seq_map) !=
                               GetSplitNumByTensorMap(dev_matrix_shape, key_seq_map) * s1_split_num_) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_)
        << name_ << ": Cannot split the seq-dimension of key. query_seq_slice: "
        << GetSplitNumByTensorMap(dev_matrix_shape, query_seq_map)
        << ", key_seq_slice: " << GetSplitNumByTensorMap(dev_matrix_shape, key_seq_map)
        << ", s1_split_num: " << s1_split_num_ << " when layout is TND";
    }
  } else {
    if (query_batch_map.size() != 1 || query_seq_map.size() != 1) {
      MS_LOG(ERROR) << name_
                    << ": Each dimension of query can only be mapped to one device matrix dimension, but got the "
                       "tensor info of query is "
                    << query_layout.ToString();
      return FAILED;
    }
    dev_matrix_batch_dim_ = query_batch_map[0];
    dev_matrix_s1_dim_ = query_seq_map[0];
    batch_split_num_ = GetSplitNumByMapId(dev_matrix_shape, dev_matrix_batch_dim_);
    s1_split_num_ = GetSplitNumByMapId(dev_matrix_shape, dev_matrix_s1_dim_);
  }

  n1_split_num_ = 1;
  for (auto map_id : query_head_map) {
    n1_split_num_ *= GetSplitNumByMapId(dev_matrix_shape, map_id);
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferTensorMap() {
  if (InitInputsTensorMap() != SUCCESS) {
    return FAILED;
  }
  if (is_flatten_batch_seq_) {
    outputs_tensor_map_.push_back({inputs_tensor_map_[0]});  // softmax_max
    outputs_tensor_map_.push_back({inputs_tensor_map_[0]});  // softmax_sum
  } else {
    outputs_tensor_map_.push_back({dev_matrix_batch_dim_, dev_matrix_n1_dim_, dev_matrix_s1_dim_, -1});  // softmax_max
    outputs_tensor_map_.push_back({dev_matrix_batch_dim_, dev_matrix_n1_dim_, dev_matrix_s1_dim_, -1});  // softmax_sum
  }
  outputs_tensor_map_.push_back({-1});                   // softmax_out
  outputs_tensor_map_.push_back(inputs_tensor_map_[0]);  // attention_out
  return SUCCESS;
}

std::vector<int64_t> FlashAttentionScoreInfo::GetSplitIdAndRank() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  int64_t seq_dim = SizeToLong(dev_matrix_shape_.size()) - dev_matrix_s1_dim_ - 1;
  if (repeated_calc_num_ > 1 && repeated_num_in_dev_matrix_right()) {
    seq_dim -= 1;
  }
  if (dev_matrix.GetDevicesAlongDim(seq_dim, &group_devices) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " get group devices along dim " << seq_dim << " failed.";
  }
  auto iter = std::find(group_devices.begin(), group_devices.end(), rank);
  if (iter == group_devices.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "FlashAttentionScore S1 sequence parallel get split id failed. "
                                        << "rank " << rank << " not in group " << group_devices;
  }
  int64_t split_id = iter - group_devices.begin();
  int64_t target_split_id = s1_split_num_ - split_id - 1;
  int64_t target_rank_id = group_devices[target_split_id];
  return std::vector<int64_t>({rank, target_rank_id, split_id, target_split_id});
}

std::tuple<int64_t, int64_t> FlashAttentionScoreInfo::GetAttentionMaskAttrs(const int64_t split_id,
                                                                            const int64_t split_num) {
  int64_t q_len_each_split = q_seq_len_ / split_num;
  int64_t new_pre_tokens =
    (sparse_mode_ == ops::kSparseDefaultMask || sparse_mode_ == ops::kSparseBand) ? pre_tokens_ : q_seq_len_;
  int64_t new_next_tokens =
    (sparse_mode_ == ops::kSparseDefaultMask || sparse_mode_ == ops::kSparseBand) ? next_tokens_ : 0;
  switch (opAttrUpdateMap.at(sparse_mode_)) {
    case kLeftUpToLeftUp:
      new_pre_tokens = LongAdd(new_pre_tokens, -split_id * q_len_each_split);
      new_next_tokens = LongAdd(new_next_tokens, split_id * q_len_each_split);
      break;
    case kLeftUpToRightDown:
      new_pre_tokens = LongAdd(new_pre_tokens, (kv_seq_len_ - (split_id + 1) * q_len_each_split));
      new_next_tokens = LongAdd(new_next_tokens, -(kv_seq_len_ - (split_id + 1) * q_len_each_split));
      break;
    case kRightDownToRightDown:
      new_pre_tokens = LongAdd(new_pre_tokens, (split_num - split_id - 1) * (q_seq_len_ / split_num));
      new_next_tokens = LongAdd(new_next_tokens, -(split_num - split_id - 1) * (q_seq_len_ / split_num));
      break;
    default:
      MS_LOG_WITH_NODE(EXCEPTION, cnode_)
        << "Invalid sparse mode " << sparse_mode_ << ", sparse mode should be one of [0, 2, 3, 4].";
  }
  return std::make_tuple(new_pre_tokens, new_next_tokens);
}

std::tuple<AnfNodePtr, AnfNodePtr> FlashAttentionScoreInfo::GetAttentionMaskAttrsByDynamicSeqDim(
  const AnfNodePtr &query_shape, const AnfNodePtr &key_shape, const int64_t split_id, const int64_t split_num,
  GenerateGraph *gen_g) {
  auto query_seq_slice_length = gen_g->PushBack(
    {NewValueNode(prim::kPrimTupleGetItem), query_shape, NewValueNode(MakeValue(SizeToLong(qkv_seq_dim_)))});
  auto key_seq_length = gen_g->PushBack(
    {NewValueNode(prim::kPrimTupleGetItem), key_shape, NewValueNode(MakeValue(SizeToLong(qkv_seq_dim_)))});

  AnfNodePtr new_pre_tokens;
  AnfNodePtr new_next_tokens;
  if (sparse_mode_ == ops::kSparseDefaultMask || sparse_mode_ == ops::kSparseBand) {
    new_pre_tokens = NewValueNode(MakeValue(pre_tokens_));
    new_next_tokens = NewValueNode(MakeValue(next_tokens_));
  } else {
    new_pre_tokens = key_seq_length;
    new_next_tokens = NewValueNode(MakeValue<int64_t>(0));
  }
  AnfNodePtr new_pre_tokens_offset;
  AnfNodePtr new_next_tokens_offset;
  switch (opAttrUpdateMap.at(sparse_mode_)) {
    case kLeftUpToLeftUp:
      // new_pre_tokens_offset = -split_id * q_len_each_split
      // new_next_tokens_offset = split_id * q_len_each_split
      new_pre_tokens_offset = gen_g->PushBack(
        {NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(-split_id)), query_seq_slice_length});
      new_next_tokens_offset = gen_g->PushBack(
        {NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(split_id)), query_seq_slice_length});
      break;
    case kLeftUpToRightDown:
      // new_pre_tokens_offset = kv_seq_length - (split_id + 1) * q_len_each_split
      // new_next_tokens_offset = -kv_seq_length + (split_id + 1) * q_len_each_split
      new_pre_tokens_offset = gen_g->PushBack(
        {NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(split_id + 1)), query_seq_slice_length});
      new_pre_tokens_offset =
        gen_g->PushBack({NewValueNode(prim::kPrimScalarSub), key_seq_length, new_pre_tokens_offset});
      new_next_tokens_offset = gen_g->PushBack(
        {NewValueNode(prim::kPrimScalarMul), new_pre_tokens_offset, NewValueNode(MakeValue<int64_t>(-1))});
      break;
    case kRightDownToRightDown:
      // new_pre_tokens_offset = (split_num - split_id - 1) * (q_seq_length / split_num)
      // new_next_tokens_offset = -(split_num - split_id - 1) * (q_seq_length / split_num))
      new_pre_tokens_offset =
        gen_g->PushBack({NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(split_num - split_id - 1)),
                         query_seq_slice_length});
      new_next_tokens_offset =
        gen_g->PushBack({NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(split_id + 1 - split_num)),
                         query_seq_slice_length});
      break;
    default:
      MS_LOG(EXCEPTION) << "Invalid sparse mode " << sparse_mode_ << ", sparse mode should be one of [0, 2, 3, 4].";
  }

  new_pre_tokens = gen_g->PushBack({NewValueNode(prim::kPrimScalarAdd), new_pre_tokens, new_pre_tokens_offset});
  new_next_tokens = gen_g->PushBack({NewValueNode(prim::kPrimScalarAdd), new_next_tokens, new_next_tokens_offset});
  return std::make_tuple(new_pre_tokens, new_next_tokens);
}

Status FlashAttentionScoreInfo::ReplaceActualSeqLenForSplitSeqInTnd(const CNodePtr &cnode) {
  if (!is_input_passed_[ops::kFlashAttentionScoreInputActualSeqQlenIndex] ||
      !is_input_passed_[ops::kFlashAttentionScoreInputActualSeqKVlenIndex]) {
    MS_LOG(ERROR) << name_ << ": The input 'actual_seq_qlen' and 'actual_seq_kvlen' cannot be None under 'TND'.";
    return FAILED;
  }
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // lambda to create cnode
  auto OpNode = [&](const PrimitivePtr &prim, const std::vector<AnfNodePtr> &inputs) -> AnfNodePtr {
    AnfNodePtrList full{NewValueNode(prim)};
    full.reserve(1 + inputs.size());
    full.insert(full.end(), inputs.begin(), inputs.end());
    return func_graph->NewCNode(full);
  };

  auto actual_seq_qlen_input_index = ops::kFlashAttentionScoreInputActualSeqQlenIndex + 1;
  auto actual_seq_kvlen_input_index = ops::kFlashAttentionScoreInputActualSeqKVlenIndex + 1;
  auto actual_seq_qlen_node = cnode->input(actual_seq_qlen_input_index);
  auto actual_seq_kvlen_node = cnode->input(actual_seq_kvlen_input_index);
  auto split_id = GetSplitIdAndRank().at(kIndex2);

  AnfNodePtr slice_tq_node;
  AnfNodePtr slice_tk_node;
  AnfNodePtr offset_node;
  AnfNodePtr scatter_indices_node;
  AnfNodePtr scatter_updates_node;

  // Compute slice_tq, slice_tk and offset
  if (dynamic_seq_flag_) {
    auto query_shape = OpNode(prim::kPrimShape, {cnode->input(ops::kFlashAttentionScoreInputQueryIndex + 1)});
    auto key_shape = OpNode(prim::kPrimShape, {cnode->input(ops::kFlashAttentionScoreInputKeyIndex + 1)});
    auto slice_tq_scalar_node = OpNode(prim::kPrimTupleGetItem, {query_shape, NewValueNode(MakeValue(qkv_batch_dim_))});
    auto slice_tk_scalar_node = OpNode(prim::kPrimTupleGetItem, {key_shape, NewValueNode(MakeValue(qkv_batch_dim_))});
    slice_tq_node = OpNode(prim::kPrimScalarToTensor, {slice_tq_scalar_node});
    slice_tk_node = OpNode(prim::kPrimScalarToTensor, {slice_tk_scalar_node});
    offset_node = OpNode(prim::kPrimMul, {slice_tq_node, CreateInt32Tensor(split_id, true)});

    // Construct update value as kv_len
    scatter_updates_node = OpNode(prim::kPrimReshape, {slice_tk_node, CreateTuple({1})});
  } else {
    int64_t tq = inputs_shape_[GetStrategyRealIndex(ops::kFlashAttentionScoreInputQueryIndex)][qkv_batch_dim_];
    int64_t tk = inputs_shape_[GetStrategyRealIndex(ops::kFlashAttentionScoreInputKeyIndex)][qkv_batch_dim_];
    auto slice_tq = tq / s1_split_num_ / batch_split_num_;
    auto slice_tk = tk / batch_split_num_;
    slice_tq_node = CreateInt32Tensor(slice_tq, true);
    offset_node = CreateInt32Tensor(slice_tq * split_id, true);

    // Construct update value as kv_len
    scatter_updates_node = ValuePtrToAnfNodePtr(MakeValue(tensor::from_vector(std::vector<int64_t>{slice_tk}, kInt64)));
    scatter_updates_node->set_abstract(std::make_shared<abstract::AbstractTensor>(kInt64, Shape{1}));
  }

  // Construct indices as new_actual_kv_len.shape[0] - 1
  auto dim = inputs_shape_[GetStrategyRealIndex(ops::kFlashAttentionScoreInputActualSeqKVlenIndex)].at(kIndex0);
  if (dim == -1) {
    // actual_seq_kv_len is a dynamic shape input, scatter_indices_node = actual_seq_kvlen.shape[0] - 1
    auto actual_kv_shape =
      OpNode(prim::kPrimShape, {cnode->input(ops::kFlashAttentionScoreInputActualSeqKVlenIndex + 1)});
    auto actual_kv_dim_node =
      OpNode(prim::kPrimScalarToTensor,
             {OpNode(prim::kPrimTupleGetItem, {actual_kv_shape, NewValueNode(MakeValue(kSizeZero))})});
    scatter_indices_node = OpNode(prim::kPrimSub, {actual_kv_dim_node, NewValueNode(MakeValue(kSizeOne))});
  } else {
    int64_t scatter_indices[] = {dim / batch_split_num_ - 1};
    ShapeVector scatter_indices_node_shape = {1, 1};
    scatter_indices_node = ValuePtrToAnfNodePtr(
      MakeValue(tensor::from_buffer(kNumberTypeInt64, ShapeVector{1, 1}, scatter_indices, sizeof(int64_t))));
    scatter_indices_node->set_abstract(std::make_shared<abstract::AbstractTensor>(kInt64, ShapeVector{1, 1}));
  }

  // new_actual_seq_qlen = clip(actual_seq_qlen - offset, 0, slice_tq)
  auto qlen_offset_sub_cnode = OpNode(prim::kPrimSub, {actual_seq_qlen_node, offset_node});
  auto new_actual_seq_qlen_cnode =
    OpNode(prim::kPrimClipByValue, {qlen_offset_sub_cnode, CreateInt32Tensor(0, true), slice_tq_node});
  manager->SetEdge(cnode, actual_seq_qlen_input_index, new_actual_seq_qlen_cnode);

  // new_actual_seq_kvlen = actual_seq_kvlen - (ReLU(actual_seq_qlen - offset) - new_actual_seq_qlen)
  auto relu_cnode = OpNode(prim::kPrimReLU, {qlen_offset_sub_cnode});
  auto kvlen_offset_sub_cnode = OpNode(prim::kPrimSub, {relu_cnode, new_actual_seq_qlen_cnode});
  auto new_actual_seq_kvlen_cnode = OpNode(prim::kPrimSub, {actual_seq_kvlen_node, kvlen_offset_sub_cnode});

  // new_actual_seq_kvlen[-1] = slice_tk if actual_seq_kvlen[-1] == slice_tk else actual_seq_kvlen[-1]
  auto GetActualKvLenLastElementNode = [&]() -> AnfNodePtr {
    int64_t kvlen_dim = inputs_shape_[GetStrategyRealIndex(ops::kFlashAttentionScoreInputActualSeqKVlenIndex)].at(0);
    if (kvlen_dim == -1) {
      auto shape = OpNode(prim::kPrimShape, {actual_seq_kvlen_node});
      auto len = OpNode(prim::kPrimReshape,
                        {OpNode(prim::kPrimTupleGetItem, {shape, NewValueNode<int64_t>(0)}), CreateTuple({1})});
      auto last_idx = OpNode(prim::kPrimSub, {OpNode(prim::kPrimScalarToTensor, {len}), NewValueNode<int64_t>(1)});
      return OpNode(prim::kPrimGather, {actual_seq_kvlen_node, last_idx, NewValueNode(MakeValue(int64_t{0}))});
    } else {
      int64_t last_idx_list[] = {kvlen_dim / batch_split_num_ - 1};
      auto last_idx =
        ValuePtrToAnfNodePtr(MakeValue(tensor::from_buffer(kNumberTypeInt64, {1}, last_idx_list, sizeof(int64_t))));
      last_idx->set_abstract(std::make_shared<abstract::AbstractTensor>(kInt64, ShapeVector{1}));
      return OpNode(prim::kPrimGather,
                    {actual_seq_kvlen_node, last_idx, NewValueNode<int64_t>(0), NewValueNode<int64_t>(0)});
    }
  };
  auto actual_seq_kvlen_last_value_node = GetActualKvLenLastElementNode();
  auto equal_node = OpNode(prim::kPrimEqual, {actual_seq_kvlen_last_value_node, scatter_updates_node});
  scatter_updates_node =
    OpNode(prim::kPrimSelect, {equal_node, scatter_updates_node, actual_seq_kvlen_last_value_node});
  auto scatter_update =
    OpNode(prim::kPrimScatterNdUpdate, {new_actual_seq_kvlen_cnode, scatter_indices_node, scatter_updates_node});
  manager->SetEdge(cnode, actual_seq_kvlen_input_index, scatter_update);
  return SUCCESS;
}

std::tuple<AnfNodePtr, AnfNodePtr> FlashAttentionScoreInfo::GetAttentionMaskAttrsByDynamicSeqDim(
  const CNodePtr &fa_cnode, const int64_t split_id, const int64_t split_num) {
  MS_EXCEPTION_IF_NULL(fa_cnode);
  auto func_graph = fa_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto query_shape = func_graph->NewCNode(
    {NewValueNode(prim::kPrimShape), fa_cnode->input(ops::kFlashAttentionScoreInputQueryIndex + 1)});
  auto key_shape =
    func_graph->NewCNode({NewValueNode(prim::kPrimShape), fa_cnode->input(ops::kFlashAttentionScoreInputKeyIndex + 1)});

  auto query_seq_slice_length = func_graph->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), query_shape, NewValueNode(MakeValue(SizeToLong(qkv_seq_dim_)))});
  auto key_seq_length = func_graph->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), key_shape, NewValueNode(MakeValue(SizeToLong(qkv_seq_dim_)))});

  AnfNodePtr new_pre_tokens;
  AnfNodePtr new_next_tokens;
  if (sparse_mode_ == ops::kSparseDefaultMask || sparse_mode_ == ops::kSparseBand) {
    new_pre_tokens = NewValueNode(MakeValue(pre_tokens_));
    new_next_tokens = NewValueNode(MakeValue(next_tokens_));
  } else {
    new_pre_tokens = key_seq_length;
    new_next_tokens = NewValueNode(MakeValue<int64_t>(0));
  }
  AnfNodePtr new_pre_tokens_offset;
  AnfNodePtr new_next_tokens_offset;
  switch (opAttrUpdateMap.at(sparse_mode_)) {
    case kLeftUpToLeftUp:
      // new_pre_tokens_offset = -split_id * q_len_each_split
      // new_next_tokens_offset = split_id * q_len_each_split
      new_pre_tokens_offset = func_graph->NewCNode(
        {NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(-split_id)), query_seq_slice_length});
      new_next_tokens_offset = func_graph->NewCNode(
        {NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(split_id)), query_seq_slice_length});
      break;
    case kLeftUpToRightDown:
      // new_pre_tokens_offset = kv_seq_length - (split_id + 1) * q_len_each_split
      // new_next_tokens_offset = -kv_seq_length + (split_id + 1) * q_len_each_split
      new_pre_tokens_offset = func_graph->NewCNode(
        {NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(split_id + 1)), query_seq_slice_length});
      new_pre_tokens_offset =
        func_graph->NewCNode({NewValueNode(prim::kPrimScalarSub), key_seq_length, new_pre_tokens_offset});
      new_next_tokens_offset = func_graph->NewCNode(
        {NewValueNode(prim::kPrimScalarMul), new_pre_tokens_offset, NewValueNode(MakeValue<int64_t>(-1))});
      break;
    case kRightDownToRightDown:
      // new_pre_tokens_offset = (split_num - split_id - 1) * (q_seq_length / split_num)
      // new_next_tokens_offset = -(split_num - split_id - 1) * (q_seq_length / split_num))
      new_pre_tokens_offset =
        func_graph->NewCNode({NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(split_num - split_id - 1)),
                              query_seq_slice_length});
      new_next_tokens_offset =
        func_graph->NewCNode({NewValueNode(prim::kPrimScalarMul), NewValueNode(MakeValue(split_id + 1 - split_num)),
                              query_seq_slice_length});
      break;
    default:
      MS_LOG(EXCEPTION) << "Invalid sparse mode " << sparse_mode_ << ", sparse mode should be one of [0, 2, 3, 4].";
  }

  new_pre_tokens = func_graph->NewCNode({NewValueNode(prim::kPrimScalarAdd), new_pre_tokens, new_pre_tokens_offset});
  new_next_tokens = func_graph->NewCNode({NewValueNode(prim::kPrimScalarAdd), new_next_tokens, new_next_tokens_offset});
  return std::make_tuple(new_pre_tokens, new_next_tokens);
}

void FlashAttentionScoreInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    SetValueInputToCNode<int64_t>(cnode, ops::kFlashAttentionScoreInputHeadNumIndex + 1, head_num_ / n1_split_num_);
    if (s1_split_num_ > 1 && !enable_load_balance_ && need_update_op_attrs_mode_) {
      if (is_flatten_batch_seq_) {
        if (ReplaceActualSeqLenForSplitSeqInTnd(cnode) != SUCCESS) {
          MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << ": Replace actual_seq_qlen and actual_seq_kvlen failed.";
        }
      } else {
        std::vector<int64_t> split_info = GetSplitIdAndRank();
        int64_t split_id = split_info[kIndex2];
        AnfNodePtr new_pre_tokens_node;
        AnfNodePtr new_next_tokens_node;
        int64_t new_sparse_mode;
        if (dynamic_seq_flag_) {
          std::tie(new_pre_tokens_node, new_next_tokens_node) =
            GetAttentionMaskAttrsByDynamicSeqDim(cnode, split_id, s1_split_num_);
          new_sparse_mode = is_attn_mask_compressed_ ? ops::kSparseBand : sparse_mode_;
        } else {
          int64_t new_pre_tokens;
          int64_t new_next_tokens;
          std::tie(new_pre_tokens, new_next_tokens) = GetAttentionMaskAttrs(split_id, s1_split_num_);
          auto status = ComputeSparseInfoForFlashAttentionScore(sparse_mode_, pre_tokens_, next_tokens_, split_id,
                                                                s1_split_num_, q_seq_len_ / s1_split_num_, kv_seq_len_,
                                                                &new_sparse_mode, &new_pre_tokens, &new_next_tokens);
          if (status != Status::SUCCESS) {
            MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode_) << "Compute new sparse info failed.";
          }
          new_pre_tokens_node = NewValueNode(MakeValue(new_pre_tokens));
          new_next_tokens_node = NewValueNode(MakeValue(new_next_tokens));
        }
        auto func_graph = cnode->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        auto manager = func_graph->manager();
        MS_EXCEPTION_IF_NULL(manager);
        SetValueInputToCNode<int64_t>(cnode, ops::kFlashAttentionScoreInputSparseModeIndex + 1, new_sparse_mode);
        manager->SetEdge(cnode, ops::kFlashAttentionScoreInputPreTokensIndex + 1, new_pre_tokens_node);
        manager->SetEdge(cnode, ops::kFlashAttentionScoreInputNextTokensIndex + 1, new_next_tokens_node);
      }
    }
    // If DropoutGenMask -> Reshape -> FlashAttentionScore, replace its.
    auto reshape_node = cnode->input(ops::kFlashAttentionScoreInputDropMaskIndex + 1);
    MS_EXCEPTION_IF_NULL(reshape_node);
    if (!IsPrimitiveCNode(reshape_node, prim::kPrimReshape)) {
      continue;
    }
    auto reshape_cnode = reshape_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(reshape_cnode);
    if (!IsPrimitiveCNode(reshape_cnode->input(kIndex1), prim::kPrimDropoutGenMask)) {
      continue;
    }
    auto dropout_gen_mask_cnode = reshape_cnode->input(kIndex1)->cast<CNodePtr>();
    // Update slice_shape for ReShape
    Shape input_slice_shape = inputs_tensor_info_.at(ops::kFlashAttentionScoreInputDropMaskIndex).slice_shape();
    ValuePtr new_shape = MakeValue(input_slice_shape);
    AnfNodePtr val = NewValueNode(new_shape);
    auto manager = cnode->func_graph()->manager();
    MS_EXCEPTION_IF_NULL(manager);
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
      q_split_axis = SizeToLong(qkv_seq_dim_);
      split_attrs = {std::make_pair(AXIS, MakeValue(q_split_axis)),
                     std::make_pair(OUTPUT_NUM, MakeValue(kLoadBalanceSplitNum))};
      *split_node = gen_g->PushBack({gen_g->NewOpInst(SPLIT, split_attrs), gen_g->virtual_input_node()});
      *keep_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(0)});
      *exchange_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(1)});
      break;
    case ops::kFlashAttentionScoreInputRealShiftIndex:
      if (is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
        split_attrs = {std::make_pair(AXIS, MakeValue<int64_t>(kInputRealShiftSeqDim)),
                       std::make_pair(OUTPUT_NUM, MakeValue(kLoadBalanceSplitNum))};
        *split_node = gen_g->PushBack({gen_g->NewOpInst(SPLIT, split_attrs), gen_g->virtual_input_node()});
        *keep_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(0)});
        *exchange_node = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *split_node, CreatInt64Imm(1)});
      } else {
        *keep_node = gen_g->virtual_input_node();
        *exchange_node = gen_g->virtual_input_node();
      }
      break;
    case ops::kFlashAttentionScoreInputDropMaskIndex:
      if (is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex]) {
        split_attrs = {std::make_pair(AXIS, MakeValue<int64_t>(kInputDropMaskSeqDim)),
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
      MS_LOG_WITH_NODE(EXCEPTION, cnode_)
        << "Invalid input index. Only 0(query), 3(real_shift), 4(drop_mask) and 6(attn_mask)"
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
                                                           const AnfNodePtr &attn_mask, const AnfNodePtr &q_shape,
                                                           const AnfNodePtr &k_shape, AnfNodePtr *fa_op,
                                                           GenerateGraph *gen_g) {
  int64_t new_sparse_mode = is_attn_mask_compressed_ ? ops::kSparseBand : sparse_mode_;
  AnfNodePtr new_pre_tokens_node;
  AnfNodePtr new_next_tokens_node;
  if (!need_update_op_attrs_mode_) {
    new_pre_tokens_node = NewValueNode(MakeValue(pre_tokens_));
    new_next_tokens_node = NewValueNode(MakeValue(next_tokens_));
  } else {
    if (dynamic_seq_flag_) {
      MS_EXCEPTION_IF_NULL(q_shape);
      MS_EXCEPTION_IF_NULL(k_shape);
      std::tie(new_pre_tokens_node, new_next_tokens_node) =
        GetAttentionMaskAttrsByDynamicSeqDim(q_shape, k_shape, split_id, split_num, gen_g);
    } else {
      int64_t new_pre_tokens, new_next_tokens;
      std::tie(new_pre_tokens, new_next_tokens) = GetAttentionMaskAttrs(split_id, split_num);
      new_pre_tokens_node = NewValueNode(MakeValue(new_pre_tokens));
      new_next_tokens_node = NewValueNode(MakeValue(new_next_tokens));
    }
  }
  auto head_num = NewValueNode(MakeValue(head_num_ / n1_split_num_));
  auto keep_prob = NewValueNode(MakeValue(keep_prob_));
  auto scale_value = NewValueNode(MakeValue(scale_value_));
  auto inner_precise = NewValueNode(MakeValue<int64_t>(0));
  auto input_layout = NewValueNode(MakeValue(input_layout_));
  auto sparse_mode = NewValueNode(MakeValue<int64_t>(new_sparse_mode));
  *fa_op = gen_g->PushBack({gen_g->NewOpInst(FLASH_ATTENTION_SCORE), q, gen_g->virtual_input_node(),
                            gen_g->virtual_input_node(), real_shift, drop_mask, gen_g->virtual_input_node(), attn_mask,
                            gen_g->virtual_input_node(), gen_g->virtual_input_node(), gen_g->virtual_input_node(),
                            head_num, keep_prob, scale_value, new_pre_tokens_node, new_next_tokens_node, inner_precise,
                            input_layout, sparse_mode});
}

std::vector<std::pair<AnfNodePtr, int64_t>> FlashAttentionScoreInfo::ReplaceGraphGetInputNodes(
  const AnfNodePtr &q_split, const AnfNodePtr &real_shift_split, const AnfNodePtr &drop_mask_split,
  const AnfNodePtr &attn_mask_split, const AnfNodePtr &flash_attention_score_keep,
  const AnfNodePtr &flash_attention_score_target, const AnfNodePtr &key_shape) {
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
                                                              std::make_pair(flash_attention_score_keep, kIndex9),
                                                              std::make_pair(flash_attention_score_keep, kIndex10),
                                                              std::make_pair(flash_attention_score_target, kIndex2),
                                                              std::make_pair(flash_attention_score_target, kIndex3)};
  if (!is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    (void)inputs_nodes.emplace_back(std::make_pair(flash_attention_score_target, kIndex4));
  }
  if (!is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex]) {
    (void)inputs_nodes.emplace_back(std::make_pair(flash_attention_score_target, kIndex5));
  }
  (void)inputs_nodes.emplace_back(std::make_pair(flash_attention_score_target, kIndex6));
  if (!is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex] || is_attn_mask_compressed_) {
    (void)inputs_nodes.emplace_back(std::make_pair(flash_attention_score_target, kIndex7));
  }
  inputs_nodes.insert(inputs_nodes.end(), {std::make_pair(flash_attention_score_target, kIndex8),
                                           std::make_pair(flash_attention_score_target, kIndex9),
                                           std::make_pair(flash_attention_score_target, kIndex10)});
  if (key_shape != nullptr) {
    inputs_nodes.insert(inputs_nodes.end(), {std::make_pair(key_shape, kIndex2)});
  }
  return inputs_nodes;
}

Status FlashAttentionScoreInfo::ComputeReplaceGraphForLoadBalance(const CNodePtr &cnode) {
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
  std::sort(swap_group_devices.begin(), swap_group_devices.end());
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

  const int64_t all_gather_idx = (split_id < target_split_id) ? 1 : 0;
  AnfNodePtr q_target;
  LoadBalanceExchange(all_gather_idx, group, q_exchange, &q_target, &gen_g);
  AnfNodePtr real_shift_target = gen_g.virtual_input_node();
  if (is_input_passed_[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    LoadBalanceExchange(all_gather_idx, group, real_shift_exchange, &real_shift_target, &gen_g);
  }
  AnfNodePtr drop_mask_target = gen_g.virtual_input_node();
  if (is_input_passed_[ops::kFlashAttentionScoreInputDropMaskIndex]) {
    LoadBalanceExchange(all_gather_idx, group, drop_mask_exchange, &drop_mask_target, &gen_g);
  }
  AnfNodePtr attn_mask_target = gen_g.virtual_input_node();
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex] && !is_attn_mask_compressed_) {
    LoadBalanceExchange(all_gather_idx, group, attn_mask_exchange, &attn_mask_target, &gen_g);
  }

  AnfNodePtr flash_attention_score_keep;
  AnfNodePtr flash_attention_score_target;
  AnfNodePtr q_target_shape = nullptr;
  AnfNodePtr q_keep_shape = nullptr;
  AnfNodePtr k_shape = nullptr;
  if (dynamic_seq_flag_) {
    q_keep_shape = gen_g.PushBack({NewValueNode(prim::kPrimShape), q_keep});
    k_shape = gen_g.PushBack({NewValueNode(prim::kPrimShape), gen_g.virtual_input_node()});
    q_target_shape = gen_g.PushBack({NewValueNode(prim::kPrimShape), q_target});
  }

  GetFlashAttentionScoreOpNode(split_id * kLoadBalanceSplitNum, s1_split_num_ * kLoadBalanceSplitNum, q_keep,
                               real_shift_keep, drop_mask_keep, attn_mask_keep, q_keep_shape, k_shape,
                               &flash_attention_score_keep, &gen_g);
  auto softmax_max_keep = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_keep,
                                          CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxMaxIndex)});
  auto softmax_sum_keep = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_keep,
                                          CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxSumIndex)});
  auto softmax_out_keep = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_keep,
                                          CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxOutIndex)});
  auto attention_out_keep = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_keep,
                                            CreatInt64Imm(ops::kFlashAttentionScoreOutputAttentionOutIndex)});

  // Insert Depend between fa_keep->fa_target
  auto q_target_depend = gen_g.PushBack({gen_g.NewOpInst(DEPEND), q_target, attention_out_keep});

  GetFlashAttentionScoreOpNode(target_split_id * kLoadBalanceSplitNum + 1, s1_split_num_ * kLoadBalanceSplitNum,
                               q_target_depend, real_shift_target, drop_mask_target, attn_mask_target, q_target_shape,
                               k_shape, &flash_attention_score_target, &gen_g);
  auto softmax_max_target = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_target,
                                            CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxMaxIndex)});
  auto softmax_sum_target = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_target,
                                            CreatInt64Imm(ops::kFlashAttentionScoreOutputSoftmaxSumIndex)});
  auto attention_out_target = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), flash_attention_score_target,
                                              CreatInt64Imm(ops::kFlashAttentionScoreOutputAttentionOutIndex)});

  AnfNodePtr attention_out_exchange;
  LoadBalanceExchange(all_gather_idx, group, attention_out_target, &attention_out_exchange, &gen_g);

  int64_t softmax_concat_axis = kOutputSoftmaxSeqDim;
  auto softmax_max_maketuple =
    gen_g.PushBack({NewValueNode(prim::kPrimMakeTuple), softmax_max_keep, softmax_max_target});
  auto softmax_max =
    gen_g.PushBack({gen_g.NewOpInst(CONCAT), softmax_max_maketuple, CreatInt64Imm(softmax_concat_axis)});
  auto softmax_sum_maketuple =
    gen_g.PushBack({NewValueNode(prim::kPrimMakeTuple), softmax_sum_keep, softmax_sum_target});
  auto softmax_sum =
    gen_g.PushBack({gen_g.NewOpInst(CONCAT), softmax_sum_maketuple, CreatInt64Imm(softmax_concat_axis)});
  int64_t attention_out_concat_axis = SizeToLong(qkv_seq_dim_);
  auto attention_out_maketuple =
    gen_g.PushBack({NewValueNode(prim::kPrimMakeTuple), attention_out_keep, attention_out_exchange});
  auto attention_out =
    gen_g.PushBack({gen_g.NewOpInst(CONCAT), attention_out_maketuple, CreatInt64Imm(attention_out_concat_axis)});
  auto output_maketuple =
    gen_g.PushBack({NewValueNode(prim::kPrimMakeTuple), softmax_max, softmax_sum, softmax_out_keep, attention_out});

  std::vector<std::pair<AnfNodePtr, int64_t>> inputs_nodes =
    ReplaceGraphGetInputNodes(q_split, real_shift_split, drop_mask_split, attn_mask_split, flash_attention_score_keep,
                              flash_attention_score_target, k_shape);

  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(inputs_nodes, output_maketuple));
  return SUCCESS;
}

ReplaceGraphPtr FlashAttentionScoreInfo::replace_graph(const CNodePtr &cnode) {
  if (s1_split_num_ > 1 && enable_load_balance_) {
    if (ComputeReplaceGraphForLoadBalance(cnode) != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode)
        << name_ << ": FlashAttentionScore S1 sequence parallel with load balance get replace graph failed";
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
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies for dependent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": No valid strategy.";
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
