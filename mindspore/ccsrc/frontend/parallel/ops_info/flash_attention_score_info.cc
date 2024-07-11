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
#include <algorithm>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/ops_func_impl/flash_attention_score.h"
#include "ops/op_enum.h"

namespace mindspore {
using mindspore::ops::FASInputLayoutMode;
namespace parallel {
namespace {
constexpr size_t kInputRealShiftSeqDim = 2;
constexpr size_t kInputDropMaskSeqDim = 2;
constexpr size_t kOutputSoftmaxSeqDim = 2;
constexpr int64_t kLoadBalanceSplitNum = 2;
enum OpAttrUpdateMode : int64_t {
  kLeftUpToLeftUp = 0,
  kLeftUpToRightDown = 1,
  kRightDownToRightDown = 2,
};
const std::vector<int64_t> needCompressAttnMask = {ops::kSparseLeftUpCausal, ops::kSparseRightDownCausal,
                                                   ops::kSparseBand, ops::kSparseBlockLocal};
const std::map<int64_t, int64_t> opAttrUpdateMap = {{ops::kSparseDefaultMask, kLeftUpToLeftUp},
                                                    {ops::kSparseLeftUpCausal, kLeftUpToRightDown},
                                                    {ops::kSparseRightDownCausal, kRightDownToRightDown},
                                                    {ops::kSparseBand, kRightDownToRightDown},
                                                    {ops::kSparseBlockLocal, kLeftUpToRightDown}};

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

RankList FlashAttentionScoreInfo::GetSPRankList() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  int64_t seq_dim = SizeToLong(dev_matrix_shape_.size()) - dev_matrix_s1_dim_ - 1;
  if (dev_matrix.GetDevicesAlongDim(seq_dim, &group_devices) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " get group devices along dim " << seq_dim << " failed.";
  }
  return group_devices;
}

Status FlashAttentionScoreInfo::InitAttnMaskStrategies() {
  if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    auto attn_mask_shape = inputs_shape_.at(GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
    int64_t s1_split_num_attn_mask = is_attn_mask_compressed_ ? 1 : s1_split_num_;
    int64_t s2_split_num_attn_mask = enable_ring_attention_ ? s1_split_num_attn_mask : 1;
    if (attn_mask_shape.size() == kSizeTwo) {
      // attn_mask_shape: (S1, S2)
      expect_strategies_[ops::kFlashAttentionScoreInputAttnMaskIndex] = {s1_split_num_attn_mask,
                                                                         s2_split_num_attn_mask};
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
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      qkv_batch_dim_ = kSizeZero;
      qkv_seq_dim_ = kSizeOne;
      qkv_head_dim_ = kSizeTwo;
      break;
    case FASInputLayoutMode::SBH:
      qkv_seq_dim_ = kSizeZero;
      qkv_batch_dim_ = kSizeOne;
      qkv_head_dim_ = kSizeTwo;
      break;
    case FASInputLayoutMode::BNSD:
      qkv_batch_dim_ = kSizeZero;
      qkv_head_dim_ = kSizeOne;
      qkv_seq_dim_ = kSizeTwo;
      break;
    case FASInputLayoutMode::BSND:
      qkv_batch_dim_ = kSizeZero;
      qkv_seq_dim_ = kSizeOne;
      qkv_head_dim_ = kSizeTwo;
      break;
    case FASInputLayoutMode::TND:
      qkv_batch_dim_ = kSizeZero;
      qkv_seq_dim_ = kSizeZero;
      qkv_head_dim_ = kSizeOne;
      break;
    default:
      MS_LOG(ERROR) << name_ << ": Not support layout in parallel currently.";
      return FAILED;
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InitAttrs() { return GetAttrs(); }

Status FlashAttentionScoreInfo::CheckInputLayout() {
  if (InferSplitNumAndDevMatrixShapeByLayout() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer device matrix shape by layout failed.";
    return FAILED;
  }

  auto query_shape = inputs_shape_[ops::kFlashAttentionScoreInputQueryIndex];
  auto key_shape = inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex];
  if (s1_split_num_ > 1 && input_layout_ == FASInputLayoutMode::TND &&
      (sparse_mode_ != ops::kSparseRightDownCausal || query_shape[0] != key_shape[0])) {
    MS_LOG(ERROR)
      << name_
      << ": When input_layout is TND, sparse_mode is 3, and the T-dimension of query and key are the same, the "
         "T-dimension of query can be sliced. query_shape: "
      << query_shape << ", key_shape: " << key_shape << ", sparse_mode: " << sparse_mode_;
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

  // Construct layout for softmax_max and softmax_sum
  std::vector<Shape> softmax_max_sum_tensor_map;
  Shape softmax_max_sum_tensor_shape;
  if (input_layout_ == FASInputLayoutMode::TND) {
    softmax_max_tensor_layout_ = query_layout;
    softmax_sum_tensor_layout_ = query_layout;
  } else {
    softmax_max_sum_tensor_map.push_back(query_layout.tensor_map_before()[qkv_batch_dim_]);              // B
    softmax_max_sum_tensor_shape.push_back(query_layout.tensor_shape_before().array()[qkv_batch_dim_]);  // B
    softmax_max_sum_tensor_map.push_back(query_layout.tensor_map_before()[qkv_head_dim_]);               // N
    softmax_max_sum_tensor_shape.push_back(head_num_);                                                   // N
    softmax_max_sum_tensor_map.push_back(query_layout.tensor_map_before()[qkv_seq_dim_]);                // S
    softmax_max_sum_tensor_shape.push_back(query_layout.tensor_shape_before().array()[qkv_seq_dim_]);    // S
    softmax_max_sum_tensor_map.push_back({MAP_NONE});                                                    // 8
    softmax_max_sum_tensor_shape.push_back(8);                                                           // 8
    softmax_max_tensor_layout_.InitFromExtendVector(query_layout.device_arrangement_origin().array(),
                                                    softmax_max_sum_tensor_map,
                                                    outputs_shape()[ops::kFlashAttentionScoreOutputSoftmaxMaxIndex]);
    softmax_sum_tensor_layout_.InitFromExtendVector(query_layout.device_arrangement_origin().array(),
                                                    softmax_max_sum_tensor_map,
                                                    outputs_shape()[ops::kFlashAttentionScoreOutputSoftmaxSumIndex]);
  }

  // Construct layout for softmax_out
  softmax_out_tensor_layout_.InitFromExtendVector(query_layout.device_arrangement_origin().array(),
                                                  std::vector<Shape>{{MAP_NONE}},
                                                  outputs_shape()[ops::kFlashAttentionScoreOutputSoftmaxOutIndex]);
  attention_out_tensor_layout_ = query_layout;
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
    if (is_auto_parallel_) {
      if (g_device_manager->CheckDeviceList(repeated_rank_list) != SUCCESS) {
        MS_LOG(INFO) << name_ << ": Try to create communication group : " << repeated_rank_list
                     << " failed in auto parallel mode, "
                        "this error can be ignored in parallel strategies searching step";
        return FAILED;
      }
      return SUCCESS;
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

Status FlashAttentionScoreInfo::GetAttrs() {
  InitIsInputPassed();
  head_num_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFlashAttentionScoreInputHeadNumIndex + 1);
  keep_prob_ = GetInputValueFromCNode<float>(cnode_, ops::kFlashAttentionScoreInputKeepProbIndex + 1);
  scale_value_ = GetInputValueFromCNode<float>(cnode_, ops::kFlashAttentionScoreInputScaleValueIndex + 1);
  pre_tokens_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFlashAttentionScoreInputPreTokensIndex + 1);
  next_tokens_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFlashAttentionScoreInputNextTokensIndex + 1);
  input_layout_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFlashAttentionScoreInputLayoutIndex + 1);
  sparse_mode_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFlashAttentionScoreInputSparseModeIndex + 1);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  enable_load_balance_ = ms_context->get_param<bool>(MS_CTX_ENABLE_FLASH_ATTENTION_LOAD_BALANCE);

  if (input_layout_ == FASInputLayoutMode::TND && enable_load_balance_) {
    MS_LOG(WARNING) << name_ << ": Load balancing is not supported in the layout 'TND' and will be disabled.";
    enable_load_balance_ = false;
  }

  auto enable_ring_attention_iter = attrs_.find(ENABLE_RING_ATTENTION);
  if (enable_ring_attention_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(enable_ring_attention_iter->second);
    if (enable_ring_attention_iter->second->isa<BoolImm>()) {
      enable_ring_attention_ = enable_ring_attention_iter->second->cast<BoolImmPtr>()->value();
      enable_load_balance_ = false;
      MS_LOG(DEBUG) << "enable_ring_attention_: " << enable_ring_attention_;
    } else {
      MS_LOG(ERROR) << "enable_ring_attention should be bool";
    }
  }
  if (enable_ring_attention_) {
    if (input_layout_ != FASInputLayoutMode::BSH && input_layout_ != FASInputLayoutMode::BNSD) {
      MS_LOG(ERROR) << "Ring attention currently only supports BSH and BNSD layout";
    }
    if (sparse_mode_ != 0) {
      MS_LOG(ERROR) << "Ring attention currently only supports sparse mode 0";
    }
    if (keep_prob_ != 1.0) {
      MS_LOG(ERROR) << "Ring attention currently only supports keep prob 1.0";
    }
    if (is_input_passed_[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
      MS_LOG(ERROR) << "Ring attention do not need input attn mask";
    }
  }

  is_attn_mask_compressed_ =
    std::find(needCompressAttnMask.begin(), needCompressAttnMask.end(), sparse_mode_) != needCompressAttnMask.end();
  need_update_op_attrs_mode_ = sparse_mode_ != ops::kSparseAllMask;
  if (InitQKVHeadAndSeqDimFromInputLayout() != Status::SUCCESS) {
    return FAILED;
  }

  kv_split_ = inputs_shape_[ops::kFlashAttentionScoreInputQueryIndex][qkv_head_dim_] !=
              inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex][qkv_head_dim_] * head_num_;

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
  if (head_num_ % query_strategy[qkv_head_dim_] != 0) {
    MS_LOG(ERROR) << name_ << ": head_num % query_strategy[" << qkv_head_dim_ << "] must be 0, but got " << head_num_
                  << "(head_num) and " << query_strategy[qkv_head_dim_] << "(query_strategy[" << qkv_head_dim_ << "])";
    return FAILED;
  }
  if (!kv_split_ && key_strategy[qkv_head_dim_] != 1) {
    MS_LOG(ERROR) << name_ << ": Under the MQA，the hidden-dim of input 'key' cannot be split.";
    return FAILED;
  }

  if (input_layout_ == FASInputLayoutMode::TND) {
    if (query_strategy[qkv_seq_dim_] != key_strategy[qkv_seq_dim_]) {
      MS_LOG(ERROR)
        << name_ << ": The split num of seq-dim between query and key must be the same when layout is 'TND'. But got "
        << query_strategy[qkv_seq_dim_] << " and " << key_strategy[qkv_seq_dim_];
      return FAILED;
    }
  } else {
    auto s2_split_num = key_strategy[qkv_seq_dim_];
    if (s2_split_num != 1 && !enable_ring_attention_) {
      MS_LOG(ERROR) << name_ << ": The S-Dimension of input 'key' cannot be split, but got the strategy of key is "
                    << key_strategy;
      return FAILED;
    }
  }

  if (input_layout_ == FASInputLayoutMode::TND) {
    batch_split_num_ = key_strategy[qkv_batch_dim_];
    s1_split_num_ = query_strategy[qkv_batch_dim_] / batch_split_num_;
  } else {
    batch_split_num_ = query_strategy[qkv_batch_dim_];
    s1_split_num_ = query_strategy[qkv_seq_dim_];
  }
  n1_split_num_ = query_strategy[qkv_head_dim_];

  s2_split_num_ = enable_ring_attention_ ? s1_split_num_ : 1;

  n2_split_num_ = key_strategy[qkv_head_dim_];

  if (kv_split_ && n1_split_num_ != n2_split_num_) {
    MS_LOG(ERROR) << name_ << ": The split num of N1-dim and N2-dim must be equal if N2 > 1, but got " << n1_split_num_
                  << " and " << n2_split_num_;
    return FAILED;
  }

  if (s1_split_num_ > 1 && input_layout_ == FASInputLayoutMode::TND) {
    MS_LOG(ERROR)
      << name_
      << ": Currently, input_layout is TND, and the seq dimension of query is segmented. Please use Layout to "
         "set the strategy.";
    return FAILED;
  }

  if (InitExpectedStrategies() != SUCCESS) {
    return FAILED;
  }
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
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
    case FASInputLayoutMode::BSND:
    case FASInputLayoutMode::TND:
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
  if (input_layout_ == FASInputLayoutMode::TND) {
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
    n1_split_num_ = 1;
    for (auto map_id : query_head_map) {
      n1_split_num_ *= GetSplitNumByMapId(dev_matrix_shape, map_id);
    }
  } else {
    if (query_batch_map.size() != 1 || query_seq_map.size() != 1 || query_head_map.size() != 1) {
      MS_LOG(ERROR) << name_
                    << ": Each dimension of query can only be mapped to one device matrix dimension, but got the "
                       "tensor info of query is "
                    << query_layout.ToString();
      return FAILED;
    }
    dev_matrix_batch_dim_ = query_batch_map[0];
    dev_matrix_s1_dim_ = query_seq_map[0];
    dev_matrix_n1_dim_ = query_head_map[0];
    n1_split_num_ = GetSplitNumByMapId(dev_matrix_shape, dev_matrix_n1_dim_);
  }
  batch_split_num_ = GetSplitNumByMapId(dev_matrix_shape, dev_matrix_batch_dim_);
  s1_split_num_ = GetSplitNumByMapId(dev_matrix_shape, dev_matrix_s1_dim_);
  if (s1_split_num_ > 1 && GetSplitNumByTensorMap(dev_matrix_shape, query_seq_map) !=
                             GetSplitNumByTensorMap(dev_matrix_shape, key_seq_map) * s1_split_num_) {
    MS_LOG(EXCEPTION) << name_ << ": Cannot split the seq-dimension of key. query_seq_slice: "
                      << GetSplitNumByTensorMap(dev_matrix_shape, query_seq_map)
                      << ", key_seq_slice: " << GetSplitNumByTensorMap(dev_matrix_shape, key_seq_map)
                      << ", s1_split_num: " << s1_split_num_;
  }
  return SUCCESS;
}

Status FlashAttentionScoreInfo::InferTensorMap() {
  if (InitInputsTensorMap() != SUCCESS) {
    return FAILED;
  }
  if (input_layout_ == FASInputLayoutMode::TND) {
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
  kv_seq_length = inputs_shape_[ops::kFlashAttentionScoreInputKeyIndex][qkv_seq_dim_];
  q_seq_length = inputs_shape_[ops::kFlashAttentionScoreInputQueryIndex][qkv_seq_dim_];
  int64_t q_len_each_split = q_seq_length / split_num;
  int64_t new_pre_tokens =
    (sparse_mode_ == ops::kSparseDefaultMask || sparse_mode_ == ops::kSparseBand) ? pre_tokens_ : kv_seq_length;
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

Status FlashAttentionScoreInfo::ReplaceActualSeqLenForSplitSeqInTnd(const CNodePtr &cnode) {
  std::vector<int64_t> split_info = GetSplitIdAndRank();
  int64_t tq = inputs_shape_[GetStrategyRealIndex(ops::kFlashAttentionScoreInputQueryIndex)][qkv_batch_dim_];
  int64_t tk = inputs_shape_[GetStrategyRealIndex(ops::kFlashAttentionScoreInputKeyIndex)][qkv_batch_dim_];
  int64_t slice_tq = tq / s1_split_num_ / batch_split_num_;
  int64_t slice_tk = tk / batch_split_num_;
  int64_t split_id = split_info[kIndex2];
  int64_t offset = slice_tq * split_id;
  if (!is_input_passed_[ops::kFlashAttentionScoreInputActualSeqQlenIndex] ||
      !is_input_passed_[ops::kFlashAttentionScoreInputActualSeqKVlenIndex]) {
    MS_LOG(ERROR) << name_ << ": The input 'actual_seq_qlen' and 'actual_seq_kvlen' cannot be None under 'TND'.";
    return FAILED;
  }
  auto actual_seq_qlen_input_index = ops::kFlashAttentionScoreInputActualSeqQlenIndex + 1;
  auto actual_seq_kvlen_input_index = ops::kFlashAttentionScoreInputActualSeqKVlenIndex + 1;
  auto actual_seq_qlen_node = cnode->input(actual_seq_qlen_input_index);
  auto actual_seq_kvlen_node = cnode->input(actual_seq_kvlen_input_index);

  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // new_actual_seq_qlen = clip(actual_seq_qlen - offset, 0, slice_tq)
  auto qlen_offset_sub_cnode =
    func_graph->NewCNode({NewValueNode(prim::kPrimSub), actual_seq_qlen_node, CreateInt32Tensor(offset, true)});
  auto new_actual_seq_qlen_cnode =
    func_graph->NewCNode({NewValueNode(prim::kPrimClipByValue), qlen_offset_sub_cnode, CreateInt32Tensor(0, true),
                          CreateInt32Tensor(slice_tq, true)});
  manager->SetEdge(cnode, actual_seq_qlen_input_index, new_actual_seq_qlen_cnode);

  // new_actual_seq_kvlen = actual_seq_kvlen - (ReLU(actual_seq_qlen - offset) - new_actual_seq_qlen)
  auto relu_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimReLU), qlen_offset_sub_cnode});
  auto kvlen_offset_sub_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimSub), actual_seq_qlen_node, relu_cnode});
  auto tmp_new_actual_seq_kvlen_cnode =
    func_graph->NewCNode({NewValueNode(prim::kPrimSub), actual_seq_kvlen_node, kvlen_offset_sub_cnode});

  // new_actual_seq_kvlen[actual_seq_kvlen == slice_tk] = slice_tk
  auto equal =
    func_graph->NewCNode({NewValueNode(prim::kPrimEqual), actual_seq_kvlen_node, CreateInt32Tensor(slice_tk, true)});
  auto new_actual_seq_kvlen_cnode = func_graph->NewCNode(
    {NewValueNode(prim::kPrimSelect), equal, actual_seq_kvlen_node, tmp_new_actual_seq_kvlen_cnode});
  manager->SetEdge(cnode, actual_seq_kvlen_input_index, new_actual_seq_kvlen_cnode);

  return SUCCESS;
}

void FlashAttentionScoreInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    SetValueInputToCNode<int64_t>(cnode, ops::kFlashAttentionScoreInputHeadNumIndex + 1, head_num_ / n1_split_num_);
    if (s1_split_num_ > 1 && !enable_load_balance_ && need_update_op_attrs_mode_) {
      if (input_layout_ == FASInputLayoutMode::TND) {
        if (ReplaceActualSeqLenForSplitSeqInTnd(cnode) != SUCCESS) {
          MS_LOG(EXCEPTION) << name_ << ": Replace actual_seq_qlen and actual_seq_kvlen failed.";
        }
      } else {
        int64_t new_pre_tokens, new_next_tokens;
        std::vector<int64_t> split_info = GetSplitIdAndRank();
        int64_t split_id = split_info[kIndex2];
        std::tie(new_pre_tokens, new_next_tokens) = GetAttentionMaskAttrs(split_id, s1_split_num_);
        int64_t new_sparse_mode = is_attn_mask_compressed_ ? ops::kSparseBand : sparse_mode_;
        SetValueInputToCNode<int64_t>(cnode, ops::kFlashAttentionScoreInputSparseModeIndex + 1, new_sparse_mode);
        SetValueInputToCNode<int64_t>(cnode, ops::kFlashAttentionScoreInputPreTokensIndex + 1, new_pre_tokens);
        SetValueInputToCNode<int64_t>(cnode, ops::kFlashAttentionScoreInputNextTokensIndex + 1, new_next_tokens);
      }
    }
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
                            gen_g->virtual_input_node(), gen_g->virtual_input_node(), gen_g->virtual_input_node()});
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
                              flash_attention_score_target);

  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(inputs_nodes, output_maketuple));
  return SUCCESS;
}

ReplaceGraphPtr FlashAttentionScoreInfo::replace_graph(const CNodePtr &cnode) {
  if (s1_split_num_ > 1 && enable_load_balance_) {
    if (ComputeReplaceGraphForLoadBalance(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_
                        << ": FlashAttentionScore S1 sequence parallel with load balance get replace graph failed";
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
