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
#include <map>
#include <tuple>

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
constexpr size_t kRank2 = 2;
constexpr size_t kRank3 = 3;
constexpr size_t kRank4 = 4;
constexpr size_t kDpAxis = 2;
enum SparseMode : int64_t {
  kSparseDefaultMask = 0,
  kSparseAllMask,
  kSparseLeftUpCausal,
  kSparseRightDownCausal,
  kSparseBand,
  kSparsePrefix,
  kSparseGlobal,
  kSparseDilated,
  kSparseBlockLocal,
};
enum OpAttrUpdateMode : int64_t {
  kLeftUpToLeftUp = 0,
  kLeftUpToRightDown = 1,
  kRightDownToRightDown = 2,
};

const std::map<int64_t, int64_t> opAttrUpdateMap = {{kSparseDefaultMask, kLeftUpToLeftUp},
                                                    {kSparseLeftUpCausal, kLeftUpToRightDown},
                                                    {kSparseRightDownCausal, kRightDownToRightDown},
                                                    {kSparseBand, kRightDownToRightDown}};

const std::vector<int64_t> needCompressAttnMask = {kSparseLeftUpCausal, kSparseRightDownCausal, kSparseBand};
}  // namespace

template <typename T>
T GetInputValueFromCNode(const CNodePtr &cnode, size_t index) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto inputs = cnode->inputs();
  if (index >= inputs.size()) {
    MS_LOG(EXCEPTION) << "The input index (" << index << ") is exceed of inputs size (" << inputs.size() << ").";
  }
  auto input_node = inputs[index];
  MS_EXCEPTION_IF_NULL(input_node);
  if (!input_node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The " << GetSerialNumberString(index) << " input is not a value node.";
  }
  auto value = input_node->cast<ValueNodePtr>()->value();
  MS_EXCEPTION_IF_NULL(value);
  return GetValue<T>(value);
}

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

void FusedInferAttentionScoreInfo::SetOptionalInputs() {
  optional_inputs_.resize(ops::kFusedInferAttentionScoreInputKvPaddingSizeIndex + 1, true);
  size_t valid_input_index = 3;  // after qkv index
  for (size_t index = ops::kFusedInferAttentionScoreInputPseShiftIndex; index < input_value_.size(); index++) {
    auto optional_input_ptr = input_value_[index];
    if (optional_input_ptr == nullptr || optional_input_ptr->isa<tensor::Tensor>()) {
      if (index == ops::kFusedInferAttentionScoreInputPseShiftIndex && valid_input_index < inputs_shape_new_.size()) {
        auto padding_mask_shape = inputs_shape_new_[valid_input_index]->GetAllElements();
        padding_mask_rank_ = padding_mask_shape[0].size();
      }
      if (index == ops::kFusedInferAttentionScoreInputAttnMaskIndex && valid_input_index < inputs_shape_new_.size()) {
        auto atten_mask_shape = inputs_shape_new_[valid_input_index]->GetAllElements();
        atten_mask_rank_ = atten_mask_shape[0].size();
      }
      valid_input_index++;
    } else if (optional_input_ptr->isa<None>()) {
      optional_inputs_[index] = False;
    }
  }
  // init optional tensor map
  Shape atten_mask_tensor_map(atten_mask_rank_, -1);
  Shape atten_mask_strategy_map(atten_mask_rank_, 0);
  Shape padding_mask_tensor_map(padding_mask_rank_, -1);
  Shape padding_mask_strategy_map(padding_mask_rank_, 0);
  optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = atten_mask_tensor_map;
  optional_tensor_map_[ops::kFusedInferAttentionScoreInputPseShiftIndex] = padding_mask_tensor_map;
  optional_op_strategies_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = atten_mask_strategy_map;
  optional_op_strategies_[ops::kFusedInferAttentionScoreInputPseShiftIndex] = padding_mask_strategy_map;
}

void FusedInferAttentionScoreInfo::GenerateExpectStrategies() {
  expect_strategies_ = {{}, {}, {}, {dp_, 1, 1, 1}, {dp_, 1, 1}, {dp_}, {dp_}, {}, {}, {}, {}, {}, {}, {}, {}};
  if (atten_mask_rank_ == kRank2) {
    expect_strategies_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {1, 1};
  }
  if (atten_mask_rank_ == kRank3) {
    expect_strategies_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {dp_, 1, 1};
  }
  if (padding_mask_rank_ == kRank2) {
    expect_strategies_[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {1, 1};
  }
  if (padding_mask_rank_ == kRank3) {
    expect_strategies_[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {dp_, 1, 1};
  }
}

Status FusedInferAttentionScoreInfo::GetAttrs() {
  head_num_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFusedInferAttentionScoreInputNumHeadsIndex + 1);
  kv_head_num_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFusedInferAttentionScoreInputNumKeyValueHeadsIndex + 1);
  pre_tokens_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFusedInferAttentionScoreInputPreTokensIndex + 1);
  next_tokens_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFusedInferAttentionScoreInputNextTokensIndex + 1);
  sparse_mode_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFusedInferAttentionScoreInputSparseModeIndex + 1);
  is_attn_mask_compressed_ =
    std::find(needCompressAttnMask.begin(), needCompressAttnMask.end(), sparse_mode_) != needCompressAttnMask.end();
  need_update_op_attrs_mode_ = sparse_mode_ != kSparseAllMask;
  input_layout_ = GetInputValueFromCNode<int64_t>(cnode_, ops::kFusedInferAttentionScoreInputLayoutIndex + 1);
  softmax_lse_flag_ = GetInputValueFromCNode<bool>(cnode_, ops::kFusedInferAttentionScoreInputSoftmaxLseFlagIndex + 1);
  SetOptionalInputs();
  return SUCCESS;
}

Status FusedInferAttentionScoreInfo::CheckQueryStrategy(const NewStrategies &stra) {
  auto query_strategys = stra[ops::kFusedInferAttentionScoreInputQueryIndex]->GetAllElements();
  auto key_strategys = stra[ops::kFusedInferAttentionScoreInputKeyIndex]->GetAllElements();
  auto query_strategy = query_strategys[0];
  auto query_input = inputs_shape_new_[ops::kFusedInferAttentionScoreInputQueryIndex]->GetAllElements()[0];
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      if (head_num_ % query_strategy[kInputQueryHiddenDimBSH] != 0) {
        MS_LOG(ERROR) << "For " << name_ << ": head_num % query_strategy[2] must be 0, but got " << head_num_
                      << "(head_num) and " << query_strategy[kInputQueryHiddenDimBSH] << "(query_strategy[2])";
        return FAILED;
      }
      is_ifa_ = query_input[kInputQuerySeqDimBSH] == 1;
      dp_ = query_strategy[kInputQueryBatchDimBSH];
      mp_ = query_strategy[kInputQueryHiddenDimBSH];
      sp_ = key_strategys[0][kInputQuerySeqDimBSH];
      break;
    case FASInputLayoutMode::BNSD:
      if (!CheckStrategyOnIndex(query_strategy[kInputQueryHiddenDimBNSD], 1, "D-Dimention", "query")) {
        return FAILED;
      }
      is_ifa_ = query_input[kInputQuerySeqDimBNSD] == 1;
      dp_ = query_strategy[kInputQueryBatchDimBNSD];
      mp_ = query_strategy[kInputQueryNDimBNSD];
      sp_ = key_strategys[0][kInputQuerySeqDimBNSD];
      break;
    case FASInputLayoutMode::BSND:
      if (!CheckStrategyOnIndex(query_strategy[kInputQueryHiddenDimBSND], 1, "D-Dimention", "query")) {
        return FAILED;
      }
      is_ifa_ = query_input[kInputQuerySeqDimBSND] == 1;
      dp_ = query_strategy[kInputQueryBatchDimBSND];
      mp_ = query_strategy[kInputQueryNDimBSND];
      sp_ = key_strategys[0][kInputQuerySeqDimBSND];
      break;
    default:
      MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
      return FAILED;
  }
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

  if (CheckQueryStrategy(stra) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": The query strategy check failed.";
    return FAILED;
  }

  if (optional_inputs_.empty()) {
    SetOptionalInputs();
  }
  return SUCCESS;
}

Status FusedInferAttentionScoreInfo::InferDevMatrixShape() {
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      dev_matrix_shape_ = {dp_, sp_, mp_};
      dev_matrix_batch_dim_ = kDpAxis;
      dev_matrix_s1_dim_ = 1;
      dev_matrix_n1_dim_ = 0;
      break;
    case FASInputLayoutMode::BNSD:
      dev_matrix_shape_ = {dp_, mp_, sp_};
      dev_matrix_batch_dim_ = kDpAxis;
      dev_matrix_s1_dim_ = 0;
      dev_matrix_n1_dim_ = 1;
      break;
    case FASInputLayoutMode::BSND:
      dev_matrix_shape_ = {dp_, sp_, mp_};
      dev_matrix_batch_dim_ = kDpAxis;
      dev_matrix_s1_dim_ = 1;
      dev_matrix_n1_dim_ = 0;
      break;
    default:
      MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
      return FAILED;
  }
  return SUCCESS;
}

void FusedInferAttentionScoreInfo::InferOptionalTensorMap() {
  if (is_ifa_) {  // IFA
    if (atten_mask_rank_ == kRank2) {
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex][0] = dev_matrix_batch_dim_;
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex][1] = dev_matrix_s1_dim_;
    } else if (atten_mask_rank_ == kRank3) {
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex][0] = dev_matrix_batch_dim_;
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex][2] = dev_matrix_s1_dim_;
    } else if (atten_mask_rank_ == kRank4) {
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex][0] = dev_matrix_batch_dim_;
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex][3] = dev_matrix_s1_dim_;
    }
  } else {
    int32_t pos_s = 0;
    if (sparse_mode_ == 0) {
      pos_s = 1;
    } else {
      pos_s = -1;
    }
    if (atten_mask_rank_ == kRank2) {
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {pos_s, -1};
      optional_op_strategies_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {0, 0};
    }
    if (atten_mask_rank_ == kRank3) {
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {2, pos_s, -1};
      optional_op_strategies_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {2, 0, 0};
    }
    if (atten_mask_rank_ == kRank4) {
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {2, -1, pos_s, -1};
      optional_op_strategies_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] = {2, 0, 0, 0};
    }
    if (padding_mask_rank_ == kRank2) {
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {pos_s, -1};
      optional_op_strategies_[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {0, 0};
    }
    if (padding_mask_rank_ == kRank3) {
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {2, pos_s, -1};
      optional_op_strategies_[ops::kFusedInferAttentionScoreInputPseShiftIndex] = {2, 0, 0};
    }
    if (padding_mask_rank_ == kRank4) {
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputPseShiftIndex][0] = dev_matrix_batch_dim_;
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputPseShiftIndex][1] = dev_matrix_n1_dim_;
      optional_tensor_map_[ops::kFusedInferAttentionScoreInputPseShiftIndex][3] = dev_matrix_s1_dim_;
    }
  }

  for (auto index = static_cast<size_t>(ops::kFusedInferAttentionScoreInputPseShiftIndex);
       index < optional_inputs_.size(); index++) {
    if (optional_inputs_[index]) {
      (void)inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(optional_tensor_map_[index]));
    }
  }
}

int64_t FusedInferAttentionScoreInfo::GetSplitIdAndRank() {
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
    MS_LOG(EXCEPTION) << "PromptFlashAttention S1 sequence parallel get split id failed. "
                      << "rank " << rank << " not in group " << group_devices;
  }
  int64_t split_id = iter - group_devices.begin();
  return split_id;
}

int64_t LongAddNew(int64_t base, int64_t shift) {
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

std::tuple<int64_t, int64_t> FusedInferAttentionScoreInfo::GetAttentionMaskAttrs(const int64_t split_id,
                                                                                 const int64_t split_num) {
  int64_t kv_seq_length;
  int64_t q_seq_length;
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      kv_seq_length =
        inputs_shape_new_[ops::kFusedInferAttentionScoreInputKeyIndex]->GetElement(0)->GetValue()[kInputQuerySeqDimBSH];
      q_seq_length = inputs_shape_new_[ops::kFusedInferAttentionScoreInputQueryIndex]
                       ->GetElement(0)
                       ->GetValue()[kInputQuerySeqDimBSH];
      break;
    case FASInputLayoutMode::BSND:
      kv_seq_length = inputs_shape_new_[ops::kFusedInferAttentionScoreInputKeyIndex]
                        ->GetElement(0)
                        ->GetValue()[kInputQuerySeqDimBSND];
      q_seq_length = inputs_shape_new_[ops::kFusedInferAttentionScoreInputQueryIndex]
                       ->GetElement(0)
                       ->GetValue()[kInputQuerySeqDimBSND];
      break;
    default:
      kv_seq_length = inputs_shape_new_[ops::kFusedInferAttentionScoreInputKeyIndex]
                        ->GetElement(0)
                        ->GetValue()[kInputQuerySeqDimBNSD];
      q_seq_length = inputs_shape_new_[ops::kFusedInferAttentionScoreInputQueryIndex]
                       ->GetElement(0)
                       ->GetValue()[kInputQuerySeqDimBNSD];
      break;
  }
  int64_t q_len_each_split = q_seq_length / split_num;
  int64_t new_pre_tokens;
  if (sparse_mode_ == kSparseDefaultMask || sparse_mode_ == kSparseBand) {
    new_pre_tokens = pre_tokens_;
  } else if (sparse_mode_ == kSparseLeftUpCausal) {
    new_pre_tokens = q_seq_length;
  } else {
    new_pre_tokens = kv_seq_length;
  }
  int64_t new_next_tokens = (sparse_mode_ == kSparseDefaultMask || sparse_mode_ == kSparseBand) ? next_tokens_ : 0;
  switch (opAttrUpdateMap.at(sparse_mode_)) {
    case kLeftUpToLeftUp:
      new_pre_tokens = LongAddNew(new_pre_tokens, -split_id * q_len_each_split);
      new_next_tokens = LongAddNew(new_next_tokens, split_id * q_len_each_split);
      break;
    case kLeftUpToRightDown:
      new_pre_tokens = LongAddNew(new_pre_tokens, (kv_seq_length - (split_id + 1) * q_len_each_split));
      new_next_tokens = LongAddNew(new_next_tokens, -(kv_seq_length - (split_id + 1) * q_len_each_split));
      break;
    case kRightDownToRightDown:
      new_pre_tokens = LongAddNew(new_pre_tokens, (split_num - split_id - 1) * (q_seq_length / split_num));
      new_next_tokens = LongAddNew(new_next_tokens, -(split_num - split_id - 1) * (q_seq_length / split_num));
      break;
    default:
      MS_LOG(EXCEPTION) << "Invalid sparse mode " << sparse_mode_ << ", sparse mode should be one of [0, 2, 3, 4].";
  }
  return std::make_tuple(new_pre_tokens, new_next_tokens);
}

Status FusedInferAttentionScoreInfo::InferTensorMap() {
  if (optional_inputs_.empty()) {
    SetOptionalInputs();
  }
  Shape validMap;
  Shape valid_q_map;
  switch (input_layout_) {
    case FASInputLayoutMode::BSH:
      // (b, s, h) -> (dp_, sp_, mp_) -> (2, 1, 0)
      validMap = Shape{2, 1, 0};
      valid_q_map = is_ifa_ ? Shape{2, -1, 0} : validMap;
      break;
    case FASInputLayoutMode::BNSD:
      // (b, n, s) -> (dp_, mp_, sp_) -> (2, 1, 0) BNSD -> (2, 1, 0, )
      validMap = Shape{2, 1, 0, -1};
      valid_q_map = is_ifa_ ? Shape{2, 1, -1, -1} : validMap;
      break;
    case FASInputLayoutMode::BSND:
      // (b, s, n) -> (dp_, sp_, mp_) -> (2, 1, 0) BSND -> (2, 1, 0, )
      validMap = Shape{2, 1, 0, -1};
      valid_q_map = is_ifa_ ? Shape{2, -1, 0, -1} : validMap;
      break;
    default:
      MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
      return FAILED;
  }
  Shape lse_tensor_map = Shape{dev_matrix_batch_dim_, dev_matrix_n1_dim_, -1, -1};
  std::vector<ShapeBasePtr> key_value_tensorist_map_idx;
  for (size_t i = 0; i < inputs_shape_new_[ops::kFusedInferAttentionScoreInputKeyIndex]->size(); i++) {
    key_value_tensorist_map_idx.emplace_back(std::make_shared<ShapeValue>(validMap));
  }
  inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(valid_q_map));                 // query
  inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeList>(key_value_tensorist_map_idx));  // key
  inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeList>(key_value_tensorist_map_idx));  // value
  outputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(valid_q_map));                // attention_out
  outputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(lse_tensor_map));             // softmax_lse

  InferOptionalTensorMap();
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
  if (optional_inputs_.empty()) {
    SetOptionalInputs();
  }
  split_flag_list_[ops::kFusedInferAttentionScoreInputQueryIndex] = true;
  split_flag_list_[ops::kFusedInferAttentionScoreInputKeyIndex] = true;
  split_flag_list_[ops::kFusedInferAttentionScoreInputValueIndex] = true;
  split_flag_list_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] =
    (optional_inputs_[ops::kFusedInferAttentionScoreInputAttnMaskIndex] && atten_mask_rank_ > kRank2);
  split_flag_list_[ops::kFusedInferAttentionScoreInputPseShiftIndex] =
    (optional_inputs_[ops::kFusedInferAttentionScoreInputPseShiftIndex] && padding_mask_rank_ > kRank2);
  split_flag_list_[ops::kFusedInferAttentionScoreInputActualSeqLengthsIndex] =
    optional_inputs_[ops::kFusedInferAttentionScoreInputActualSeqLengthsIndex];
  split_flag_list_[ops::kFusedInferAttentionScoreInputActualSeqLengthsKvIndex] =
    optional_inputs_[ops::kFusedInferAttentionScoreInputActualSeqLengthsKvIndex];
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
                                  kv_head_num_ / mp_);
    if (sp_ > 1 && need_update_op_attrs_mode_) {
      int64_t split_id = GetSplitIdAndRank();
      int64_t new_pre_tokens, new_next_tokens;
      std::tie(new_pre_tokens, new_next_tokens) = GetAttentionMaskAttrs(split_id, sp_);
      int64_t new_sparse_mode = is_attn_mask_compressed_ ? kSparseBand : sparse_mode_;
      SetValueInputToCNode<int64_t>(cnode, ops::kFusedInferAttentionScoreInputSparseModeIndex + 1, new_sparse_mode);
      SetValueInputToCNode<int64_t>(cnode, ops::kFusedInferAttentionScoreInputPreTokensIndex + 1, new_pre_tokens);
      SetValueInputToCNode<int64_t>(cnode, ops::kFusedInferAttentionScoreInputNextTokensIndex + 1, new_next_tokens);
    }
    cnode->set_input(0, NewValueNode(clone_prim)->cast<AnfNodePtr>());
  }
}

void FusedInferAttentionScoreInfo::SplitKVSequenceGraph(const Group &group, GenerateGraph *gen_g,
                                                        AnfNodePtr *fused_attention_score, AnfNodePtr *output) {
  *fused_attention_score = gen_g->PushBack({gen_g->NewOpInst(FUSED_INFER_ATTENTION_SCORE),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node(),
                                            gen_g->virtual_input_node()});
  auto attention_out = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *fused_attention_score,
                                        CreatInt64Imm(ops::kFusedInferAttentionScoreOutputAttentionOutIndex)});
  auto softmax_max_lse = gen_g->PushBack({gen_g->NewOpInst(TUPLE_GETITEM), *fused_attention_score,
                                          CreatInt64Imm(ops::kFusedInferAttentionScoreOutputSoftmaxLseIndex)});

  auto dtype = gen_g->PushBack({gen_g->NewOpInst(DTYPE), attention_out});
  auto dtype_id =
    gen_g->PushBack({gen_g->NewOpInst(DTYPETOENUM), CreateStringImm("DtypeToEnum"), CreateStringImm("dtype"), dtype});
  auto cast_lse = gen_g->PushBack({gen_g->NewOpInst(CAST), softmax_max_lse, dtype_id});

  OperatorAttrs all_gather_attrs = {std::make_pair(GROUP, MakeValue(group.name()))};
  AnfNodePtr all_gather = gen_g->PushBack({gen_g->NewOpInst(ALL_GATHER, all_gather_attrs), cast_lse});

  auto max = gen_g->PushBack({gen_g->NewOpInst(MAX), all_gather});

  auto sub = gen_g->PushBack({gen_g->NewOpInst(SUB), cast_lse, max});

  auto exp = gen_g->PushBack({gen_g->NewOpInst(EXP), sub});

  Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group.name()));
  OperatorAttrs attrs = {attr_op, attr_group};
  auto reduce_op_lse_sum = gen_g->PushBack({gen_g->NewOpInst(ALL_REDUCE, attrs), exp});

  auto log = gen_g->PushBack({gen_g->NewOpInst(LOG), reduce_op_lse_sum});

  auto add = gen_g->PushBack({gen_g->NewOpInst(ADD), log, max});

  auto sub_lse = gen_g->PushBack({gen_g->NewOpInst(SUB), cast_lse, add});

  auto exp_lse = gen_g->PushBack({gen_g->NewOpInst(EXP), sub_lse});

  if (input_layout_ == FASInputLayoutMode::BSH) {
    auto query_input = inputs_shape_new_[ops::kFusedInferAttentionScoreInputQueryIndex]->GetAllElements()[0];
    auto batch_size = query_input[kInputQueryBatchDimBSH];
    auto hidden_size = query_input[kInputQueryHiddenDimBSH];
    auto hidden_dim = hidden_size / head_num_;
    std::vector<int64_t> make_shape = {batch_size / dp_, head_num_ / mp_, 1, hidden_dim};
    attention_out = gen_g->PushBack({gen_g->NewOpInst(RESHAPE), attention_out, NewValueNode(MakeValue(make_shape))});
  }

  auto mul = gen_g->PushBack({gen_g->NewOpInst(MUL), attention_out, exp_lse});

  // Then aggregate all segments with AllReduce.
  auto reduce_op = gen_g->PushBack({gen_g->NewOpInst(ALL_REDUCE, attrs), mul});

  *output = gen_g->PushBack({NewValueNode(prim::kPrimMakeTuple), reduce_op, exp_lse});
}

Status FusedInferAttentionScoreInfo::ComputeReplaceGraphForSplitKVSeq(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << "GenerateGraph Init failed";
    return FAILED;
  }

  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  int64_t seq_dim = SizeToLong(dev_matrix_shape_.size()) - dev_matrix_s1_dim_ - 1;
  if (dev_matrix.GetDevicesAlongDim(seq_dim, &group_devices) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " get group devices along dim " << seq_dim << " failed.";
    return FAILED;
  }
  Group group;
  if (g_device_manager->CreateGroup(group_devices, &group) != SUCCESS) {
    MS_LOG(ERROR) << "Create communication group for " << group_devices << " failed";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": The rank is " << g_device_manager->rank_index_in_stage();

  AnfNodePtr fused_attention_score, output_maketuple;
  SplitKVSequenceGraph(group, &gen_g, &fused_attention_score, &output_maketuple);

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {
    std::make_pair(fused_attention_score, kIndex1),  std::make_pair(fused_attention_score, kIndex2),
    std::make_pair(fused_attention_score, kIndex3),  std::make_pair(fused_attention_score, kIndex4),
    std::make_pair(fused_attention_score, kIndex5),  std::make_pair(fused_attention_score, kIndex6),
    std::make_pair(fused_attention_score, kIndex7),  std::make_pair(fused_attention_score, kIndex8),
    std::make_pair(fused_attention_score, kIndex9),  std::make_pair(fused_attention_score, kIndex10),
    std::make_pair(fused_attention_score, kIndex11), std::make_pair(fused_attention_score, kIndex12),
    std::make_pair(fused_attention_score, kIndex13), std::make_pair(fused_attention_score, kIndex14),
    std::make_pair(fused_attention_score, kIndex15), std::make_pair(fused_attention_score, kIndex16),
    std::make_pair(fused_attention_score, kIndex17), std::make_pair(fused_attention_score, kIndex18),
    std::make_pair(fused_attention_score, kIndex19), std::make_pair(fused_attention_score, kIndex20),
    std::make_pair(fused_attention_score, kIndex21), std::make_pair(fused_attention_score, kIndex22),
    std::make_pair(fused_attention_score, kIndex23), std::make_pair(fused_attention_score, kIndex24),
    std::make_pair(fused_attention_score, kIndex25), std::make_pair(fused_attention_score, kIndex26),
    std::make_pair(fused_attention_score, kIndex27), std::make_pair(fused_attention_score, kIndex28)};

  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, output_maketuple));

  return SUCCESS;
}

ReplaceGraphPtr FusedInferAttentionScoreInfo::replace_graph(const CNodePtr &cnode) {
  if (softmax_lse_flag_ && sp_ != 1 && !IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
    if (ComputeReplaceGraphForSplitKVSeq(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": FusedInferFlashAttentionScore sequence parallel get replace graph failed";
    }
  }
  return replace_graph_;
}

REGISTER(FusedInferAttentionScoreInfo);
}  // namespace parallel
}  // namespace mindspore
