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
#include <map>

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
constexpr size_t kRank4 = 4;
constexpr size_t kSparseMode0 = 0;
constexpr size_t kDpAxis = 2;
constexpr size_t kAttenSeqPosRank4 = 2;
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

bool PromptFlashAttentionInfo::CheckStrategyOnIndex(int64_t strategy, int64_t true_value, const std::string &dim_name,
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
    atten_mask_tensor_map[0] = kDpAxis;
    atten_mask_strategy_map[0] = kDpAxis;
  }
  if (padding_mask_rank_ >= kRank3) {
    padding_mask_tensor_map[0] = kDpAxis;
    padding_mask_strategy_map[0] = kDpAxis;
  }
  optinal_tensor_map_[ops::kPromptFlashAttentionInputAttnMaskIndex] = atten_mask_tensor_map;
  optinal_tensor_map_[ops::kPromptFlashAttentionInputPaddingMaskIndex] = padding_mask_tensor_map;
  optinal_tensor_map_[ops::kPromptFlashAttentionInputActualSeqLengthsIndex] = {kDpAxis};
  optinal_tensor_map_[ops::kPromptFlashAttentionInputActualSeqLengthsKvIndex] = {kDpAxis};

  optinal_op_strategies_[ops::kPromptFlashAttentionInputAttnMaskIndex] = atten_mask_strategy_map;
  optinal_op_strategies_[ops::kPromptFlashAttentionInputPaddingMaskIndex] = padding_mask_strategy_map;
  optinal_op_strategies_[ops::kPromptFlashAttentionInputActualSeqLengthsIndex] = {kDpAxis};
  optinal_op_strategies_[ops::kPromptFlashAttentionInputActualSeqLengthsKvIndex] = {kDpAxis};
}

Status PromptFlashAttentionInfo::GetAttrs() {
  head_num_ = GetIntAttr(kAttrHeadNum);
  kv_head_num_ = GetIntAttr(kAttrKVHeadNum);
  input_layout_ = GetStringAttr(kAttrInputLayout);
  sparse_mode_ = GetIntAttr(kAttrSparseMode);
  pre_tokens_ = GetIntAttr(kAttrPreTokens);
  next_tokens_ = GetIntAttr(kAttrNextTokens);
  is_attn_mask_compressed_ =
    std::find(needCompressAttnMask.begin(), needCompressAttnMask.end(), sparse_mode_) != needCompressAttnMask.end();
  need_update_op_attrs_mode_ = sparse_mode_ != kSparseAllMask;
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
  auto attn_strategy = strategies[input_index];
  attn_sp_shard_ = !is_attn_mask_compressed_;
  int64_t attn_seq_dim;
  if (atten_mask_rank_ == kRank2) {
    attn_seq_dim = 0;
  } else if (atten_mask_rank_ == kRank3) {
    attn_seq_dim = 1;
  } else {
    attn_seq_dim = kAttenSeqPosRank4;
  }
  int64_t attn_sp_dim = attn_sp_shard_ ? sp_ : 1;
  if (!CheckStrategyOnIndex(attn_strategy[attn_seq_dim], attn_sp_dim, "S-Dimention", "attn_mask")) {
    return FAILED;
  }
  return SUCCESS;
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

int64_t PromptFlashAttentionInfo::GetSplitIdAndRank() {
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

std::tuple<int64_t, int64_t> PromptFlashAttentionInfo::GetAttenionMaskAttrs(const int64_t split_id,
                                                                            const int64_t split_num) {
  int64_t kv_seq_length;
  int64_t q_seq_length;
  if (input_layout_ == "BSH") {
    kv_seq_length = inputs_shape_[ops::kPromptFlashAttentionInputKeyIndex][kInputQuerySeqDimBSH];
    q_seq_length = inputs_shape_[ops::kPromptFlashAttentionInputQueryIndex][kInputQuerySeqDimBSH];
  } else {
    kv_seq_length = inputs_shape_[ops::kPromptFlashAttentionInputKeyIndex][kInputQuerySeqDimBNSD];
    q_seq_length = inputs_shape_[ops::kPromptFlashAttentionInputQueryIndex][kInputQuerySeqDimBNSD];
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

Status PromptFlashAttentionInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto strategies = strategy->GetInputDim();
  auto query_strategy = strategies[ops::kPromptFlashAttentionInputQueryIndex];
  auto key_strategy = strategies[ops::kPromptFlashAttentionInputKeyIndex];
  auto value_strategy = strategies[ops::kPromptFlashAttentionInputValueIndex];

  if (key_strategy != value_strategy) {
    MS_LOG(ERROR) << "For " << name_ << " : The in_strategy among 'key' and 'value' must be same.";
    return FAILED;
  }

  if (input_layout_ == "BSH") {
    if (head_num_ % query_strategy[kInputQueryHiddenDimBSH] != 0) {
      MS_LOG(ERROR) << "For " << name_ << ": head_num % query_strategy[2] must be 0, but got " << head_num_
                    << "(head_num) and " << query_strategy[kInputQueryHiddenDimBSH] << "(query_strategy[2])";
      return FAILED;
    }
    dp_ = query_strategy[kInputBatchDim];
    mp_ = query_strategy[kInputQueryHiddenDimBSH];
    sp_ = query_strategy[kInputQuerySeqDimBSH];
  } else if (input_layout_ == "BNSD") {
    if (!CheckStrategyOnIndex(query_strategy[kInputQueryHiddenDimBNSD], 1, "D-Dimention", "query")) {
      return FAILED;
    }
    dp_ = query_strategy[kInputBatchDim];
    mp_ = query_strategy[kInputQueryNDimBNSD];
    sp_ = query_strategy[kInputQuerySeqDimBNSD];
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }
  if (optinal_inputs_.empty()) {
    SetOptinalInputs();
  }

  if (padding_mask_rank_ == kRank2) {
    if (!CheckStrategyOnIndex(strategies[ops::kPromptFlashAttentionInputPaddingMaskIndex][kInputBatchDim], 1,
                              "B-Dimention", "padding_mask")) {
      return FAILED;
    }
  }
  if (atten_mask_rank_ >= kRank2) {
    if (CheckAttenMaskStrategy(strategy, ops::kPromptFlashAttentionInputAttnMaskIndex) != SUCCESS) {
      MS_LOG(ERROR) << "Check strategy for atten mask failed";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status PromptFlashAttentionInfo::InferDevMatrixShape() {
  if (input_layout_ == "BSH") {
    dev_matrix_shape_ = {dp_, sp_, mp_};
    dev_matrix_batch_dim_ = kDpAxis;
    dev_matrix_s1_dim_ = 1;
    dev_matrix_n1_dim_ = 0;
  } else if (input_layout_ == "BNSD") {
    dev_matrix_shape_ = {dp_, mp_, sp_};
    dev_matrix_batch_dim_ = kDpAxis;
    dev_matrix_s1_dim_ = 0;
    dev_matrix_n1_dim_ = 1;
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }
  return SUCCESS;
}

Status PromptFlashAttentionInfo::InferTensorMap() {
  if (optinal_inputs_.empty()) {
    SetOptinalInputs();
  }
  if (input_layout_ == "BSH") {
    (void)inputs_tensor_map_.emplace_back(
      Shape{dev_matrix_batch_dim_, dev_matrix_s1_dim_, dev_matrix_n1_dim_});                      // query
    (void)inputs_tensor_map_.emplace_back(Shape{dev_matrix_batch_dim_, -1, dev_matrix_n1_dim_});  // key
    (void)inputs_tensor_map_.emplace_back(Shape{dev_matrix_batch_dim_, -1, dev_matrix_n1_dim_});  // value
    outputs_tensor_map_.push_back(
      Shape{dev_matrix_batch_dim_, dev_matrix_s1_dim_, dev_matrix_n1_dim_});  // attention_out
  } else if (input_layout_ == "BNSD") {
    (void)inputs_tensor_map_.emplace_back(
      Shape{dev_matrix_batch_dim_, dev_matrix_n1_dim_, dev_matrix_s1_dim_, -1});                      // query
    (void)inputs_tensor_map_.emplace_back(Shape{dev_matrix_batch_dim_, dev_matrix_n1_dim_, -1, -1});  // key
    (void)inputs_tensor_map_.emplace_back(Shape{dev_matrix_batch_dim_, dev_matrix_n1_dim_, -1, -1});  // value
    outputs_tensor_map_.push_back(
      Shape{dev_matrix_batch_dim_, dev_matrix_n1_dim_, dev_matrix_s1_dim_, -1});  // attention_out
  } else {
    MS_LOG(ERROR) << "For" << name_ << ": The input layout" << input_layout_ << "is not supported.";
    return FAILED;
  }

  if (atten_mask_rank_ == kRank2) {
    optinal_tensor_map_[ops::kPromptFlashAttentionInputAttnMaskIndex][0] = dev_matrix_s1_dim_;
  } else if (atten_mask_rank_ == kRank3) {
    optinal_tensor_map_[ops::kPromptFlashAttentionInputAttnMaskIndex][1] = dev_matrix_s1_dim_;
  } else if (atten_mask_rank_ == kRank4) {
    optinal_tensor_map_[ops::kPromptFlashAttentionInputAttnMaskIndex][kAttenSeqPosRank4] = dev_matrix_s1_dim_;
  } else {
    MS_LOG(INFO) << "Attention mask rank is not in [2, 3, 4]， rank is " << atten_mask_rank_;
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
    if (sp_ > 1 && need_update_op_attrs_mode_) {
      int64_t split_id = GetSplitIdAndRank();
      int64_t new_pre_tokens, new_next_tokens;
      std::tie(new_pre_tokens, new_next_tokens) = GetAttenionMaskAttrs(split_id, sp_);
      int64_t new_sparse_mode = is_attn_mask_compressed_ ? kSparseBand : sparse_mode_;
      clone_prim->set_attr(kAttrSparseMode, MakeValue(new_sparse_mode));
      clone_prim->set_attr(kAttrPreTokens, MakeValue(new_pre_tokens));
      clone_prim->set_attr(kAttrNextTokens, MakeValue(new_next_tokens));
    }
    cnode->set_input(0, NewValueNode(clone_prim)->cast<AnfNodePtr>());
  }
}

REGISTER(PromptFlashAttentionInfo);
}  // namespace parallel
}  // namespace mindspore
