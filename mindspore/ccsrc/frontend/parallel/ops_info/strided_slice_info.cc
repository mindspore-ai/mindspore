/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/strided_slice_info.h"

#include <bitset>
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status StridedSliceInfo::GetMask(const std::string &mask_name, int64_t *mask_value) {
  if (mask_value == nullptr) {
    return FAILED;
  }
  auto mask_iter = attrs_.find(mask_name);
  if (mask_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(mask_iter->second);
    if (mask_iter->second->isa<Int64Imm>()) {
      *mask_value = mask_iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of " << mask_name << " is not int64_t";
      return FAILED;
    }
  }

  if (*mask_value != 0) {
    MS_LOG(INFO) << "For StridedSlice op attr:" << mask_name << ", the value is " << *mask_value;
  }

  return SUCCESS;
}

constexpr auto kStridedSliceMaxDims = 8;
static std::vector<bool> Dec2Bin(int64_t mask) {
  auto mask_str = std::bitset<kStridedSliceMaxDims>(mask).to_string();
  int64_t dim_idx = 0;
  std::vector<bool> result(kStridedSliceMaxDims, false);
  for (int64_t i = mask_str.size() - 1; i >= 0; --i) {
    if (mask_str[i] == '1') {
      result[dim_idx] = true;
    }
    dim_idx++;
  }
  return result;
}

void StridedSliceInfo::ComputeBeginMask(int64_t begin_mask_) {
  auto begin_mask = Dec2Bin(begin_mask_);
  for (size_t i = 0; i < begin_mask.size(); ++i) {
    if (i < kStridedSliceMaxDims && begin_mask[i]) {
      begin_[i] = strides_[i] < 0 ? SizeToLong(inputs_shape_[0][i]) - 1 : 0;
    }
  }
}

void StridedSliceInfo::ComputeEndMask(int64_t end_mask_) {
  auto end_mask = Dec2Bin(end_mask_);
  for (size_t j = 0; j < end_mask.size(); ++j) {
    if (j < kStridedSliceMaxDims && end_mask[j]) {
      end_[j] = strides_[j] < 0 ? -1 : SizeToLong(inputs_shape_[0][j]);
    }
  }
}

void StridedSliceInfo::ComputeEllipsisMask(int64_t ellipsis_mask_) {
  auto ellipsis_mask = Dec2Bin(ellipsis_mask_);
  for (size_t k = 0; k < ellipsis_mask.size(); ++k) {
    if (k < kStridedSliceMaxDims && ellipsis_mask[k]) {
      begin_[k] = 0;
      end_[k] = SizeToLong(inputs_shape_[0][k]);
      strides_[k] = 1;
    }
  }
}

void StridedSliceInfo::ComputeNewAxisMask(int64_t new_axis_mask_) {
  auto new_axis_mask = Dec2Bin(new_axis_mask_);
  for (size_t l = 0; l < new_axis_mask.size(); ++l) {
    if (l < kStridedSliceMaxDims && new_axis_mask[l]) {
      begin_[l] = 0;
      end_[l] = SizeToLong(inputs_shape_[0][l]);
      strides_[l] = 1;
    }
  }
}

void StridedSliceInfo::ComputShrinkAxisMask(int64_t shrink_axis_mask_) {
  auto shrink_axis_mask = Dec2Bin(shrink_axis_mask_);
  for (size_t m = 0; m < shrink_axis_mask.size(); ++m) {
    if (m < kStridedSliceMaxDims && shrink_axis_mask[m]) {
      end_[m] = end_[m] > begin_[m] ? begin_[m] + 1 : begin_[m] - 1;
      strides_[m] = end_[m] > begin_[m] ? 1 : -1;
    }
  }
}

Status StridedSliceInfo::GetAttrs() {
  if (attrs_.size() < STRIDED_SLICE_ATTRS_SIZE) {
    MS_LOG(ERROR) << name_ << ": The size of attrs small than " << STRIDED_SLICE_ATTRS_SIZE;
    return FAILED;
  }

  if ((GetMask(BEGIN_MASK, &begin_mask_) != SUCCESS) || (GetMask(END_MASK, &end_mask_) != SUCCESS) ||
      (GetMask(ELLIPSIS_MASK, &ellipsis_mask_) != SUCCESS) || (GetMask(NEW_AXIS_MASK, &new_axis_mask_) != SUCCESS) ||
      (GetMask(SHRINK_AXIS_MASK, &shrink_axis_mask_) != SUCCESS)) {
    return FAILED;
  }

  if (input_value_.size() != STRIDED_SLICE_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": The size of input value must be " << STRIDED_SLICE_INPUTS_SIZE << ", but got "
                  << input_value_.size();
    return FAILED;
  }

  if ((TransValueSequeueToVector(input_value_[STRIDED_SLICE_BEGIN_INDEX], &begin_) != SUCCESS) ||
      (TransValueSequeueToVector(input_value_[STRIDED_SLICE_END_INDEX], &end_) != SUCCESS) ||
      (TransValueSequeueToVector(input_value_[STRIDED_SLICE_STRIDES_INDEX], &strides_) != SUCCESS)) {
    return FAILED;
  }
  ComputeBeginMask(begin_mask_);
  ComputeEndMask(end_mask_);
  ComputeEllipsisMask(ellipsis_mask_);
  ComputeNewAxisMask(new_axis_mask_);
  ComputShrinkAxisMask(shrink_axis_mask_);
  return SUCCESS;
}

Status StridedSliceInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  Dimensions strategy_value = stra[0];
  if (strategy_value.size() < strides_.size()) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be larger or equal to the size of strides";
    return FAILED;
  }
  for (size_t i = 0; i < strides_.size(); ++i) {
    if ((strides_[i] != 1) && (strategy_value[i] > 1)) {
      MS_LOG(ERROR) << name_ << ": When a certain dimension is split, now does not support that the stride is not 1";
      return FAILED;
    }
  }

  if ((begin_.size() != end_.size()) || (begin_.size() != strides_.size())) {
    MS_LOG(ERROR) << name_ << ": The size of begin " << begin_.size() << ", end " << end_.size() << " and strides "
                  << strides_.size() << " must be equal";
    return FAILED;
  }

  for (size_t i = 0; i < begin_.size(); ++i) {
    bool no_fully_fetch = ((begin_[i] != 0) || (end_[i] < inputs_shape_[0][i]));
    if (no_fully_fetch && (strategy_value[i] != 1)) {
      MS_LOG(ERROR) << name_ << "When a dimension is not fully fetched, the dimension can not be split now";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status StridedSliceInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status StridedSliceInfo::InferTensorMap() {
  TensorMap tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << "The inputs shape is empty";
    return FAILED;
  }

  // cannot use dev_matrix_shape_ replace inputs_shape_[0], because it may not be fully split in all devices.
  int64_t size = SizeToLong(inputs_shape_[0].size());
  for (int64_t i = 0; i < size; ++i) {
    tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(tensor_map);
  if (new_axis_mask_ != 0) tensor_map.insert(tensor_map.begin() + (new_axis_mask_ - 1), -1);
  if (shrink_axis_mask_ != 0) tensor_map.erase(tensor_map.begin() + (shrink_axis_mask_ - 1));
  outputs_tensor_map_.push_back(tensor_map);
  return SUCCESS;
}

Status StridedSliceInfo::InferMirrorOps() {
  mirror_ops_.clear();
  if (inputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs tensor map is empty";
    return FAILED;
  }
  Shape input_tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(input_tensor_map, &group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }

  if (group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror group is empty.";
    return SUCCESS;
  }

  OperatorVector input_op, begin_op, end_op, strides_op;
  input_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(input_op);
  mirror_ops_.push_back(begin_op);
  mirror_ops_.push_back(end_op);
  mirror_ops_.push_back(strides_op);
  return SUCCESS;
}

// Note: if the batch dimension is not fully fetched, the batch strategy may not work.
std::shared_ptr<Strategys> StridedSliceInfo::GenerateBatchStrategies() {
  split_flag_list_ = {true};
  return GenerateBatchStrategiesBySplitFlag(inputs_shape_, split_flag_list_);
}

Status StridedSliceInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> StridedSliceInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input_split(inputs_shape_[0].size(), 1);
  for (size_t i = 0; i < begin_.size(); ++i) {
    bool no_fully_fetch = ((begin_[i] != 0) || (end_[i] < inputs_shape_[0][i]));
    if (no_fully_fetch || (strides_[i] != 1)) {
      input_split[i] = 0;
    }
  }
  Shapes splittable_inputs = {input_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": generate strategies failed";
  }

  return sp_vector;
}
}  // namespace parallel
}  // namespace mindspore
