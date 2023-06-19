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

#include "frontend/parallel/ops_info/pad_info.h"
#include <utility>
#include <algorithm>

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status PadV3Info::GetAttrs() {
  mode_ = GetStringAttr(MODE);
  if (mode_ != CONSTANT) {
    MS_LOG(ERROR) << name_ << ": only support the constant mode, but the mode is " << mode_;
    return FAILED;
  }

  // If paddings_contiguous is true, paddings is arranged as [begin0, end0, begin1, end1, ...]
  // If paddings_contiguous is false, paddings is arranged as [begin0, begin1, ..., end1, end2, ...]
  bool paddings_contiguous = GetBoolAttr(PADDINGS_CONTIGUOUS);
  MS_LOG(INFO) << name_ << ": the paddings_contiguous is " << paddings_contiguous;

  if (input_value_.size() < PAD_V3_INPUT_VALUE_MIN_SIZE || input_value_[PADDINGS_INDEX] == nullptr) {
    MS_LOG(ERROR) << name_ << ": the input_value[1] is null";
    return FAILED;
  }

  std::vector<int64_t> paddings;
  if (input_value_[PADDINGS_INDEX]->isa<tensor::Tensor>()) {
    paddings = GetTensorValue(input_value_[PADDINGS_INDEX]);
  } else {
    paddings = GetValue<Shape>(input_value_[PADDINGS_INDEX]);
  }

  if (paddings.size() % PADDINGS_PAIR_SIZE != 0) {
    MS_LOG(ERROR) << name_ << ": the size of paddings must be the multiples of 2, but got " << paddings.size();
    return FAILED;
  }

  if (paddings.size() / PADDINGS_PAIR_SIZE > inputs_shape_[0].size()) {
    MS_LOG(ERROR) << name_
                  << ": the dim of input must be larger than or equal to padding dim num, but the shape of input is "
                  << inputs_shape_[0] << ", and the padding dim num is " << (paddings.size() / PADDINGS_PAIR_SIZE);
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": the paddings is " << paddings;

  // handle the paddings contiguous
  Shape con_paddings;  // con_paddings arrange the paddings as [begin0, end0, begin1, end1, ...]
  if (paddings_contiguous) {
    con_paddings = paddings;
  } else {
    for (size_t i = 0; i < (paddings.size() / PADDINGS_PAIR_SIZE); ++i) {
      con_paddings.push_back(paddings[i]);
      con_paddings.push_back(paddings[i + paddings.size() / PADDINGS_PAIR_SIZE]);
    }
  }
  MS_LOG(INFO) << name_ << ": the contiguous paddings is " << con_paddings;

  paddings_flag_ = std::vector<int64_t>(inputs_shape_[0].size(), 0);
  for (size_t i = inputs_shape_[0].size() - (con_paddings.size() / PADDINGS_PAIR_SIZE), j = 0;
       i < inputs_shape_[0].size(); ++i) {
    if (con_paddings[j] != 0 || con_paddings[j + 1] != 0) {
      paddings_flag_[i] = 1;
    }
    j *= PADDINGS_PAIR_SIZE;
  }

  MS_LOG(INFO) << name_ << ": the paddings flag is " << paddings_flag_;
  return SUCCESS;
}

Status PadV3Info::CheckStrategy(const StrategyPtr &strategy) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": the strategy is null";
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": the strategy is empty";
    return FAILED;
  }
  auto input_strategy = stra[0];

  // only check the strategy of first input
  if (CheckStrategyByVector({input_strategy}, {inputs_shape_[0]}) != SUCCESS) {
    return FAILED;
  }

  // if the paddings flag is 1, the dimension can not be split
  for (size_t i = 0; i < input_strategy.size(); ++i) {
    if (paddings_flag_[i] == 1 && input_strategy[i] != NO_SPLIT_STRATEGY) {
      MS_LOG(ERROR) << name_ << ": the padding dimension of input can not be split, the strategy of input is "
                    << input_strategy << ", and the paddings flag is " << paddings_flag_;
      return FAILED;
    }
  }
  return SUCCESS;
}

Status PadV3Info::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  Strategies strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << name_ << ": the strategy is empty";
    return FAILED;
  }
  dev_matrix_shape_ = strategies[0];

  MS_LOG(INFO) << name_ << ": dev matrix: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status PadV3Info::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Shape input_tensor_map;
  Strategies strategies = strategy_->GetInputDim();
  size_t dim = strategies[0].size();
  for (size_t i = 0; i < dim; ++i) {
    input_tensor_map.push_back(dim - i - 1);
  }

  inputs_tensor_map_.push_back(input_tensor_map);   // input
  outputs_tensor_map_.push_back(input_tensor_map);  // output
  return SUCCESS;
}

Status PadV3Info::InferMirrorOps() {
  mirror_ops_.clear();

  std::vector<Group> group;
  if (CreateGroupByTensorMap(inputs_tensor_map_[0], &group) != SUCCESS) {
    ReportError(name_ + ": Create group failed");
    mirror_ops_.clear();
    return FAILED;
  }

  OperatorVector mirror_op;
  if (group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror group is empty";
    return SUCCESS;
  }

  mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(mirror_op);
  return SUCCESS;
}

std::vector<StrategyPtr> PadV3Info::GenerateOpStrategies(int64_t stage_id) {
  Shape splittable_flag;
  (void)std::transform(paddings_flag_.begin(), paddings_flag_.end(), std::back_inserter(splittable_flag),
                       [](int64_t ele) { return 1 - ele; });
  Shapes splittable_input = {splittable_flag};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy";
  }

  return sp_vector;
}

void PadV3Info::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = true;

  for (size_t i = 1; i < split_flag_list_.size(); ++i) {
    split_flag_list_[i] = false;
  }

  auto paddings_dims = std::accumulate(paddings_flag_.begin(), paddings_flag_.end(), int64_t(0));
  if (SizeToLong(inputs_shape_[0].size()) == paddings_dims) {
    split_flag_list_[0] = false;
  }
}

REGISTER(PadV3Info);
}  // namespace parallel
}  // namespace mindspore
