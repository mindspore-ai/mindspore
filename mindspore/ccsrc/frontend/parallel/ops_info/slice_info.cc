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

#include "frontend/parallel/ops_info/slice_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status SliceInfo::GetInput(const ValuePtr &input_value, std::vector<int64_t> *input) {
  input->clear();
  MS_EXCEPTION_IF_NULL(input_value);
  ValueTuplePtr value_tuple = input_value->cast<ValueTuplePtr>();
  if (value_tuple == nullptr) {
    MS_LOG(ERROR) << name_ << ": Input value must be ValueTuplePtr.";
    return FAILED;
  }

  for (auto &element : value_tuple->value()) {
    MS_EXCEPTION_IF_NULL(element);
    if (element->isa<Int64Imm>()) {
      int64_t value = element->cast<Int64ImmPtr>()->value();
      input->push_back(value);
    } else {
      MS_LOG(ERROR) << name_ << ": The value must be int64";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status SliceInfo::GetAttrs() {
  if (input_value_.size() != SLICE_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": The size of input value must be " << SLICE_INPUTS_SIZE << ", but got "
                  << input_value_.size();
    return FAILED;
  }

  if ((GetInput(input_value_[SLICE_BEGIN_INDEX], &begin_) != SUCCESS) ||
      (GetInput(input_value_[SLICE_SIZE_INDEX], &size_) != SUCCESS)) {
    return FAILED;
  }

  return SUCCESS;
}

Status SliceInfo::CheckStrategy(const StrategyPtr &strategy) {
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

  for (size_t i = 0; i < begin_.size(); ++i) {
    bool no_fully_fetch = ((begin_[i] != 0) || (size_[i] < inputs_shape_[0][i]));
    if (no_fully_fetch && (strategy_value[i] != 1)) {
      MS_LOG(ERROR) << name_ << ": When a dimension is not fully fetched, the dimension can not be split now";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status SliceInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status SliceInfo::InferTensorMap() {
  TensorMap tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  // cannot use dev_matrix_shape_ replace inputs_shape_[0], because it may not be fully split in all devices.
  int64_t size = SizeToInt(inputs_shape_[0].size());
  for (int i = 0; i < size; ++i) {
    tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(tensor_map);
  outputs_tensor_map_.push_back(tensor_map);
  return SUCCESS;
}

Status SliceInfo::InferMirrorOps() {
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

  OperatorVector input_op, begin_op, end_op;
  input_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(input_op);
  mirror_ops_.push_back(begin_op);
  mirror_ops_.push_back(end_op);
  return SUCCESS;
}

// Note: if the batch dimension is not fully fetched, the batch strategy may not work.
std::shared_ptr<Strategies> SliceInfo::GenerateBatchStrategies() {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << "generate batch parallel strategies failed.";
  }
  split_flag_list_ = {true};
  bool no_fully_fetch = ((begin_[0] != 0) || (size_[0] < inputs_shape_[0][0]));
  if (no_fully_fetch) {
    split_flag_list_ = {false};
  }
  return GenerateBatchStrategiesBySplitFlag(inputs_shape_, split_flag_list_);
}

Status SliceInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> SliceInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input_split(inputs_shape_[0].size(), 1);
  for (size_t i = 0; i < begin_.size(); ++i) {
    bool no_fully_fetch = ((begin_[i] != 0) || (size_[i] < inputs_shape_[0][i]));
    if (no_fully_fetch) {
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

ReplaceGraphPtr SliceInfo::replace_graph(const CNodePtr &cnode) {
  const auto &input_dim = strategy_->GetInputDim();
  auto input_strategy = input_dim.at(0);
  if (std::any_of(input_strategy.begin(), input_strategy.end(), [](const int64_t &shard) { return shard > 1; })) {
    if (ComputeReplaceGraph(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": InferReplaceOp failed.";
    }
  }
  return replace_graph_;
}

Status SliceInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  const auto input_dim = strategy_->GetInputDim();
  Dimensions input_stra = input_dim.at(0);

  std::vector<int64_t> sliced_size_shape_int;
  Shape input_slice_shape = inputs_tensor_info_[0].slice_shape();
  for (uint64_t i = 0; i < size_.size(); i++) {
    if (input_stra[i] == 1) {
      sliced_size_shape_int.push_back(size_[i]);
    } else {
      sliced_size_shape_int.push_back(input_slice_shape[i]);
    }
  }
  auto new_begin = CreateValueTupleAnfNodePtr(begin_);
  auto new_size = CreateValueTupleAnfNodePtr(sliced_size_shape_int);

  auto slice = gen_g.PushBack({gen_g.NewOpInst(SLICE), gen_g.virtual_input_node(), new_begin, new_size});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(slice, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, slice));

  return SUCCESS;
}

REGISTER(SliceInfo);
}  // namespace parallel
}  // namespace mindspore
