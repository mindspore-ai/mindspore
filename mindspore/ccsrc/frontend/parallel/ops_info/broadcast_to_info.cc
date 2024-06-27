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

#include "frontend/parallel/ops_info/broadcast_to_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "pipeline/jit/ps/resource.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
Status BroadcastToInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  is_stand_alone_ = false;
  Shapes stra = strategy->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": the strategy is empty";
    return FAILED;
  }
  int64_t shard_num = std::accumulate(stra[0].begin(), stra[0].end(), 1, std::multiplies<int64_t>());
  is_stand_alone_ = (shard_num == 1);
  if (is_stand_alone_) {
    return SUCCESS;
  }
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy " << stra;
    return FAILED;
  }

  return SUCCESS;
}

Status BroadcastToInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status BroadcastToInfo::InferTensorMap() {
  TensorMap in_tensor_map;
  TensorMap out_tensor_map;

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << "The inputs shape is empty";
    return FAILED;
  }

  int32_t size = SizeToInt(inputs_shape_[0].size());
  for (int i = 0; i < size; ++i) {
    in_tensor_map.push_back(size - i - 1);
  }
  inputs_tensor_map_.push_back(in_tensor_map);

  size_t len_diff = outputs_shape_.at(0).size() - inputs_shape_.at(0).size();
  for (size_t i = 0; i < len_diff; ++i) {
    out_tensor_map.push_back(MAP_NONE);
  }
  (void)std::copy(in_tensor_map.begin(), in_tensor_map.end(), std::back_inserter(out_tensor_map));
  outputs_tensor_map_.push_back(out_tensor_map);
  return SUCCESS;
}

Status BroadcastToInfo::InferMirrorOps() {
  mirror_ops_.clear();

  if (inputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": the inputs tensor map is empty";
    return FAILED;
  }

  std::vector<Group> group;
  if (CreateGroupByTensorMap(inputs_tensor_map_[0], &group) != SUCCESS) {
    ReportError(name_ + ": Create group failed");
    mirror_ops_.clear();
    return FAILED;
  }

  OperatorVector mirror_op;
  if (group.empty()) {
    return SUCCESS;
  }

  mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(mirror_op);

  return SUCCESS;
}

Status BroadcastToInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> BroadcastToInfo::GenerateOpStrategies(int64_t stage_id) {
  if (inputs_shape_.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": The inputs shape is empty";
  }
  Shape input_split;
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (inputs_shape_[0][i] == 1) {
      input_split.push_back(0);
    } else {
      input_split.push_back(1);
    }
  }

  // to generate the first input's strategy
  Shapes splittable_input = {input_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  // the others strategies are equal to the first input's strategy
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }
    Strategies tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    for (size_t i = 0; i < inputs_shape_.size(); ++i) {
      tmp_strategy.push_back(first_input_strategy);
    }
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

Status BroadcastToInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }

  if (!is_dynamic_shape_) {  // static shape
    Shape to_shape = outputs_tensor_info_[0].slice_shape();
    auto new_broadcast_to =
      gen_g.PushBack({gen_g.NewOpInst(BROADCAST_TO), gen_g.virtual_input_node(), CreateTuple(to_shape)});
    std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(new_broadcast_to, 1)};
    replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
      std::make_pair(input_nodes, new_broadcast_to));
    return SUCCESS;
  }

  // dynamic shape
  if (!IsPrimitiveCNode(cnode->input(DST_SHAPE_INDEX), prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << name_ << ": the dst shape is not make tuple";
  }

  ChangeMakeTupleConstant(cnode, DST_SHAPE_INDEX);
  return SUCCESS;
}

ReplaceGraphPtr BroadcastToInfo::replace_graph(const CNodePtr &cnode) {
  if (is_stand_alone_) {
    return nullptr;
  }
  if (ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
  }
  return replace_graph_;
}

REGISTER(BroadcastToInfo);
}  // namespace parallel
}  // namespace mindspore
