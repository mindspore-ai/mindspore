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
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
Status BroadcastToInfo::GetAttrs() {
  out_shape_.clear();
  auto shape_iter = attrs_.find(SHAPE);
  if (shape_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(shape_iter->second);
    auto var = shape_iter->second->cast<ValueTuplePtr>();
    if (var == nullptr) {
      MS_LOG(ERROR) << name_ << ": shape format is wrong! Need ValueSequeue";
      return FAILED;
    }
    for (auto &ele : var->value()) {
      if (!ele->isa<Int64Imm>()) {
        MS_LOG(ERROR) << name_ << ": The element of shape must be int";
        return FAILED;
      }
      out_shape_.push_back(static_cast<int64_t>(GetValue<int64_t>(ele)));
    }
  } else {
    MS_LOG(ERROR) << name_ << ": Can not find the shape attr";
    return FAILED;
  }
  if (out_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": shape cannot be empty";
    return FAILED;
  }

  return SUCCESS;
}

Status BroadcastToInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  auto stra = strategy->GetInputDim().at(0);
  auto in_shape = inputs_shape_.at(0);
  for (size_t i = 0; i < stra.size(); ++i) {
    if ((in_shape[i] == 1) && (stra[i] != 1)) {
      MS_LOG(ERROR) << name_ << ": dimension with size 1 is not splitable.";
      return FAILED;
    }
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
  std::copy(in_tensor_map.begin(), in_tensor_map.end(), std::back_inserter(out_tensor_map));
  outputs_tensor_map_.push_back(out_tensor_map);
  return SUCCESS;
}

Status BroadcastToInfo::InferMirrorOps() {
  mirror_ops_.clear();
  if (inputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs tensor map is empty";
    return FAILED;
  }

  Shape input_tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(input_tensor_map, &group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group for input failed.";
    return FAILED;
  }

  if (group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror group is empty.";
    return SUCCESS;
  }

  OperatorVector input_op;
  input_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(input_op);

  return SUCCESS;
}

Status BroadcastToInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }

  TensorLayout input_layout, output_layout;
  // infer tensor layout
  if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], inputs_shape_[0]) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed.";
    return FAILED;
  }
  TensorInfo input_tensor_info(input_layout);
  inputs_tensor_info_.push_back(input_tensor_info);

  if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], outputs_shape_[0]) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed.";
    return FAILED;
  }
  TensorInfo output_tensor_info(output_layout);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status BroadcastToInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Status BroadcastToInfo::GenerateStrategies(int64_t stage_id) {
  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer attrs failed";
    return FAILED;
  }
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
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
    MS_LOG(ERROR) << name_ << ": Generate strategies failed";
    return FAILED;
  }

  // the others strategies are equal to the first input's strategy
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(ERROR) << name_ << ": The strategy is null or empty";
      return FAILED;
    }
    Strategys tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    for (size_t i = 0; i < inputs_shape_.size(); ++i) {
      tmp_strategy.push_back(first_input_strategy);
    }
    sp->ResetInputs(tmp_strategy);
  }

  size_t success = 0;
  for (auto &sp : sp_vector) {
    PrintStrategy(sp);
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

Status BroadcastToInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph();
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }

  Shape to_shape = outputs_tensor_info_[0].slice_shape();
  Attr attr_shape = std::make_pair(SHAPE, MakeValue(to_shape));
  OperatorAttrs attrs = {attr_shape};
  auto new_broadcast_to = gen_g.PushBack({gen_g.NewOpInst(BROADCAST_TO, attrs), gen_g.virtual_input_node()});
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(new_broadcast_to, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, new_broadcast_to));

  return SUCCESS;
}

ReplaceGraphPtr BroadcastToInfo::replace_graph(const CNodePtr &cnode) {
  if (ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
  }
  return replace_graph_;
}

Status BroadcastToInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status BroadcastToInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
