/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/gamma_info.h"

#include <map>
#include <utility>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/auto_parallel/edge_costmodel.h"

namespace mindspore {
namespace parallel {
Status GammaInfo::InferAttrs() {
  if (infer_attrs_completed_) {
    return SUCCESS;
  }

  if (GetAttrs() != SUCCESS) {
    return FAILED;
  }
  ResetInputsShape();
  infer_attrs_completed_ = true;
  return SUCCESS;
}

Status GammaInfo::GetAttrs() {
  seed_ = GetIntAttr(SEED);
  if (seed_ < 0) {
    MS_LOG(ERROR) << name_ << ": Seed must be greater or equal to zero, bug got " << seed_;
    return FAILED;
  }
  seed2_ = GetIntAttr(SEED2);
  if (seed2_ < 0) {
    MS_LOG(ERROR) << name_ << ": Seed2 must be greater or equal to zero, bug got " << seed2_;
    return FAILED;
  }
  infer_strategy_mode_ = INDEPENDENT_MODE;
  return SUCCESS;
}

Status GammaInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }
  auto strategies = strategy->GetInputDim();
  auto alpha_strategy = strategies.at(1);
  auto beta_strategy = strategies.at(2);
  if (!IsNotSplittableStrategy(alpha_strategy) || !IsNotSplittableStrategy(beta_strategy)) {
    MS_LOG(ERROR) << name_ << ": Cannot shard the input `alpha` and `beta`, but got strategy "
                  << StrategyToString(strategies);
    return FAILED;
  }
  return SUCCESS;
}

Status GammaInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();
  auto strategies = strategy_->GetInputDim();
  auto shape_strategy = strategies.at(0);
  dev_matrix_shape_ = shape_strategy;

  MS_LOG(INFO) << name_ << ": The dev matrix is " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status GammaInfo::InferTensorMap() {
  auto strategies = strategy_->GetInputDim();
  auto shape_strategy = strategies.at(0);
  size_t size = shape_strategy.size();
  TensorMap shape_tensor_map;
  for (size_t i = 0; i < size; ++i) {
    shape_tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(shape_tensor_map);
  (void)inputs_tensor_map_.emplace_back(TensorMap(strategies.at(1).size(), -1));
  (void)inputs_tensor_map_.emplace_back(TensorMap(strategies.at(2).size(), -1));
  (void)outputs_tensor_map_.emplace_back(shape_tensor_map);
  return SUCCESS;
}

std::vector<StrategyPtr> GammaInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shape input1_split(inputs_shape_[1].size(), 0);
  Shape input2_split(inputs_shape_[2].size(), 0);
  Shapes splittable_inputs = {input0_split, input1_split, input2_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy.";
  }
  return sp_vector;
}

void GammaInfo::UpdateShape(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = cnode->input(1)->cast<ValueNodePtr>();
  std::vector<int64_t> input_shape = GetValue<std::vector<int64_t>>(input_node->value());
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  for (size_t i = 0; i < stra[0].size(); i++) {
    input_shape[i] /= stra[0][i];
  }
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ValuePtr new_shape = MakeValue(input_shape);
  AnfNodePtr val = NewValueNode(new_shape);
  cnode->set_input(kIndex1, val);
}

void GammaInfo::ReplaceNodeInputOrAttrs() {
  // Replace input 'shape' to slice shape
  auto cnode = cnode_;
  UpdateShape(cnode);

  // Update seed according rank_id
  int64_t rank_id = g_device_manager->rank_index_in_stage();
  int64_t seed_bias;
  if (repeated_num_in_dev_matrix_right_) {
    seed_bias = rank_id / repeated_calc_num_;
  } else {
    int64_t device_num = stage_device_size_;
    seed_bias = rank_id % (device_num / repeated_calc_num_);
  }

  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  prim->set_attr(SEED, MakeValue(seed_ + seed_bias));
  prim->set_attr(SEED2, MakeValue(seed2_ + seed_bias));
}

void GammaInfo::ResetInputsShape() {
  if (inputs_shape_.size() == input_value_.size()) {
    return;
  }

  ValueTuplePtr shape_value = input_value_[0]->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(shape_value);
  (void)inputs_shape_.insert(inputs_shape_.cbegin(), GetValue<Shape>(shape_value));
}

bool GammaInfo::IsNotSplittableStrategy(const Dimensions &strategy) const {
  return std::all_of(strategy.cbegin(), strategy.cend(), [](int64_t val) { return val == 1; });
}

void GammaInfo::ReComputeBatchSplitFlagList() {
  ResetInputsShape();
  split_flag_list_.clear();
  split_flag_list_ = std::vector<bool>(inputs_shape_.size(), false);
  if (!split_flag_list_.empty()) {
    split_flag_list_[0] = true;
  }
}

int64_t GammaInfo::ComputeOpAndPrevEdgeParameterInvolved() {
  if (is_output_parameter_involve_ != -1) {
    return is_output_parameter_involve_;
  }
  is_parameter_involve_ = is_parameter_;
  const auto &prev_edges = this->GetAlivePrevEdges();
  for (auto &p_edge : prev_edges) {
    auto input_index = p_edge->next_op_input_index();
    auto prev_op_para = p_edge->prev_operator()->ComputeOpAndPrevEdgeParameterInvolved();
    if (input_index - 1 >= is_parameter_involve_.size()) {
      MS_LOG(EXCEPTION) << name_ << " has input length: " << is_parameter_involve_.size()
                        << ", but got wrong input_index: " << input_index;
    }
    if (prev_op_para == 0) {
      is_parameter_involve_[input_index - 1] = false;
    } else if (prev_op_para == 1) {
      is_parameter_involve_[input_index - 1] = true;
    } else {
      MS_LOG(EXCEPTION) << name_ << " got wrong value: " << prev_op_para << ", input_index: " << input_index;
    }
    p_edge->set_parameter_involve(prev_op_para);
  }
  if (std::any_of(is_parameter_involve_.begin(), is_parameter_involve_.end(), [](bool value) { return value; })) {
    // If anyone of the input is a parameter_involved, the output is parameter_involved.
    is_output_parameter_involve_ = 1;
  } else {
    is_output_parameter_involve_ = 0;
  }
  // Set 'is_parameter_involve_' and 'is_output_parameter_involve_' into operatorCost, which are used in
  // calculating 'inputs_in_memory' and 'output_in_memory', respectively.
  operator_cost()->set_is_parameter_involve(is_parameter_involve_);
  operator_cost()->set_output_parameter_involve(is_output_parameter_involve_);
  // Calculating 'output_in_memory'
  operator_cost()->CalculateOutputInMemory();
  // Calculating 'inputs_in_memory'
  std::map<size_t, bool> input_in_memory;
  for (auto &p_edge : prev_edges) {
    auto input_index = p_edge->next_op_input_index();
    auto is_in_mem = p_edge->prev_operator()->operator_cost()->is_output_in_memory();
    (void)input_in_memory.emplace(std::make_pair(input_index, is_in_mem));
  }
  operator_cost()->CalculateInputsInMemory(input_in_memory);

  return is_output_parameter_involve_;
}

REGISTER(GammaInfo);
}  // namespace parallel
}  // namespace mindspore
