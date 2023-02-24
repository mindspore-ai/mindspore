/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/uniform_real_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status UniformRealInfo::InferAttrs() {
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

Status UniformRealInfo::GetAttrs() {
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
  return SUCCESS;
}

Status UniformRealInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 1) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 1, but got " << stra.size();
    return FAILED;
  }
  return SUCCESS;
}

Status UniformRealInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status UniformRealInfo::InferTensorMap() {
  TensorMap input_tensor_map;
  TensorMap output_tensor_map;
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  size_t size = stra[0].size();

  for (size_t i = 0; i < size; i++) {
    input_tensor_map.push_back(SizeToLong(size - i - 1));
    output_tensor_map.push_back(SizeToLong(size - i - 1));
  }

  (void)inputs_tensor_map_.emplace_back(std::move(input_tensor_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_tensor_map));
  return SUCCESS;
}

Status UniformRealInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> UniformRealInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy.";
  }
  return sp_vector;
}

void UniformRealInfo::UpdateShape(const CNodePtr &cnode) {
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

void UniformRealInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    // Replace input 'shape' to slice shape
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
}

void UniformRealInfo::ResetInputsShape() {
  ValueTuplePtr shape_value = input_value_[0]->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(shape_value);
  inputs_shape_.push_back(GetValue<Shape>(shape_value));
  is_parameter_.push_back(false);
}

REGISTER(UniformRealInfo);
}  // namespace parallel
}  // namespace mindspore
