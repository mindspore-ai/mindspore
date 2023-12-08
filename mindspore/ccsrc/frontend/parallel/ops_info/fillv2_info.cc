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
#include "frontend/parallel/ops_info/fillv2_info.h"

#include <functional>
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status FillV2Info::InferAttrs() {
  if (infer_attrs_completed_) {
    return SUCCESS;
  }
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GetAttrs failed.";
    return FAILED;
  }
  ResetInputsShape();
  infer_attrs_completed_ = true;
  fake_inputs_shape_ = inputs_shape_;
  MS_LOG(INFO) << name_ << ": The origin shape is " << inputs_shape_;
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (inputs_shape_[0][i] == -1) {  // if dynamic shape, replace -1 to 1, this dimension can not be split
      fake_inputs_shape_[0][i] = 1;
      is_dynamic_shape_ = true;
    }
  }

  if (is_dynamic_shape_) {
    MS_LOG(INFO) << name_ << ": the fake shape is " << fake_inputs_shape_;
  }

  return SUCCESS;
}

Status FillV2Info::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, fake_inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy " << strategy->ToString();
    return FAILED;
  }
  return SUCCESS;
}

Status FillV2Info::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  auto strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << name_ << ": Infer device matric failed, inputs_startegy is empty.";
    return FAILED;
  }
  dev_matrix_shape_ = strategies.at(0);
  return SUCCESS;
}

Status FillV2Info::InferTensorMap() {
  TensorMap tensor_map;
  std::vector<Dimensions> strategies = strategy_->GetInputDim();
  auto input_shape_strategy = strategies.at(0);
  auto size = input_shape_strategy.size();
  for (size_t i = 0; i < size; ++i) {
    tensor_map.push_back(SizeToLong(size - i - 1));
  }
  inputs_tensor_map_.push_back(tensor_map);
  (void)inputs_tensor_map_.emplace_back(TensorMap());
  outputs_tensor_map_.push_back(tensor_map);
  return SUCCESS;
}

std::vector<StrategyPtr> FillV2Info::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(fake_inputs_shape_.at(0).size(), 1);
  Shape input1_split;
  Shapes splittable_inputs = {input0_split, input1_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, fake_inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy.";
  }
  return sp_vector;
}

void FillV2Info::ReplaceDynamicInput(const CNodePtr &cnode, const Shape &strategy) {
  auto dynamic_node = cnode->input(kIndex1);
  if (!IsPrimitiveCNode(dynamic_node, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << name_ << "The dynamic input must be MakeTuple cnode, but got "
                      << dynamic_node->fullname_with_scope();
    return;
  }

  auto make_tuple_cnode = dynamic_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple_cnode);

  for (size_t i = 1; i < make_tuple_cnode->inputs().size(); ++i) {
    if (strategy[i - 1] <= 1) {
      continue;
    }

    auto input_node = make_tuple_cnode->input(i);
    MS_EXCEPTION_IF_NULL(input_node);
    auto value_node = GetValueNode(input_node);
    if (value_node != nullptr && value_node->isa<Int64Imm>()) {
      auto origin_ele = GetValue<int64_t>(value_node);
      if (origin_ele % strategy[i - 1] != 0) {
        MS_LOG(EXCEPTION) << name_ << ": the origin shape is " << origin_ele << ", can not be div by shard size "
                          << strategy[i - 1];
      }
      int64_t replace_shape = origin_ele / strategy[i - 1];
      MS_LOG(INFO) << name_ << ": replace shape from " << origin_ele << " to " << replace_shape << ", the index is "
                   << (i - 1);
      auto replace_value_ptr = MakeValue(replace_shape);
      auto replace_value_node = std::make_shared<ValueNode>(replace_value_ptr);
      make_tuple_cnode->set_input(i, replace_value_node);
    }
  }
}

void FillV2Info::ReplaceNodeInputOrAttrs() {
  Shape strategy = strategy_->GetInputDim()[0];
  if (std::accumulate(strategy.cbegin(), strategy.cend(), 1, std::multiplies<int64_t>()) == 1) {
    return;
  }

  for (auto &cnode : cnodes_) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (!is_dynamic_shape_) {  // static shape
      auto input_shape = inputs_shape_.at(kIndex0);
      for (size_t i = 0; i < strategy.size(); i++) {
        input_shape[i] /= strategy[i];
      }
      auto func_graph = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      auto manager = func_graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      auto val_tensor_node = NewValueNode(MakeValue(std::make_shared<tensor::Tensor>(input_shape)));
      MS_LOG(INFO) << name_ << ": the new shape is " << input_shape;
      cnode->set_input(kIndex1, val_tensor_node);
    } else {  // dynamic shape
      ReplaceDynamicInput(cnode, strategy);
    }
  }
}

Status FillV2Info::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }
  if (mirror_ops_.size() == kSizeOne) {
    // Insert empty mirror op for shape
    (void)mirror_ops_.insert(mirror_ops_.begin(), OperatorVector());
  }
  return SUCCESS;
}

Shape FillV2Info::GetShapeFromTensor(const tensor::TensorPtr &shape_tensor) {
  MS_EXCEPTION_IF_NULL(shape_tensor);
  auto dim = shape_tensor->DataDim();
  if (IntToSize(dim) != kDim1) {
    MS_LOG(EXCEPTION) << name_ << ": The rank of 'input_shape' must be 1, but got rank " << dim;
  }
  auto size = shape_tensor->DataSize();
  if (size <= 0) {
    MS_LOG(EXCEPTION) << name_ << ": The size of 'input_shape' must be greater than 0, but got size " << size;
  }
  auto dtype = shape_tensor->data_type();
  auto data = shape_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data);
  if (dtype == kNumberTypeInt32) {
    auto shape_data = static_cast<int32_t *>(data);
    Shape shape(shape_data, shape_data + size);
    return shape;
  } else if (dtype == kNumberTypeInt64) {
    auto shape_data = static_cast<int64_t *>(data);
    Shape shape(shape_data, shape_data + size);
    return shape;
  }
  MS_LOG(EXCEPTION) << name_ << ": The dtype of 'input_shape' must be int32 or int64, but got type "
                    << TypeIdToString(dtype);
}

void FillV2Info::ResetInputsShape() {
  auto input_value_shape = input_value_[0];
  if (input_value_shape == nullptr) {
    MS_LOG(EXCEPTION) << name_ << ": The value of input 'shape' must be a constant. "
                      << "If you pass this value via construct, try to define its value in __init__";
  }
  MS_EXCEPTION_IF_NULL(input_value_shape);
  if (input_value_shape->isa<tensor::Tensor>()) {
    auto tensor_shape_ptr = GetValue<tensor::TensorPtr>(input_value_shape);
    auto shape = GetShapeFromTensor(tensor_shape_ptr);
    inputs_shape_[0] = shape;
    is_parameter_[0] = false;
    return;
  } else if (input_value_shape->isa<ValueTuple>()) {
    (void)inputs_shape_.insert(inputs_shape_.begin(), GetValue<Shape>(input_value_shape));
    (void)is_parameter_.insert(is_parameter_.begin(), false);
    return;
  }
  MS_LOG(EXCEPTION) << name_ << ": The type of input 'shape' must be Tensor or Tuple, but got "
                    << input_value_shape->type()->ToString();
}

REGISTER(FillV2Info);
}  // namespace parallel
}  // namespace mindspore
