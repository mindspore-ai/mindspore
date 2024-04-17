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

#include "frontend/parallel/ops_info/tile_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <functional>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace parallel {
// get the multiples
Status TileInfo::GetAttrs() {
  if (input_value_.size() < 2) {
    MS_LOG(ERROR) << name_ << ": The size of input value is smaller than 2.";
    return FAILED;
  }
  if (input_value_[1] == nullptr) {
    MS_LOG(ERROR) << name_ << ": The dims is null.";
    return FAILED;
  }

  std::vector<ValuePtr> elements;
  ValueTuplePtr multiples = input_value_[1]->cast<ValueTuplePtr>();
  if (multiples == nullptr) {
    MS_LOG(ERROR) << name_ << ": Input_value_[1] must be ValueTuplePtr.";
    return FAILED;
  }
  elements = multiples->value();
  if (elements.size() != outputs_shape_[0].size()) {
    MS_LOG(ERROR) << name_ << ": Elements size must be equal to outputs shape[0] size.";
    return FAILED;
  }

  for (auto &element : elements) {
    MS_EXCEPTION_IF_NULL(element);
    if (element->isa<Int64Imm>()) {
      int64_t axis = static_cast<int64_t>(element->cast<Int64ImmPtr>()->value());
      full_multiples_.push_back(axis);
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis must be int32.";
      return FAILED;
    }
  }

  // multiple >= 1, or = -1
  auto it =
    std::find_if(full_multiples_.begin(), full_multiples_.end(), [](auto ele) { return (ele < -1 || ele == 0); });
  if (it != full_multiples_.end()) {
    MS_LOG(ERROR) << name_ << ": the value of multiples must be >= 1 or = -1, but it's " << full_multiples_;
    return FAILED;
  }

  if (full_multiples_.size() < inputs_shape_[0].size()) {
    MS_LOG(ERROR) << name_
                  << ": the size of multiples must be larger than or equal to the size of input's shape, but the size "
                     "of multiples is "
                  << full_multiples_.size() << ", and the size of input's shape is " << inputs_shape_[0].size();
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": full multiples is " << full_multiples_;
  return SUCCESS;
}

// the len of strategy is equal to multiples
// 1. If multiple > 1,split the multiple.
// 2. If multiple = 1:
//    1) If the dimension corresponding to multiple is empty: can not split
//    2) Otherwise, split the input shape
// 3. If multiple = -1: can not split
Status TileInfo::CheckStrategy(const StrategyPtr &strategy) {
  Shape tmp;
  for (size_t i = 0; i < full_multiples_.size(); ++i) {
    if (full_multiples_[i] == -1) {
      tmp.push_back(NO_SPLIT_STRATEGY);
    } else if (full_multiples_[i] > 1) {
      tmp.push_back(full_multiples_[i]);
    } else {
      auto len = full_multiples_.size() - inputs_shape_[0].size();
      if (i < len) {  // the dimension corresponding to multiple is empty
        tmp.push_back(NO_SPLIT_STRATEGY);
      } else {
        tmp.push_back(inputs_shape_[0][i - len]);
      }
    }
  }
  Shapes multiples = {tmp};
  MS_LOG(INFO) << name_ << ": The input shape is " << ShapeToString(inputs_shape_[0]) << ", the dims is "
               << ShapeToString(full_multiples_) << ", so the 'shape' can be split is " << ShapeToString(tmp);
  return CheckStrategyValue(strategy, multiples);
}

Status TileInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }
  if (full_multiples_.size() != stra[0].size()) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];

  slice_multiples_ = full_multiples_;
  for (size_t i = 0; i < full_multiples_.size(); ++i) {
    if (full_multiples_[i] > 1) {  // split the multiple only when multiple > 1
      MS_EXCEPTION_IF_ZERO("dev_matrix_shape_[i]", dev_matrix_shape_[i]);
      slice_multiples_[i] = slice_multiples_[i] / dev_matrix_shape_[i];
    }
  }
  return SUCCESS;
}

Status TileInfo::InferTensorMap() {
  TensorMap input_tensor_map;
  TensorMap output_tensor_map;
  if (inputs_shape_.empty() || outputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs or outputs' shape is empty";
    return FAILED;
  }

  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    input_tensor_map.push_back(inputs_shape_[0].size() - i - 1);
  }

  auto len = full_multiples_.size() - inputs_shape_[0].size();
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (full_multiples_[i + len] != 1) {  // split the input shape only when multiple = 1
      input_tensor_map[i] = MAP_NONE;
    }
  }

  // cannot use dev_matrix_shape_ replace outputs_shape_[0], because it may not be fully split in all devices.
  int64_t size = SizeToLong(outputs_shape_[0].size());
  for (int64_t i = 0; i < size; ++i) {
    output_tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(input_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
  return SUCCESS;
}

Status TileInfo::InferMirrorOps() {
  mirror_ops_.clear();
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

  OperatorVector input_op, multiples_op;
  input_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(input_op);
  mirror_ops_.push_back(multiples_op);
  return SUCCESS;
}

void TileInfo::UpdateDynamicMultiples(const AnfNodePtr &multiples_input_node) {
  auto strategy = strategy_->GetInputDim()[0];
  if (std::accumulate(strategy.cbegin(), strategy.cend(), 1, std::multiplies<int64_t>()) == 1) {
    return;
  }

  MS_EXCEPTION_IF_NULL(multiples_input_node);
  if (!IsPrimitiveCNode(multiples_input_node, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << "The dynamic input only support MakeTuple cnode, but got "
                      << multiples_input_node->fullname_with_scope();
  }

  auto make_tuple_cnode = multiples_input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple_cnode);

  for (size_t i = 1; i < make_tuple_cnode->inputs().size(); ++i) {
    if (strategy[i - 1] <= 1) {
      continue;
    }

    auto input_node = make_tuple_cnode->input(i);
    MS_EXCEPTION_IF_NULL(input_node);
    auto value_node = GetValueNode(input_node);
    if (value_node != nullptr && value_node->isa<Int64Imm>()) {
      auto origin_multiple_ele = GetValue<int64_t>(value_node);
      if (origin_multiple_ele <= 1) {  // update the multiple only when multiple > 1
        continue;
      }
      if (origin_multiple_ele % strategy[i - 1] != 0) {
        MS_LOG(EXCEPTION) << name_ << ": the origin shape is " << origin_multiple_ele
                          << ", can not be div by shard size " << strategy[i - 1];
      }
      int64_t replace_multiple = origin_multiple_ele / strategy[i - 1];
      MS_LOG(INFO) << name_ << ": replace multiple from " << origin_multiple_ele << " to " << replace_multiple
                   << "the index is " << (i - 1);
      auto replace_value_ptr = MakeValue(replace_multiple);
      auto replace_value_node = std::make_shared<ValueNode>(replace_value_ptr);
      make_tuple_cnode->set_input(i, replace_value_node);
    }
  }
}

void TileInfo::UpdateMultiples() {
  for (auto &cnode : cnodes_) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() != 3) {
      MS_LOG(EXCEPTION) << name_ << ": The size of tile cnode's inputs must be 3";
    }

    if (IsValueNode<ValueTuple>(cnode->input(2))) {
      auto func_graph = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      auto manager = func_graph->manager();
      MS_EXCEPTION_IF_NULL(manager);

      ValuePtr new_multiples = MakeValue(slice_multiples_);
      AnfNodePtr val = NewValueNode(new_multiples);
      MS_LOG(INFO) << name_ << ": the new multiples is " << slice_multiples_;
      cnode->set_input(kIndex2, val);
    } else {
      UpdateDynamicMultiples(cnode->input(kIndex2));
    }
  }
}

void TileInfo::ReplaceNodeInputOrAttrs() { UpdateMultiples(); }

std::shared_ptr<Strategies> TileInfo::GenerateBatchStrategies() {
  if (InferAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Infer attrs failed";
  }
  Shapes multiples_shape = {full_multiples_};
  split_flag_list_ = {true};
  return GenerateBatchStrategiesBySplitFlag(multiples_shape, split_flag_list_);
}

Status TileInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> TileInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape multiples_split(full_multiples_.size(), 1);
  Shapes splittable_inputs = {multiples_split};

  std::vector<StrategyPtr> sp_vector;
  Shape tmp_input_shape = full_multiples_;
  auto len = full_multiples_.size() - inputs_shape_[0].size();
  for (size_t i = 0; i < full_multiples_.size(); ++i) {
    if (full_multiples_[i] > 1) {
      continue;
    } else if (full_multiples_[i] == -1) {
      tmp_input_shape[i] = NO_SPLIT_STRATEGY;
    } else {
      if (i < len) {
        tmp_input_shape[i] = NO_SPLIT_STRATEGY;
      } else {
        tmp_input_shape[i] = inputs_shape_[0][i - len];
      }
    }
  }
  Shapes tmp_inputs_shape = {full_multiples_};
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": generate strategies failed";
  }

  return sp_vector;
}

REGISTER(TileInfo);
}  // namespace parallel
}  // namespace mindspore
