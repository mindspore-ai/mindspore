/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/onehot_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/strategy.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status OneHotInfo::GetAttrs() {
  auto iter = attrs_.find(AXIS);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int64Imm>()) {
      axis_value_ptr_ = iter->second;
      axis_ = iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis is not int64_t.";
      return FAILED;
    }
  }

  if (inputs_shape_[0].size() != 1) {
    MS_LOG(ERROR) << name_ << ": Input's shape only support 1-D now.";
    return FAILED;
  }

  if ((axis_ > 1) || (axis_ < -1)) {
    MS_LOG(ERROR) << name_ << ": Axis " << axis_ << " is out of range[-1, 1].";
    return FAILED;
  }
  return SUCCESS;
}

Status OneHotInfo::CheckStrategy(const StrategyPtr &strategy) {
  return CheckStrategyValue(strategy, {outputs_shape_.at(0), inputs_shape_.at(1), inputs_shape_.at(2)});
}

Status OneHotInfo::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  // Now input only support 1-D tensor, so the output is a 2-D tensor
  // If input is a vector of length features, the output shape will be:
  // [features, depth] if axis == -1 (or axis == 1)
  // [depth, features] if axis == 0
  if (axis_ == 0) {
    dev_matrix_shape_.push_back(input_strategy[1]);  // the depth is un-splittable
    dev_matrix_shape_.push_back(input_strategy[0]);  // the features is splittable
  } else {
    dev_matrix_shape_.push_back(input_strategy[0]);  // the features is splittable
    dev_matrix_shape_.push_back(input_strategy[1]);  // the depth is un-splittable
  }
  old_dev_matrix_back_ = dev_matrix_shape_.back();
  repeated_num_in_dev_matrix_right_ = false;
  return SUCCESS;
}

Status OneHotInfo::InferTensorMap() {
  Shape input_tensor_map_index, output_tensor_map_index;
  size_t size = outputs_shape_[0].size();
  // such as 2: tensor_map_index [1,0]
  if (axis_ == 0) {
    for (size_t i = 0; i < size; ++i) {
      output_tensor_map_index.push_back((int64_t)(i));
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      output_tensor_map_index.push_back((int64_t)(LAST_INDEX(size) - i));
    }
  }
  outputs_tensor_map_.push_back(output_tensor_map_index);

  // Now input only support 1-D tensor
  input_tensor_map_index.push_back(1);

  inputs_tensor_map_.push_back(input_tensor_map_index);
  return SUCCESS;
}

// axis = -1
// (0,(1,16),(),()）reid   dev_matrix=(1,16)  map_in=(1) map_out=(1,0)
// (0,(16,1),(),()）data parallel dev_matrix=(16,1)  map_in=(1) map_out=(1,0)
// (0,(2,8),(),()）16 devices two machines，model parallel among devices in the same machine，data parallel between
// machines dev_matrix=(2,8)  map_in=(1) map_out=(1,0) (0, (2,4）,(),()）16 devices dev_matrix=(2,4,2)  map_in=(1)
// map_out=(1,0)
// axis = 0
// (0, (16,1),(),()）reid   dev_matrix=(1,16)  map_in=(1) map_out=(0,1)
// (0, (1,16),(),()）data parallel dev_matrix=(16,1)  map_in=(1) map_out=(0,1)
// (0, (8,2),(),()）16 devices two machines，model parallel among devices in the same machine，data parallel between
// machines dev_matrix=(2,8)  map_in=(1) map_out=(0,1) （0，（4,2）,(),()）16 devices dev_matrix=(2,4,2)  map_in=(1)
// map_out=(0,1)
Status OneHotInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);
  Shape output_shape = outputs_shape_.at(0);

  TensorLayout input_tensor_layout, output_tensor_layout;
  if ((input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], input_shape) != SUCCESS) ||
      (output_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], output_shape) != SUCCESS)) {
    return FAILED;
  }

  TensorInfo input_tensor_info(input_tensor_layout);
  TensorInfo output_tensor_info(output_tensor_layout);

  inputs_tensor_info_.push_back(input_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);

  return SUCCESS;
}

Status OneHotInfo::ExtractInputInfo() {
  CheckGlobalDeviceManager();
  rank_ = g_device_manager->rank_index_in_stage();
  mod_rank_ = rank_ % old_dev_matrix_back_;
  if (!cnode_) {
    MS_LOG(ERROR) << "Failure:OneHot cnode_ is nullptr";
    return FAILED;
  }
  if (cnode_->inputs().size() != 5) {
    MS_LOG(ERROR) << "Failure:There is 5 inputs for the CNode corresponding to OneHot Primitive, real input size is "
                  << cnode_->inputs().size();
    return FAILED;
  }
  if (input_value_.size() != 4) {
    MS_LOG(ERROR) << "Failure:There is 5 inputs for the CNode corresponding to OneHot Primitive, and input value size "
                     "must be 4, real size is "
                  << input_value_.size();
    return FAILED;
  }
  auto value_ptr = input_value_.at(1);
  if (value_ptr == nullptr) {
    MS_LOG(WARNING) << "Input 2 of cnode is not a value node, its type is " << cnode_->input(2)->type_name();
    return FAILED;
  }

  if (value_ptr->isa<Int64Imm>()) {
    total_class_number_ = value_ptr->cast<Int64ImmPtr>()->value();
  } else {
    MS_LOG(ERROR) << "OneHot Primitive depth type must be int64_t";
    return FAILED;
  }
  classes_each_device_ = total_class_number_ / old_dev_matrix_back_;

  return SUCCESS;
}

Status OneHotInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  if (old_dev_matrix_back_ == 1) {
    replace_graph_ = nullptr;
    return SUCCESS;
  }
  if (ExtractInputInfo() != SUCCESS) {
    MS_LOG(ERROR) << "ExtractInputInfo failed";
    return FAILED;
  }
  GenerateGraph gen_g = GenerateGraph();
  Status status = gen_g.Init(cnode);
  if (status != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }

  auto floor_div =
    gen_g.PushBack({gen_g.NewOpInst(FLOORDIV), gen_g.virtual_input_node(), CreateInt32Tensor(classes_each_device_)});
  auto mul1 = gen_g.PushBack({gen_g.NewOpInst(MUL), floor_div, CreateInt32Tensor(classes_each_device_)});
  auto sub1 = gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), mul1});
  auto equal = gen_g.PushBack({gen_g.NewOpInst(EQUAL), floor_div, CreateInt32Tensor(mod_rank_)});
  auto cast = gen_g.PushBack({gen_g.NewOpInst(CAST), equal, CreatTypeInt(32)});
  auto mul2 = gen_g.PushBack({gen_g.NewOpInst(MUL), sub1, cast});
  auto tensor_add = gen_g.PushBack({gen_g.NewOpInst(ADD), mul2, CreateInt32Tensor(1)});
  auto mul3 = gen_g.PushBack({gen_g.NewOpInst(MUL), cast, tensor_add});
  auto sub2 = gen_g.PushBack({gen_g.NewOpInst(SUB), mul3, CreateInt32Tensor(1)});
  Attr attr_onehot_axis = std::make_pair(AXIS, axis_value_ptr_);
  OperatorAttrs attrs_onehot = {attr_onehot_axis};
  auto onehot = gen_g.PushBack({gen_g.NewOpInst(ONEHOT, attrs_onehot), sub2, CreatInt64Imm(classes_each_device_),
                                cnode->input(3), cnode->input(4)});
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(floor_div, 1), std::make_pair(sub1, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, onehot));

  return SUCCESS;
}

ReplaceGraphPtr OneHotInfo::replace_graph(const CNodePtr &cnode) {
  if (ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": ComputeReplaceGraph failed.";
    return nullptr;
  }
  return replace_graph_;
}

Status OneHotInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  Status status = ComputeReplaceGraph(cnode_);
  if (status != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": ComputeReplaceGraph failed.";
    return status;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status OneHotInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status OneHotInfo::GenerateStrategies(int64_t stage_id) {
  Shapes splittable_inputs = {{1, 1}, {}, {}};
  std::vector<StrategyPtr> sp_vector;
  if (inputs_shape_.size() != 3) {
    MS_LOG(ERROR) << name_ << ": inputs_shape_ size must be 3, but is " << inputs_shape_.size();
    return FAILED;
  }
  if (outputs_shape_.size() != 1) {
    MS_LOG(ERROR) << name_ << ": outputs_shape_ size must be 1, but is " << outputs_shape_.size();
    return FAILED;
  }
  if (GenerateStrategiesForIndependentInputs(stage_id, {outputs_shape_.at(0), inputs_shape_.at(1), inputs_shape_.at(2)},
                                             splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateStrategies failed.";
    return FAILED;
  }

  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }

  return SUCCESS;
}

Status OneHotInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::shared_ptr<Strategys> OneHotInfo::GenerateBatchStrategies() {
  Dimensions strategy = {stage_device_size_, 1};
  Dimensions empty_strategy;
  Strategys strategy_v = {strategy, empty_strategy, empty_strategy};
  return std::make_shared<Strategys>(strategy_v);
}
}  // namespace parallel
}  // namespace mindspore
