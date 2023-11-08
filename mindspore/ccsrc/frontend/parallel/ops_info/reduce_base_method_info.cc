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

#include "frontend/parallel/ops_info/reduce_base_method_info.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "ir/value.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kNameAxis = "axis";
constexpr auto kNameKeepDims = "keep_dims";

bool IsDataParallelStrategy(const Dimensions &strategy, int32_t stage_id) {
  CheckGlobalDeviceManager();
  size_t total_dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  if (strategy.empty()) {
    MS_LOG(EXCEPTION) << "IsDataParallelStrategy: strategy is empty";
  }

  return (LongToSize(strategy[0]) == total_dev_num);
}
}  // namespace

Status ReduceBaseMethod::InferMirrorOps() {
  mirror_ops_.clear();
  Shape input_tensor_map = inputs_tensor_map_.at(0);
  std::vector<Group> input_group;
  if (CreateGroupByTensorMap(input_tensor_map, &input_group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }

  if (input_group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror ops is empty.";
    return SUCCESS;
  } else {
    auto op_for_weight = CreateMirrorOps(input_group[0].name(), input_group[0].GetDevNum());
    mirror_ops_.push_back(op_for_weight);

    OperatorVector op_helper;
    auto prim_name = GetPrimNameFromInfoName(name_);
    auto res_size = ops::GetOpInputsNum(prim_name) - mirror_ops_.size();
    for (size_t i = 0; i < res_size; ++i) {
      mirror_ops_.push_back(op_helper);
    }

    std::string group_name = input_group[0].name();
    MS_LOG(INFO) << name_ << ": Create the mirror ops for weight success, the group is " << group_name;
  }

  return SUCCESS;
}

std::vector<int64_t> ReduceBaseMethod::reduce_dim() {
  std::vector<int64_t> dim_list{};
  auto axis_opt = GetArrayValueFromInputs<int64_t>(input_value_, name_, kNameAxis);
  if (!axis_opt.has_value()) {
    MS_LOG(EXCEPTION) << "For " << name_ << ", failed to get value for " << kNameAxis << ".";
  }

  auto axis_value = axis_opt.value();
  MS_ASSERT(inputs_shape_.size() >= 1);
  auto x_dim = inputs_shape_.at(0).size();
  // axis is (), reduce all dim
  if (axis_value.empty()) {
    for (size_t i = 0; i < x_dim; ++i) {
      dim_list.push_back(SizeToLong(i));
    }
  } else {
    auto AxisCorrectFunc = [x_dim](const int64_t axis) {
      if (axis < 0) {
        return axis + SizeToLong(x_dim);
      }
      return axis;
    };
    std::transform(axis_value.begin(), axis_value.end(), std::back_inserter(dim_list), AxisCorrectFunc);
  }
  return dim_list;
}

Status ReduceBaseMethod::GetAttrs() {
  // get attr cross_batch and keep_dims
  auto keep_dims_opt = GetScalarValueFromInputs<bool>(input_value_, name_, kNameKeepDims);
  if (!keep_dims_opt.has_value()) {
    MS_LOG(EXCEPTION) << "For " << name_ << ", failed to get value for " << kNameKeepDims << ".";
  }
  keepdims_ = keep_dims_opt.value();

  auto cross_batch_iter = attrs_.find(CROSS_BATCH);
  if (cross_batch_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(cross_batch_iter->second);
    if (!cross_batch_iter->second->isa<BoolImm>()) {
      MS_LOG(ERROR) << name_ << ": cross_batch is not a bool.";
      return FAILED;
    }
    cross_batch_ = cross_batch_iter->second->cast<BoolImmPtr>()->value();
  }
  auto reducemethodcost = std::dynamic_pointer_cast<ReduceMethodCost>(operator_cost());
  if (reducemethodcost == nullptr) {
    MS_LOG(ERROR) << "Cost cast to ReduceMethodCostPtr failed!";
    return FAILED;
  }
  reducemethodcost->set_cross_batch(cross_batch_);
  return SUCCESS;
}

Status ReduceMeanInfo::InferForwardCommunication() {
  auto strategies = strategy_->GetInputDim();
  Dimensions stra = strategies.at(0);
  if (cross_batch_ && IsDataParallelStrategy(stra, stage_id_)) {
    MS_LOG(INFO) << name_ << ": cross_batch is True, don't need to InferForwardCommunication";
    return SUCCESS;
  }
  forward_op_.clear();
  std::vector<int64_t> dim_list = reduce_dim();
  size_t size = stra.size();
  // judge if the reduce dim is partitioned.
  Shape group_creat_map;

  // if repeated calculation and the repeated_calc_num_ insert to the first dimension of dev matrix,
  // it need to handle the first dimension of map.
  if ((dev_matrix_shape_.size() > size) && !repeated_num_in_dev_matrix_right_) {
    group_creat_map.push_back(SizeToInt(dev_matrix_shape_.size() - size_t(1)));
  }

  for (size_t index = 0; index < size; ++index) {
    auto pos =
      std::find_if(dim_list.begin(), dim_list.end(), [index](const int64_t &dim) { return SizeToLong(index) == dim; });
    if (pos != dim_list.end() && stra[index] != 1) {
      continue;
    }
    group_creat_map.push_back(SizeToLong(size) - SizeToLong(index) - 1);
  }

  // if repeated calculation and the repeated_calc_num_ insert to the last dimension of dev matrix,
  // it need to handle the group_creat_map and insert the 0 to the last dimension of the group_creat_map.
  if (repeated_num_in_dev_matrix_right_ && (repeated_calc_num_ > 1)) {
    for (auto &ele : group_creat_map) {
      if (ele == MAP_NONE) {
        continue;
      }
      ele += 1;
    }
    group_creat_map.push_back(0);
  }

  std::vector<Group> forward_group;
  if (CreateGroupByTensorMap(group_creat_map, &forward_group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }
  if (!forward_group.empty()) {
    if ((outputs_dtype_ == nullptr) || !outputs_dtype_->isa<mindspore::TensorType>()) {
      MS_LOG(ERROR) << name_ << ": The dtype of output is not Array";
      return FAILED;
    }

    auto element_type = outputs_dtype_->cast<mindspore::TensorTypePtr>()->element();
    forward_op_ = CreateReduceMeanForwardOp(forward_group, element_type);
  }

  return SUCCESS;
}

ForwardOp ReduceAnyInfo::CreateForwardOp(const std::vector<Group> &forward_group) const {
  // Create Cast to Int32 op
  Operator op0 = CreateCastOp(kInt32);

  // Create AllReduce op
  Operator op1 = CreateAllReduceOp(reduce_method_, forward_group[0].name());
  std::string group_name = forward_group[0].name();
  MS_LOG(INFO) << "The group of forward all reduce is " << group_name << ", method is " << reduce_method_;

  // Create Cast to Bool op
  Operator op2 = CreateCastOp(kBool);

  ForwardOp forward_op = {op0, op1, op2};

  return forward_op;
}

Status ReduceAnyInfo::InferForwardCommunication() {
  auto strategies = strategy_->GetInputDim();
  Dimensions stra = strategies.at(0);
  if (cross_batch_ && IsDataParallelStrategy(stra, stage_id_)) {
    MS_LOG(INFO) << name_ << ": cross_batch is True, don't need to InferForwardCommunication";
    return SUCCESS;
  }
  forward_op_.clear();
  std::vector<int64_t> dim_list = reduce_dim();
  size_t size = stra.size();
  // judge if the reduce dim is partitioned.
  Shape group_creat_map;

  // if repeated calculation and the repeated_calc_num_ insert to the first dimension of dev matrix,
  // it need to handle the first dimension of map.
  if ((dev_matrix_shape_.size() > size) && !repeated_num_in_dev_matrix_right_) {
    group_creat_map.push_back(SizeToInt(dev_matrix_shape_.size() - size_t(1)));
  }

  for (size_t index = 0; index < size; ++index) {
    auto pos =
      std::find_if(dim_list.begin(), dim_list.end(), [index](const int64_t &dim) { return SizeToLong(index) == dim; });
    if (pos != dim_list.end() && stra[index] != 1) {
      continue;
    }
    group_creat_map.push_back(SizeToLong(size) - SizeToLong(index) - 1);
  }

  // if repeated calculation and the repeated_calc_num_ insert to the last dimension of dev matrix,
  // it need to handle the group_creat_map and insert the 0 to the last dimension of the group_creat_map.
  if (repeated_num_in_dev_matrix_right_ && (repeated_calc_num_ > 1)) {
    for (auto &ele : group_creat_map) {
      if (ele == MAP_NONE) {
        continue;
      }
      ele += 1;
    }
    group_creat_map.push_back(0);
  }

  std::vector<Group> forward_group;
  if (CreateGroupByTensorMap(group_creat_map, &forward_group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }
  if (!forward_group.empty()) {
    forward_op_ = CreateForwardOp(forward_group);
  }

  return SUCCESS;
}

REGISTER(ReduceMaxInfo);
REGISTER(ReduceMeanInfo);
REGISTER(ReduceSumInfo);
REGISTER(ReduceAnyInfo);
REGISTER(ReduceMinInfo);
REGISTER(ReduceProdInfo);
REGISTER(ReduceAllInfo);
}  // namespace parallel
}  // namespace mindspore
