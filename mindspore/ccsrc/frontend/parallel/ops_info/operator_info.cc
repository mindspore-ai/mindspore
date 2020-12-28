/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/operator_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ir/dtype.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/edge_costmodel.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/context.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status OperatorInfo::CheckStrategyValue(const StrategyPtr &strategy, const Shapes &inputs_shape) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null.";
    return FAILED;
  }

  size_t strategy_size = strategy->GetInputNumber();
  size_t inputs_shape_size = inputs_shape.size();
  if (strategy_size != inputs_shape_size) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Strategy size: " << strategy_size
                    << " is not equal to inputs size: " << inputs_shape_size;
    } else {
      MS_LOG(ERROR) << name_ << ": Strategy size: " << strategy_size
                    << " is not equal to inputs size: " << inputs_shape_size;
    }
    return FAILED;
  }

  Strategys stra = strategy->GetInputDim();
  for (size_t i = 0; i < strategy_size; ++i) {
    Shape sub_strategy = stra.at(i);
    Shape sub_input_shape = inputs_shape.at(i);
    size_t strategy_len = sub_strategy.size();
    size_t inputs_len = sub_input_shape.size();
    if (strategy_len != inputs_len) {
      if (is_auto_parallel_) {
        MS_LOG(DEBUG) << name_ << ": Strategy len: " << strategy_len << " is not equal to inputs len: " << inputs_len
                      << ", index: " << i;
      } else {
        MS_LOG(ERROR) << name_ << ": Strategy len: " << strategy_len << " is not equal to inputs len: " << inputs_len
                      << ", index: " << i;
      }
      return FAILED;
    }

    for (size_t j = 0; j < strategy_len; ++j) {
      int64_t strategy_value = sub_strategy.at(j);
      if (strategy_value < MIN_SLICE_NUM) {
        if (is_auto_parallel_) {
          MS_LOG(DEBUG) << name_ << ": Invalid strategy value: " << strategy_value;
        } else {
          MS_LOG(ERROR) << name_ << ": Invalid strategy value: " << strategy_value;
        }
        return FAILED;
      }

      if ((LongToUlong(strategy_value) & LongToUlong(strategy_value - 1)) != 0) {
        if (is_auto_parallel_) {
          MS_LOG(DEBUG) << name_ << ": Invalid Strategy value it is not the power of 2, " << strategy_value;
        } else {
          MS_LOG(ERROR) << name_ << ": Invalid Strategy value it is not the power of 2, " << strategy_value;
        }
        return FAILED;
      }

      int64_t shape_value = sub_input_shape.at(j);
      if ((shape_value % strategy_value) != 0) {
        if (is_auto_parallel_) {
          MS_LOG(DEBUG) << name_ << ": Shape " << shape_value << " cannot be divisible by strategy " << strategy_value;
        } else {
          MS_LOG(ERROR) << name_ << ": Shape " << shape_value << " cannot be divisible by strategy " << strategy_value;
        }
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

void OperatorInfo::ResetQueueMember() {
  inputs_tensor_info_.clear();
  outputs_tensor_info_.clear();
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();
  dev_matrix_shape_.clear();
  forward_op_.clear();
  mirror_ops_.clear();
  sub_ops_.clear();
  replace_op_.clear();
  replace_op_info_.clear();
  virtual_div_op_.clear();
}

Status OperatorInfo::InferAttrs() {
  if (infer_attrs_completed_) {
    return SUCCESS;
  }

  if (GetAttrs() != SUCCESS) {
    return FAILED;
  }
  infer_attrs_completed_ = true;
  return SUCCESS;
}

Status OperatorInfo::InferMirrorOps() {
  mirror_ops_.clear();
  if (inputs_shape_.empty()) {
    MS_LOG(INFO) << name_ << ": The inputs size is empty";
    return SUCCESS;
  }

  if (inputs_tensor_map_.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": The size of inputs tensor map is not equal to the size of inputs shape";
    return FAILED;
  }

  bool group_is_empty = true;
  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    std::vector<Group> group;
    if (CreateGroupByTensorMap(inputs_tensor_map_[i], &group) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Create group failed, the input index is " << i;
      mirror_ops_.clear();
      return FAILED;
    }

    OperatorVector mirror_op;
    if (group.empty()) {
      MS_LOG(INFO) << name_ << ": The mirror group is empty, the input index is " << i;
      mirror_ops_.push_back(mirror_op);
      continue;
    }

    group_is_empty = false;
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
  }

  if (group_is_empty) {
    mirror_ops_.clear();
    MS_LOG(INFO) << name_ << ": No need to insert mirror ops";
  }
  return SUCCESS;
}

Status OperatorInfo::InferRepeatedCalcInfo() {
  int64_t g_dev_list_size = stage_device_size_;
  int64_t dev_matrix_size =
    std::accumulate(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), 1, std::multiplies<int64_t>());
  if (dev_matrix_size == 0) {
    MS_LOG(ERROR) << name_ << ": The dev matrix size is 0";
    return FAILED;
  }

  if (g_dev_list_size == dev_matrix_size) {
    repeated_calc_num_ = 1;
  } else if (g_dev_list_size % dev_matrix_size == 0) {
    repeated_calc_num_ = ((int64_t)(g_dev_list_size / dev_matrix_size));
  } else {
    MS_LOG(ERROR) << name_ << ": Dev list size " << g_dev_list_size << " can not be divisible by dev matrix size "
                  << dev_matrix_size;
    return FAILED;
  }
  return SUCCESS;
}

// If repeated calculation, set the repeated_calc_num as the last dimension of dev-matrix in default,
// because if the previous shard is (a, b), and the next shard is (a, 1), adding the repeated_calc_num
// to the last dimension of dev-matrix, there is no need to redistribution.
void OperatorInfo::SetRepeatedCalcDevMatrix() {
  if (repeated_calc_num_ <= 1) {
    return;
  }
  if (repeated_num_in_dev_matrix_right_) {
    dev_matrix_shape_.push_back(repeated_calc_num_);
  } else {
    (void)dev_matrix_shape_.insert(dev_matrix_shape_.begin(), repeated_calc_num_);
  }
}

// If repeated calculation, and the repeated_calc_num is inserted to the last dimension of the dev-matrix,
// the index value of tensor map needs to be increased by 1.
void OperatorInfo::ResetTensorMapIfRepeatedCalc() {
  if ((repeated_calc_num_ <= 1) || !repeated_num_in_dev_matrix_right_) {
    return;
  }

  MS_LOG(DEBUG) << name_ << ": the repeated calc num is " << repeated_calc_num_ << ", and reset the tensor maps";
  for (auto &tensor_map : inputs_tensor_map_) {
    for (auto &element : tensor_map) {
      if (element == MAP_NONE) {
        continue;
      }
      element += 1;
    }
  }

  for (auto &tensor_map : outputs_tensor_map_) {
    for (auto &element : tensor_map) {
      if (element == MAP_NONE) {
        continue;
      }
      element += 1;
    }
  }
}

// use for loss repeated calculation
Operator CreateVirtualDivOp(int64_t div_num) {
  OperatorName operator_name = VIRTUAL_DIV;
  ValuePtr attr0_value = MakeValue(div_num);
  Attr attr0 = std::make_pair(DIVISOR, attr0_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  return op;
}

// use for forward all reduce
Operator CreateAllReduceOp(const std::string &reduce_op, const std::string &group) {
  OperatorName operator_name = ALL_REDUCE;
  ValuePtr attr0_value = MakeValue(reduce_op);  // ReduceOP.SUM
  ValuePtr attr1_value = MakeValue(group);      // group
  Attr attr0 = std::make_pair(OP, attr0_value);
  Attr attr1 = std::make_pair(GROUP, attr1_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create all reduce op success, the reduce_op is  " << reduce_op << ", the group is " << group;
  return op;
}

Operator CreateReduceScatterOp(const std::string &reduce_op, const std::string &group) {
  OperatorName operator_name = REDUCE_SCATTER;
  ValuePtr attr0_value = MakeValue(reduce_op);  // ReduceOP.SUM
  ValuePtr attr1_value = MakeValue(group);      // group
  Attr attr0 = std::make_pair(OP, attr0_value);
  Attr attr1 = std::make_pair(GROUP, attr1_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create reduce scatter op success, the reduce_op is  " << reduce_op << ", the group is " << group;
  return op;
}

void AddCommOpFusionType(const CNodePtr &comm_node, const AnfNodePtr &param_node) {
  MS_EXCEPTION_IF_NULL(comm_node);
  MS_EXCEPTION_IF_NULL(param_node);
  if (IsPrimitiveCNode(param_node, prim::kPrimReceive)) {
    MS_LOG(WARNING) << "The mirror of Receive does not support fusion type now.";
    return;
  }
  auto param = param_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param);
  auto prim = GetValueNode<PrimitivePtr>(comm_node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto attrs = prim->attrs();
  auto param_info = param->param_info();
  if (!param_info) {
    MS_LOG(WARNING) << param->ToString() << "does not have parameter info.";
    return;
  }
  int32_t fusion_type = param_info->comm_fusion();
  attrs[FUSION] = MakeValue<int64_t>(fusion_type);
  prim->SetAttrs(attrs);
  MS_LOG(INFO) << "Set comm fusion:" << param->param_info()->name() << "'s fusion type is " << fusion_type;
}

void AddCommOpMeanFlag(const CNodePtr &comm_node) {
  MS_EXCEPTION_IF_NULL(comm_node);
  auto prim = GetValueNode<PrimitivePtr>(comm_node->input(0));
  auto attrs = prim->attrs();
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();
  attrs[MEAN_FLAG] = MakeValue<bool>(mean_flag);
  prim->SetAttrs(attrs);
}

Operator CreateAllGatherOp(const std::string &group) {
  OperatorName operator_name = ALL_GATHER;
  ValuePtr attr0_value = MakeValue(group);  // group
  Attr attr0 = std::make_pair(GROUP, attr0_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create allgather op success, the group is " << group;
  return op;
}

Operator CreateMiniStepAllGatherOp(const std::string &group) {
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();

  OperatorName operator_name = MINI_STEP_ALL_GATHER;
  ValuePtr attr0_value = MakeValue(group);  // group
  Attr attr0 = std::make_pair(GROUP, attr0_value);
  ValuePtr attr1_value = MakeValue(grad_accumulation_step);  // grad_accumulation_step
  Attr attr1 = std::make_pair(GRAD_ACCUMULATION_STEP, attr1_value);
  ValuePtr attr2_value = MakeValue(mean_flag);  // mean_flag
  Attr attr2 = std::make_pair(MEAN_FLAG, attr2_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);
  operator_attrs.push_back(attr2);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create MINI_STEP_ALL_GATHER success, the group is " << group;
  return op;
}

// use for get tensor slice
Operator CreateGetTensorSliceOp(const TensorLayout &tensor_layout) {
  Shape tensor_map = tensor_layout.tensor_map().array();
  Shape dev_matrix_shape = tensor_layout.device_arrangement().array();
  OperatorName operator_name = GET_TENSOR_SLICE;

  OperatorAttrs attrs;
  ValuePtr dev_mat_value = MakeValue(dev_matrix_shape);
  Param dev_mat_param = std::make_pair(std::make_pair(DEV_MAT, dev_mat_value), 2);
  ValuePtr tensor_map_value = MakeValue(tensor_map);
  Param tensor_map_param = std::make_pair(std::make_pair(TENSOR_MAP, tensor_map_value), 3);
  OperatorParams params = {dev_mat_param, tensor_map_param};
  OperatorArgs operator_arg = std::make_pair(attrs, params);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create get tensor slice op success, the dev mat and tensor map is "
               << ShapeToString(dev_matrix_shape) << ", " << ShapeToString(tensor_map);
  return op;
}

OperatorVector CreateMirrorOps(const std::string &group_name, size_t dev_num) {
  if ((dev_num == 0) || (dev_num == 1)) {
    MS_LOG(EXCEPTION) << "Invalid dev num: " << dev_num;
  }
  OperatorVector op_for_weight;
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();

  ValuePtr attr0_value = MakeValue(group_name);
  ValuePtr attr1_value = MakeValue(SizeToLong(dev_num));
  ValuePtr attr2_value = MakeValue(mean_flag);

  Attr attr0 = std::make_pair(GROUP, attr0_value);
  Attr attr1 = std::make_pair(DEV_NUM, attr1_value);
  Attr attr2 = std::make_pair(MEAN_FLAG, attr2_value);

  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);
  operator_attrs.push_back(attr2);

  OperatorName operator_name;
  if (grad_accumulation_step > 1) {
    operator_name = MIRROR_MINI_STEP_OPERATOR;
    ValuePtr attr3_value = MakeValue(grad_accumulation_step);
    Attr attr3 = std::make_pair(GRAD_ACCUMULATION_STEP, attr3_value);
    operator_attrs.push_back(attr3);
    MS_LOG(INFO) << "The grad accumulation step is " << grad_accumulation_step << ", use mini step mirror";
  } else {
    operator_name = MIRROR_OPERATOR;
  }

  OperatorParams operator_param;
  OperatorArgs operator_args = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_args);

  op_for_weight.push_back(op);
  MS_LOG(INFO) << "The group name is " << group_name << ", the dev num is " << dev_num << ", the mean flag is "
               << mean_flag;
  return op_for_weight;
}

Status OperatorInfo::CreateGroupByTensorMap(const Shape &tensor_map, std::vector<Group> *group) {
  if (group == nullptr) {
    MS_LOG(ERROR) << "The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(tensor_map, &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() == 1) {
    MS_LOG(INFO) << "The dev size is 1, no need to create group.";
    return SUCCESS;
  }

  Group g = g_device_manager->CreateGroup(group_devices);
  group->push_back(g);
  return SUCCESS;
}

Status OperatorInfo::CreateGroupByDim(size_t axis, std::vector<Group> *group) {
  if (group == nullptr) {
    MS_LOG(ERROR) << "The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesAlongDim(SizeToUlong(axis), &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() == 1) {
    MS_LOG(INFO) << "The dev size is 1, no need to create group.";
    return SUCCESS;
  }

  Group g = g_device_manager->CreateGroup(group_devices);
  group->push_back(g);
  return SUCCESS;
}

Shape GetSliceShape(const Shape &tensor_shape, const Dimensions &strategy) {
  Shape slice_shape;
  if (std::any_of(strategy.begin(), strategy.end(), [](int64_t value) { return value <= 0; })) {
    MS_LOG(ERROR) << "Invalid strategy: " << ShapeToString(strategy) << ", the element is less than or equal to 0";
    return slice_shape;
  }
  for (size_t i = 0; i < strategy.size(); ++i) {
    slice_shape.push_back(tensor_shape.at(i) / strategy.at(i));
  }
  return slice_shape;
}

Status InferSliceShapeByStrategy(const Strategys &strategys, const Shapes &shapes, Shapes *slice_shapes) {
  if (slice_shapes == nullptr) {
    MS_LOG(ERROR) << "The slice_shapes is null.";
    return FAILED;
  }
  if (strategys.size() != shapes.size()) {
    MS_LOG(ERROR) << "Strategy size " << strategys.size() << " not equal to shape size " << shapes.size();
    return FAILED;
  }

  for (size_t i = 0; i < strategys.size(); ++i) {
    if (strategys.at(i).size() != shapes.at(i).size()) {
      MS_LOG(ERROR) << "Strategy dimension " << strategys.at(i).size() << " not equal to shape dimension "
                    << shapes.at(i).size();
      slice_shapes->clear();
      return FAILED;
    }

    for (size_t j = 0; j < shapes.at(i).size(); ++j) {
      if (strategys.at(i).at(j) <= 0) {
        MS_LOG(ERROR) << "Invalid strategy: " << ShapeToString(strategys[i])
                      << " the element is less than or equal to 0.";
        slice_shapes->clear();
        return FAILED;
      }
      if (shapes.at(i).at(j) % strategys.at(i).at(j) != 0) {
        MS_LOG(ERROR) << "Shape cannot be divisible by strategy, " << shapes.at(i).at(j) << " : "
                      << strategys.at(i).at(j);
        slice_shapes->clear();
        return FAILED;
      }
    }
    Shape slice_shape = GetSliceShape(shapes.at(i), strategys.at(i));
    slice_shapes->push_back(slice_shape);
  }

  return SUCCESS;
}

Status OperatorInfo::InferSliceShape(const Strategys &inputs_strategy, const Strategys &outputs_strategy,
                                     Shapes *inputs_slice_shape, Shapes *outputs_slice_shape) {
  if (inputs_slice_shape == nullptr || outputs_slice_shape == nullptr) {
    MS_LOG(ERROR) << "The slice_shape is null.";
    return FAILED;
  }

  if (InferSliceShapeByStrategy(inputs_strategy, inputs_shape_, inputs_slice_shape) != SUCCESS) {
    MS_LOG(ERROR) << "Infer inputs slice shape error.";
    return FAILED;
  }

  if (InferSliceShapeByStrategy(outputs_strategy, outputs_shape_, outputs_slice_shape) != SUCCESS) {
    MS_LOG(ERROR) << "Infer outputs slice shape error.";
    inputs_slice_shape->clear();
    return FAILED;
  }

  return SUCCESS;
}

// method0: auto insert repeated_calculation_num for dev_matrix_shape when repeated_calculation_num > 1
Status OperatorInfo::InitForCostModelWithAutoRepeatCalc(const StrategyPtr &strategy) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null.";
    return FAILED;
  }

  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAttrs failed.";
    return FAILED;
  }

  // must be after InferAttrs()
  if (CheckStrategy(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": CheckStrategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": CheckStrategy failed.";
    }
    return FAILED;
  }

  // need to clear queues before Init(),
  // because Init() may be called multiple times by cost model
  ResetQueueMember();

  strategy_ = strategy;

  if (InferDevMatrixShape() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferDevMatrixShape failed.";
    return FAILED;
  }

  used_devices_ =
    ((int64_t)(std::accumulate(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), 1, std::multiplies<int64_t>())));

  // must be after InferDevMatrixShape
  if (InferRepeatedCalcInfo() != SUCCESS) {
    MS_LOG(ERROR) << ": InferRepeatedCalcInfo failed.";
    return FAILED;
  }

  // if repeated calculation, need to set the repeated_calc_num as the last dimension of dev-matrix for layout
  SetRepeatedCalcDevMatrix();

  if (InferTensorMap() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferTensorMap failed.";
    return FAILED;
  }

  ResetTensorMapIfRepeatedCalc();

  if (InferTensorInfo() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferTensorInfo failed.";
    return FAILED;
  }

  return SUCCESS;
}

// method1: manually insert repeated_calculation_num for dev_matrix_shape in InferDevMatrixShape
Status OperatorInfo::InitForCostModelWithManualRepeatCalc(const StrategyPtr &strategy) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null.";
    return FAILED;
  }

  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAttrs failed.";
    return FAILED;
  }

  // must be after InferAttrs()
  if (CheckStrategy(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": CheckStrategy failed.";
    return FAILED;
  }

  // need to clear queues before Init(),
  // because Init() may be called multiple times by cost model
  ResetQueueMember();

  strategy_ = strategy;

  if (InferDevMatrixShape() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferDevMatrixShape failed.";
    return FAILED;
  }

  // must be after InferDevMatrixShape
  if (InferRepeatedCalcInfo() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferRepeatedCalcInfo failed.";
    return FAILED;
  }

  if (InferTensorMap() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferTensorMap failed.";
    return FAILED;
  }

  if (InferTensorInfo() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferTensorInfo failed.";
    return FAILED;
  }

  return SUCCESS;
}

Status OperatorInfo::InitWithAutoRepeatCalc(const StrategyPtr &strategy) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null.";
    return FAILED;
  }

  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    return FAILED;
  }

  if (InferForwardCommunication() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferForwardCommunication failed.";
    return FAILED;
  }

  if (InferMirrorOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferMirrorOps failed.";
    return FAILED;
  }

  if (InferVirtualDivOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferVirtualDivOps failed.";
    return FAILED;
  }

  return SUCCESS;
}

Status OperatorInfo::InitWithManualRepeatCalc(const StrategyPtr &strategy) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null.";
    return FAILED;
  }

  if (InitForCostModelWithManualRepeatCalc(strategy) != SUCCESS) {
    return FAILED;
  }

  if (InferForwardCommunication() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferForwardCommunication failed.";
    return FAILED;
  }

  if (InferMirrorOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferMirrorOps failed.";
    return FAILED;
  }

  if (InferVirtualDivOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferVirtualDivOps failed.";
    return FAILED;
  }

  return SUCCESS;
}

std::vector<std::shared_ptr<Edge>> OperatorInfo::GetAliveSuccEdges() {
  std::vector<std::shared_ptr<Edge>> ret;
  for (auto &edge : succ_edges_) {
    if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(RELU) != std::string::npos)) {
      ret.push_back(edge);
    } else if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(CAST) != std::string::npos)) {
      // CAST is ordered in front of L2NORMALIZE
      ret.push_back(edge);
    }
  }
  for (auto &edge : succ_edges_) {
    if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(RELU) == std::string::npos) &&
        (edge->next_operator()->name().find(CAST) == std::string::npos)) {
      ret.push_back(edge);
    }
  }
  return ret;
}

std::vector<std::shared_ptr<Edge>> OperatorInfo::GetAlivePrevEdges() {
  std::vector<std::shared_ptr<Edge>> ret;
  for (auto &edge : prev_edges_) {
    if (edge->prev_operator()->is_alive()) {
      ret.push_back(edge);
    }
  }
  return ret;
}

void OperatorInfo::ReplacePreEdge(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplacePreEdge: the op is null.";
    return;
  }
  for (auto &edge : prev_edges_) {
    if (edge->prev_operator() == op) {
      edge = new_edge;
      return;
    }
  }
  MS_LOG(EXCEPTION) << name_ << ": Replace edge failed: no edge has been replaced";
}

void OperatorInfo::ReplaceSuccEdge(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplaceSuccEdge: the op is null.";
    return;
  }
  for (auto &edge : succ_edges_) {
    if (edge->next_operator() == op) {
      edge = new_edge;
      return;
    }
  }
  MS_LOG(EXCEPTION) << name_ << ": Replace edge failed: no edge has been replaced";
}

void OperatorInfo::ReplacePreEdges(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplacePreEdges: the op is null.";
    return;
  }
  std::vector<std::shared_ptr<Edge>> new_pre_edges;
  for (auto &edge : prev_edges_) {
    if (edge->prev_operator() != op) {
      new_pre_edges.push_back(edge);
    }
  }
  new_pre_edges.push_back(new_edge);
  prev_edges_ = new_pre_edges;
}

void OperatorInfo::ReplaceSuccEdges(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplaceSuccEdges: the op is null";
    return;
  }
  std::vector<std::shared_ptr<Edge>> new_succ_edges;
  for (auto &edge : succ_edges_) {
    if (edge->next_operator() != op) {
      new_succ_edges.push_back(edge);
    }
  }
  new_succ_edges.push_back(new_edge);
  succ_edges_ = new_succ_edges;
}

std::shared_ptr<Strategys> GenerateBatchStrategiesBySplitFlag(const Shapes &shapes,
                                                              const std::vector<bool> &split_flag_list) {
  if (shapes.size() != split_flag_list.size()) {
    MS_LOG(ERROR) << "Split_flag_list do not have the same size as inputs shape, " << split_flag_list.size() << " : "
                  << shapes.size();
    return nullptr;
  }
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  Strategys strategy_v;
  for (size_t i = 0; i != shapes.size(); i++) {
    if (shapes[i].empty()) {
      MS_LOG(INFO) << "Elements of shapes is empty.";
      Dimensions empty_element;
      strategy_v.push_back(empty_element);
    } else {
      Dimensions element(shapes[i].size(), 1);
      if (split_flag_list[i]) {
        element[0] = dev_num;
      }
      strategy_v.push_back(element);
    }
  }
  return std::make_shared<Strategys>(strategy_v);
}

void OperatorInfo::ReComputeBatchSplitFlagList() {
  if (!inputs_shape_.empty()) {
    split_flag_list_[0] = true;
  }
}

void OperatorInfo::ComputeBatchSplitFlagList() {
  split_flag_list_.clear();
  for (auto iter = inputs_shape_.begin(); iter != inputs_shape_.end(); ++iter) {
    split_flag_list_.push_back(false);
  }
  ReComputeBatchSplitFlagList();
}

// This is a common method for checking whether the generated strategy has the correct number of devuces.
Status PrepareStrategyBase(int64_t stage_id, size_t dev_num, const Shapes &inputs_partitions, StrategyPtr *const sp) {
  if (sp == nullptr) {
    MS_LOG(ERROR) << "The strategy is null.";
    return FAILED;
  }
  int64_t product = 1;

  for (auto &input_partition : inputs_partitions) {
    product *= std::accumulate(input_partition.begin(), input_partition.end(), 1, std::multiplies<int64_t>());
  }
  if (!FULLY_USE_DEVICES) {
    if (LongToSize(product) > dev_num) {
      return FAILED;
    }
  } else {
    if ((product != 1) && (LongToSize(product) != dev_num)) {
      return FAILED;
    }
  }
  Strategys stras(inputs_partitions);
  (*sp) = std::make_shared<Strategy>(stage_id, stras);
  return SUCCESS;
}

std::shared_ptr<Strategys> OperatorInfo::GenerateBatchStrategies() {
  ComputeBatchSplitFlagList();
  return GenerateBatchStrategiesBySplitFlag(inputs_shape_, split_flag_list_);
}

void PrintStrategy(const StrategyPtr &strategy) {
  if (strategy == nullptr) {
    return;
  }
  std::string all_strategy = "";
  for (size_t i = 0; i < strategy->GetInputNumber(); ++i) {
    all_strategy += "[";
    for (size_t j = 0; j < strategy->GetInputDim()[i].size(); ++j) {
      all_strategy += std::to_string(strategy->GetInputDim()[i][j]);
      if (j != strategy->GetInputDim()[i].size() - 1) {
        all_strategy += ", ";
      }
    }
    all_strategy += "]";
    if (i != strategy->GetInputNumber() - 1) {
      all_strategy += ", ";
    }
  }
  MS_LOG(INFO) << "The strategy is: " << all_strategy;
}

// generate strategies for that each dimension of input0 and input1 is relevant, such as: ([a, b, c, d], [a, b, c, d])
Status GenerateStrategiesForTwoEqualInputs(int64_t stage_id, const Shapes &inputs_shape,
                                           const Shapes &splittable_inputs, std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if ((inputs_shape.size() != 2) || (splittable_inputs.size() != 2)) {
    MS_LOG(ERROR) << "The inputs size is wrong.";
    return FAILED;
  }

  if ((inputs_shape[0].size() != inputs_shape[1].size()) ||
      (splittable_inputs[0].size() != splittable_inputs[1].size())) {
    MS_LOG(ERROR) << "The size of two inputs are not equal.";
    return FAILED;
  }

  Shapes input0_shape = {inputs_shape[0]};
  Shapes input0_splittable = {splittable_inputs[0]};
  if (GenerateStrategiesForIndependentInputs(stage_id, input0_shape, input0_splittable, sp_vector) != SUCCESS) {
    return FAILED;
  }

  for (auto &sp : *sp_vector) {
    sp->ExpandInputDimFromOneToTwo();
  }

  return SUCCESS;
}

// generate strategies for that input0 and input1 have relevant dimensions, and input0 needs to broadcast
// such as: ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
Status GenerateStrategiesForBroadcastLeft(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                          std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if (inputs_shape[0].size() >= inputs_shape[1].size()) {
    MS_LOG(ERROR) << "Invalid inputs shape.";
    return FAILED;
  }

  // first, generate strategy for input0 the same as input1
  Shapes tmp_inputs_shape = {inputs_shape[1], inputs_shape[1]};
  Shapes tmp_splittable_inputs = {splittable_inputs[1], splittable_inputs[1]};
  if (GenerateStrategiesForTwoEqualInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
    return FAILED;
  }

  // second, get the correct strategy for input0
  for (auto &sp : *sp_vector) {
    Strategys tmp_strategy;
    Dimensions input0_strategy = sp->GetInputDim()[0];
    size_t size_diff = inputs_shape[1].size() - inputs_shape[0].size();

    // erase the unnecessary part
    (void)input0_strategy.erase(input0_strategy.begin(),
                                input0_strategy.begin() + static_cast<different_type>(size_diff));

    // handle the case likes ([1, c, d], [a, b, c, d])
    for (size_t i = 0; i < inputs_shape[0].size(); ++i) {
      if (inputs_shape[0][i] == 1) {
        input0_strategy[i] = 1;
      } else {
        break;
      }
    }

    // reset the strategy
    tmp_strategy.push_back(input0_strategy);       // input0
    tmp_strategy.push_back(sp->GetInputDim()[1]);  // input1
    sp->ResetInputs(tmp_strategy);
  }
  return SUCCESS;
}

// generate strategies for that input0 and input1 have relevant dimensions, and input1 needs to broadcast
// such as: ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
Status GenerateStrategiesForBroadcastRight(int64_t stage_id, const Shapes &inputs_shape,
                                           const Shapes &splittable_inputs, std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if (inputs_shape[0].size() <= inputs_shape[1].size()) {
    MS_LOG(ERROR) << "Invalid inputs shape.";
    return FAILED;
  }

  // first, generate strategy for input1 the same as input0
  Shapes tmp_inputs_shape = {inputs_shape[0], inputs_shape[0]};
  Shapes tmp_splittable_inputs = {splittable_inputs[0], splittable_inputs[0]};
  if (GenerateStrategiesForTwoEqualInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
    return FAILED;
  }

  // second, get the correct strategy for input1
  for (auto &sp : *sp_vector) {
    Strategys tmp_strategy;
    tmp_strategy.push_back(sp->GetInputDim()[0]);  // input0

    Dimensions input1_strategy = sp->GetInputDim()[1];
    size_t size_diff = inputs_shape[0].size() - inputs_shape[1].size();

    // erase the unnecessary part
    (void)input1_strategy.erase(input1_strategy.begin(),
                                input1_strategy.begin() + static_cast<different_type>(size_diff));

    // handle the case likes ([a, b, c, d], [1, c, d])
    for (size_t i = 0; i < inputs_shape[1].size(); ++i) {
      if (inputs_shape[1][i] == 1) {
        input1_strategy[i] = 1;
      } else {
        break;
      }
    }

    // reset the strategy
    tmp_strategy.push_back(input1_strategy);  // input1
    sp->ResetInputs(tmp_strategy);
  }
  return SUCCESS;
}

// generate strategies for that input0 and input1 have same size, and input0 or input1 needs to broadcast
// such as: ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
Status GenerateStrategiesForBroadcastBoth(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                          std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if (inputs_shape[0].size() != inputs_shape[1].size()) {
    MS_LOG(ERROR) << "Invalid inputs shape.";
    return FAILED;
  }

  // step1: ([a, 1], [1, b]) -> [a, b]
  Shape max_shape, splittable_vector;
  for (size_t i = 0; i < inputs_shape[0].size(); ++i) {
    if (inputs_shape[0][i] >= inputs_shape[1][i]) {
      max_shape.push_back(inputs_shape[0][i]);
      splittable_vector.push_back(splittable_inputs[0][i]);
    } else {
      max_shape.push_back(inputs_shape[1][i]);
      splittable_vector.push_back(splittable_inputs[1][i]);
    }
  }

  // step2: ([a, 1], [1, b]) -> generate strategy for ([a, b], [a, b])
  Shapes tmp_inputs_shape = {max_shape, max_shape};
  Shapes tmp_splittable_inputs = {splittable_vector, splittable_vector};
  if (GenerateStrategiesForTwoEqualInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
    return FAILED;
  }

  // step3: reset the strategy if the dimension is 1
  for (auto &sp : *sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    for (size_t i = 0; i < inputs_shape[0].size(); ++i) {
      if (inputs_shape[0][i] == 1) {
        input0_strategy[i] = 1;
      }

      if (inputs_shape[1][i] == 1) {
        input1_strategy[i] = 1;
      }
    }
    sp->ResetInputs({input0_strategy, input1_strategy});
  }

  return SUCCESS;
}

// 'splittable_inputs' has the same dimensions as 'inputs_shape_'. '0' in 'splittable_inputs' means that
// the corresponding dimension is unsplittable, '1' in 'splittable_inputs' means that the corresponding
// dimension is splittable. 'inputs_partitions' is the result of partitions.
// NOTE: This implementation would partition all splittable dimensions in all inputs. Some operators requiring
// specific dimensions in inputs have the identical partition should have individual implementation.
Status GenerateStrategiesForIndependentInputs(int64_t stage_id, const Shapes &inputs_shape,
                                              const Shapes &splittable_inputs,
                                              std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }
  if (splittable_inputs.size() != inputs_shape.size()) {
    MS_LOG(ERROR) << "Splittable_inputs do not have the same input number of inputs shape, " << splittable_inputs.size()
                  << " : " << inputs_shape.size();
    return FAILED;
  }
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

  Shape combined_inputs_shape, combined_splittable_inputs, combined_partitions;
  for (size_t j = 0; j < inputs_shape.size(); ++j) {
    (void)combined_inputs_shape.insert(combined_inputs_shape.end(), inputs_shape[j].begin(), inputs_shape[j].end());
    (void)combined_splittable_inputs.insert(combined_splittable_inputs.end(), splittable_inputs[j].begin(),
                                            splittable_inputs[j].end());
  }
  std::function<void(uint64_t, size_t)> recursive = [&stage_id, &dev_num, &sp_vector, &combined_inputs_shape,
                                                     &combined_splittable_inputs, &combined_partitions, &recursive,
                                                     &inputs_shape](uint64_t current_index, size_t n) {
    if (current_index == combined_inputs_shape.size()) {
      MS_LOG(DEBUG) << "The value of combined_splittable_inputs.size is: " << combined_splittable_inputs.size();
      Shapes inputs_partitions;
      size_t global_index = 0;
      for (auto &shape : inputs_shape) {
        Shape tmp_partition;
        for (size_t j = 0; j < shape.size(); ++j) {
          tmp_partition.push_back(combined_partitions[global_index]);
          global_index++;
        }
        inputs_partitions.push_back(tmp_partition);
      }
      StrategyPtr sp;
      if (PrepareStrategyBase(stage_id, dev_num, inputs_partitions, &sp) == SUCCESS) {
        sp_vector->push_back(sp);
      }
      return;
    } else {
      MS_LOG(DEBUG) << "The value of sp_vector size is " << sp_vector->size();
      if (combined_splittable_inputs[current_index] == 0) {
        combined_partitions.push_back(MIN_SLICE_NUM);
        recursive(current_index + 1, n / MIN_SLICE_NUM);
        combined_partitions.pop_back();
      } else if (combined_splittable_inputs[current_index] == 1) {
        for (uint64_t i = 1; i <= n; i *= 2) {
          if (n % i == 0 && LongToSize(combined_inputs_shape[current_index]) % i == 0) {
            combined_partitions.push_back(i);
            recursive(current_index + 1, n / i);
            combined_partitions.pop_back();
          }
        }
      }
    }
  };
  recursive(0, dev_num);
  if (sp_vector->empty()) {
    MS_LOG(EXCEPTION) << "No available strategy for current OperatorInfo.";
  }
  return SUCCESS;
}

// generate strategies for that have two inputs, and input0 or input1 maybe broadcast,
// and the corresponding dimensions that are not broadcast are all relevant dimensions
// such as: ([a, b, c, d], [a, b, c, d]) or ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
// or ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
// or ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
Status GenerateStrategiesWithBroadcast(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                       std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if ((inputs_shape.size() != 2) || (splittable_inputs.size() != 2)) {
    MS_LOG(ERROR) << "The inputs' size is wrong.";
    return FAILED;
  }

  if (inputs_shape[0] == inputs_shape[1]) {
    // element wise operation([a, b, c, d], [a, b, c, d]), so input0's strategy is equal to input1's strategy
    if (GenerateStrategiesForTwoEqualInputs(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForTwoEqualInputs success.";
  } else if (inputs_shape[0].empty() || inputs_shape[1].empty()) {
    // ([a, b, c, d], []) or ([], [a, b, c, d])
    if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "Generate strategies for scalar case failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "Generate strategies for scalar case success.";
  } else if (inputs_shape[0].size() > inputs_shape[1].size()) {
    // ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
    if (GenerateStrategiesForBroadcastRight(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForBroadcastRight failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForBroadcastRight success.";
  } else if (inputs_shape[0].size() < inputs_shape[1].size()) {
    // ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
    if (GenerateStrategiesForBroadcastLeft(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForBroadcastLeft failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForBroadcastLeft success.";
  } else {  // same size, but different value
    // ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
    if (GenerateStrategiesForBroadcastBoth(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForBroadcastBoth failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForBroadcastBoth success.";
  }
  return SUCCESS;
}

Status OperatorInfo::SetCostUnderStrategyBase(const StrategyPtr &strategy) {
  if (InitForCostModel(strategy) == FAILED) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Initialization under the strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Initialization under the strategy failed.";
    }
    return FAILED;
  }
  int64_t stage_id = strategy->GetInputStage();
  double computation_cost =
    operator_cost()->GetForwardComputationCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  double communication_cost = operator_cost()->GetCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  std::shared_ptr<Cost> result = std::make_shared<Cost>(computation_cost, communication_cost);
  result->communication_without_parameter_ =
    operator_cost()->GetForwardCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  result->communication_with_partial_para_ =
    result->communication_without_parameter_ +
    COST_MODEL_GAMMA * (communication_cost - result->communication_without_parameter_);

  // Breaking ties for preferring data parallelization
  BreakingTiesForPerferringDataParallel(strategy, result);
  // refine communication cost calculation for practice
  RefineForPracticalCost(result, false);
  result->communication_forward_ = result->communication_without_parameter_;

  std::shared_ptr<StrategyWithCost> swc =
    std::make_shared<StrategyWithCost>(strategy, inputs_tensor_info_, outputs_tensor_info_);
  swc->cost_list.push_back(result);
  strategy_cost_.emplace_back(swc);

  return SUCCESS;
}

// Keep at most (1.0 / epsilon) number of available strategies for each operator.
void OperatorInfo::ApproximateStrategies() {
  auto enable_approxi = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (!enable_approxi) {
    return;
  }
  MS_LOG(INFO) << "Approximating strategy-cost for: " << name_;
  auto epsilon = CostModelContext::GetInstance()->dp_algo_approxi_epsilon();
  auto target_num = static_cast<size_t>(std::ceil(1.0 / epsilon));
  if (strategy_cost_.size() <= target_num) {
    MS_LOG(INFO) << name_ << "'s strategy number is: " << strategy_cost_.size()
                 << ", no greater than target-num: " << target_num;
    return;
  }
  std::vector<std::shared_ptr<StrategyWithCost>> ret;
  auto &origin_stra_cost = strategy_cost_;
  auto alpha = CostModelContext::GetInstance()->costmodel_alpha();
  auto beta = CostModelContext::GetInstance()->costmodel_beta();
  // sort
  std::sort(
    origin_stra_cost.begin(), origin_stra_cost.end(),
    [&alpha, &beta](const std::shared_ptr<StrategyWithCost> &s1, const std::shared_ptr<StrategyWithCost> &s2) {
      if (alpha * s1->cost_list[0]->computation_cost_ + beta * s1->cost_list[0]->communication_with_partial_para_ <
          alpha * s2->cost_list[0]->computation_cost_ + beta * s2->cost_list[0]->communication_with_partial_para_) {
        return true;
      }
      return false;
    });
  size_t step_length = origin_stra_cost.size() / target_num;
  for (size_t i = 0; ret.size() < target_num && static_cast<size_t>(i * step_length) < origin_stra_cost.size(); ++i) {
    ret.push_back(origin_stra_cost[static_cast<size_t>(i * step_length)]);
  }

  strategy_cost_ = ret;
  is_strategy_cost_exact_ = false;
}

void OperatorInfo::ExactStrategiesAndRelatedEdges() {
  if (is_strategy_cost_exact()) {
    return;
  }
  ClearStrategyCost();
  if (GenerateStrategies(0) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Strategy search for Operator " << name() << " failed.";
    return;
  }
  SetIsStrategyCostExactTrue();
  // re-init the previous edges
  for (auto &prev_edge : prev_edges()) {
    if (prev_edge->InitEdgeCost() != SUCCESS) {
      MS_LOG(EXCEPTION) << "Edge: " << prev_edge->edge_name() << " cost init failed.";
    }
  }
  // re-init the successive edges
  for (auto &next_edge : succ_edges()) {
    if (next_edge->InitEdgeCost() != SUCCESS) {
      MS_LOG(EXCEPTION) << "Edge: " << next_edge->edge_name() << " cost init failed.";
    }
  }
}

int64_t OperatorInfo::ComputeOpAndPrevEdgeParameterInvolved() {
  if (is_output_parameter_involve_ != -1) {
    return is_output_parameter_involve_;
  }
  is_parameter_involve_ = is_parameter_;
  const auto &prev_edges = this->GetAlivePrevEdges();
  for (auto &p_edge : prev_edges) {
    auto input_index = p_edge->next_op_input_index();
    auto prev_op_para = p_edge->prev_operator()->ComputeOpAndPrevEdgeParameterInvolved();
    if (input_index >= is_parameter_involve_.size()) {
      MS_LOG(EXCEPTION) << name_ << " has input length: " << is_parameter_involve_.size()
                        << ", but got wrong input_index: " << input_index;
    }
    if (prev_op_para == 0) {
      is_parameter_involve_[input_index] = false;
    } else if (prev_op_para == 1) {
      is_parameter_involve_[input_index] = true;
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
    input_in_memory.emplace(std::make_pair(input_index, is_in_mem));
  }
  operator_cost()->CalculateInputsInMemory(input_in_memory);

  return is_output_parameter_involve_;
}

Status OperatorInfo::set_is_parameter(const std::vector<bool> &is_parameter) {
  if (is_parameter.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << "Is_parameter: " << is_parameter.size()
                  << " do not have the same number of inputs_shape_: " << inputs_shape_.size();
    return FAILED;
  }
  is_parameter_ = is_parameter;
  operator_cost()->set_is_parameter(is_parameter);
  return SUCCESS;
}

Status OperatorInfo::CalculateMemoryCost() {
  if (is_parameter_involve_.size() != is_parameter_.size()) {
    MS_LOG(ERROR) << "'is_parameter_' does not have the same number of input size of 'is_parameter_involve_'.";
    return FAILED;
  }
  // Set the memory cost in the 'strategy_cost_'
  for (auto &swc : strategy_cost_) {
    auto mem_cost = operator_cost()->GetMemoryCost(swc->inputs_ptr, swc->outputs_ptr);
    swc->cost_list[0]->memory_with_reuse_ = mem_cost;
  }
  return SUCCESS;
}

Status OperatorInfo::CalculateMemoryCostForInference() {
  // First, set the 'is_outputs_critical_' flag into OperatorCost.
  if (is_output_critical_ == -1) {
    MS_LOG(EXCEPTION) << "The critical flag is not set.";
    return FAILED;
  }
  operator_cost()->set_output_critical(is_output_critical_);
  // Set the memory cost in the 'strategy_cost_'
  for (auto &swc : strategy_cost_) {
    auto mem_cost = operator_cost()->GetMemoryCostForInference(swc->inputs_ptr, swc->outputs_ptr);
    swc->cost_list[0]->memory_with_reuse_ = mem_cost;
  }
  return SUCCESS;
}

Status OperatorInfo::CorrectMemoryCost(size_t input_index) {
  for (auto &swc : strategy_cost_) {
    double parameter_mem_cost = ListProduct(swc->inputs_ptr[input_index].slice_shape()) *
                                static_cast<double>(operator_cost()->inputs_type_lengths()[input_index]);
    swc->cost_list[0]->memory_with_reuse_ -= parameter_mem_cost;
    if (swc->cost_list[0]->memory_with_reuse_ < 0) {
      MS_LOG(ERROR) << "The memory cost after correction is: " << swc->cost_list[0]->memory_with_reuse_
                    << ", the parameter memory cost is: " << parameter_mem_cost;
      return FAILED;
    }
  }
  return SUCCESS;
}

int64_t ComputeRepeatDeviceNumByTensorMap(const Shape &dev_matrix_shape, const Shape &tensor_map) {
  int64_t ret = -1;

  // The number of repetitions is equal to the number of all devices divided by the number of devices use for
  // tensor map.
  int64_t device_num = std::accumulate(dev_matrix_shape.begin(), dev_matrix_shape.end(), 1, std::multiplies<int64_t>());
  for (auto &element : tensor_map) {
    // -1 means the corresponding dimension is not split.
    if (element == MAP_NONE) {
      continue;
    } else if ((element < 0) || (LongToSize(element) >= dev_matrix_shape.size())) {
      MS_LOG(ERROR) << "Invalid tensor map: " << ShapeToString(tensor_map) << ", the dev matrix shape is "
                    << ShapeToString(dev_matrix_shape);
      return ret;
    } else {
      size_t index = dev_matrix_shape.size() - LongToSize(element) - 1;
      if (dev_matrix_shape[index] <= 0) {
        MS_LOG(ERROR) << "Invalid dev matrix shape: " << ShapeToString(dev_matrix_shape);
        return ret;
      }
      device_num /= dev_matrix_shape[index];
    }
  }

  return (int64_t)device_num;
}

Status OperatorInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor map is empty.";
    return FAILED;
  }

  if (outputs_tensor_map_.size() > 1) {
    MS_LOG(ERROR) << name_ << ": The output size is " << outputs_tensor_map_.size()
                  << ", need to override this function ";
    return FAILED;
  }

  if (outputs_tensor_map_[0].empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[0]) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

// If the operator is used as a loss, a div node is inserted for the grad of all its inputs.
Status OperatorInfo::InferVirtualDivOps() {
  if (InferAsLossDivisor() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAsLossDivisor failed.";
    return FAILED;
  }

  if (as_loss_divisor_ <= 0) {
    MS_LOG(ERROR) << name_ << ": Invalid loss divisor: " << as_loss_divisor_;
    return FAILED;
  } else if (as_loss_divisor_ == 1) {
    MS_LOG(INFO) << name_ << ": The loss divisor is 1, no need to create virtual div op.";
    return SUCCESS;
  }

  virtual_div_op_.clear();
  // if loss is repeated calculation, insert div op
  Operator op = CreateVirtualDivOp(as_loss_divisor_);
  virtual_div_op_.push_back(op);
  return SUCCESS;
}

Status OperatorInfo::SetInputAndOutputTypeLength(const std::vector<size_t> &input_lengths,
                                                 const std::vector<size_t> &output_lengths) {
  if (input_lengths.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << "Input_lengths: " << input_lengths.size()
                  << " do not have the same number of inputs shape: " << inputs_shape_.size();
    return FAILED;
  }
  if (output_lengths.size() != outputs_shape_.size()) {
    MS_LOG(ERROR) << "Output_lengths: " << output_lengths.size()
                  << " do not have the same number of outputs shape: " << outputs_shape_.size();
    return FAILED;
  }
  inputs_type_lengths_ = input_lengths;
  outputs_type_lengths_ = output_lengths;
  operator_cost()->SetInputAndOutputTypeLength(input_lengths, output_lengths);
  return SUCCESS;
}

double OperatorInfo::GetOutputsTotalSize() {
  if (is_calculated_outputs_size_) {
    return outputs_total_size_;
  }
  if (outputs_type_lengths_.size() != outputs_shape_.size()) {
    MS_LOG(EXCEPTION) << "Output_lengths: " << outputs_type_lengths_.size()
                      << " do not have the same number of outputs shape: " << outputs_shape_.size();
  }
  double sum = 0.0;
  for (size_t i = 0; i < outputs_type_lengths_.size(); ++i) {
    auto size = std::accumulate(outputs_shape_[i].begin(), outputs_shape_[i].end(), static_cast<double>(1.0),
                                std::multiplies<double>());
    sum += size * static_cast<double>(outputs_type_lengths_[i]);
  }
  is_calculated_outputs_size_ = true;
  outputs_total_size_ = sum;
  return outputs_total_size_;
}

Status OperatorInfo::set_outputs_type(const std::vector<TypePtr> &outputs_type) {
  if (outputs_type.size() != outputs_shape_.size()) {
    MS_LOG(ERROR) << "Outputs type: " << outputs_type.size()
                  << " do not have the same number of outputs shape: " << outputs_shape_.size();
    return FAILED;
  }
  outputs_type_ = outputs_type;
  return SUCCESS;
}

void OperatorInfo::BreakingTiesForPerferringDataParallel(const StrategyPtr &stra, const CostPtr &cost) {
  if (!stra->GetInputDim().empty() && !stra->GetInputDim()[0].empty()) {
    if (stra->GetInputDim()[0][0] == stage_device_size_) {
      if (cost->computation_cost_ > 1.0) {
        cost->computation_cost_ -= 1.0;
      }
      if (cost->communication_cost_ > 1.0) {
        cost->communication_cost_ -= 1.0;
      }
      if (cost->communication_with_partial_para_ > 1.0) {
        cost->communication_with_partial_para_ -= 1.0;
      }
      if (cost->communication_without_parameter_ > 1.0) {
        cost->communication_without_parameter_ -= 1.0;
      }
    }
  }
}

double OperatorInfo::GetForwardMemoryCostFromCNode() {
  return operator_cost()->GetForwardComputationCost(inputs_tensor_info_, outputs_tensor_info_, 0);
}

void OperatorInfo::CheckSelectedStrategy(const StrategyPtr &s_strategy) {
  MS_EXCEPTION_IF_NULL(s_strategy);
  if (!s_strategy->IsEqual(selected_strategy_)) {
    MS_LOG(INFO) << name() << "'s strategy may cause suboptimal, the determined strategy:";
    PrintStrategy(selected_strategy_);
    MS_LOG(INFO) << "The minimal strategy:";
    PrintStrategy(s_strategy);
  }
}

void OperatorInfo::SetStrategyCost(const std::vector<std::shared_ptr<StrategyWithCost>> &stra_cost) {
  strategy_cost_ = stra_cost;
}
}  // namespace parallel
}  // namespace mindspore
