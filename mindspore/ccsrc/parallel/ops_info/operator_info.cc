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

#include "parallel/ops_info/operator_info.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ir/dtype.h"
#include "ir/meta_tensor.h"
#include "ir/value.h"
#include "parallel/auto_parallel/edge_costmodel.h"
#include "parallel/auto_parallel/graph_costmodel.h"
#include "parallel/context.h"
#include "utils/context/ms_context.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status CheckStrategyValue(const StrategyPtr& strategy, const Shapes& inputs_shape, bool is_auto_parallel) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << "The strategy is null.";
    return FAILED;
  }

  size_t strategy_size = strategy->GetInputNumber();
  size_t inputs_shape_size = inputs_shape.size();
  if (strategy_size != inputs_shape_size) {
    if (is_auto_parallel) {
      MS_LOG(DEBUG) << "Strategy size: " << strategy_size << " is not equal to inputs size: " << inputs_shape_size;
    } else {
      MS_LOG(ERROR) << "Strategy size: " << strategy_size << " is not equal to inputs size: " << inputs_shape_size;
    }
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  for (size_t i = 0; i < strategy_size; ++i) {
    Shape sub_strategy = stra.at(i);
    Shape sub_input_shape = inputs_shape.at(i);
    size_t strategy_len = sub_strategy.size();
    size_t inputs_len = sub_input_shape.size();
    if (strategy_len != inputs_len) {
      if (is_auto_parallel) {
        MS_LOG(DEBUG) << "Strategy len: " << strategy_len << " is not equal to inputs len: " << inputs_len
                      << ", index: " << i;
      } else {
        MS_LOG(ERROR) << "Strategy len: " << strategy_len << " is not equal to inputs len: " << inputs_len
                      << ", index: " << i;
      }
      return FAILED;
    }

    for (size_t j = 0; j < strategy_len; ++j) {
      int32_t strategy_value = sub_strategy.at(j);
      if (strategy_value < MIN_SLICE_NUM) {
        if (is_auto_parallel) {
          MS_LOG(DEBUG) << "Invalid strategy value: " << strategy_value;
        } else {
          MS_LOG(ERROR) << "Invalid strategy value: " << strategy_value;
        }
        return FAILED;
      }

      if ((IntToUint(strategy_value) & IntToUint(strategy_value - 1)) != 0) {
        if (is_auto_parallel) {
          MS_LOG(DEBUG) << "Invalid Strategy value it is not the power of 2, " << strategy_value;
        } else {
          MS_LOG(ERROR) << "Invalid Strategy value it is not the power of 2, " << strategy_value;
        }
        return FAILED;
      }

      int32_t shape_value = sub_input_shape.at(j);
      if ((shape_value % strategy_value) != 0) {
        if (is_auto_parallel) {
          MS_LOG(DEBUG) << "Shape " << shape_value << " cannot be divisible by strategy " << strategy_value;
        } else {
          MS_LOG(ERROR) << "Shape " << shape_value << " cannot be divisible by strategy " << strategy_value;
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
  replace_op_.clear();
  replace_op_info_.clear();
  virtual_div_op_.clear();
  global_device_list_.clear();
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

void OperatorInfo::SetDeviceListByStrategy() {
  int32_t stage = strategy_->GetInputStage();
  CheckGlobalDeviceManager();
  global_device_list_ = g_device_manager->GetDeviceListByStageId(stage);
}

Status OperatorInfo::InferRepeatedCalcInfo() {
  int32_t g_dev_list_size = SizeToInt(global_device_list_.size());
  int32_t dev_matrix_size =
    std::accumulate(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), 1, std::multiplies<int>());
  if (dev_matrix_size == 0) {
    MS_LOG(ERROR) << name_ << ": The dev matrix size is 0";
    return FAILED;
  }

  if (g_dev_list_size == dev_matrix_size) {
    repeated_calc_num_ = 1;
  } else if (g_dev_list_size % dev_matrix_size == 0) {
    repeated_calc_num_ = g_dev_list_size / dev_matrix_size;
  } else {
    MS_LOG(ERROR) << name_ << ": Dev list size " << g_dev_list_size << " can not be divisible by dev matrix size "
                  << dev_matrix_size;
    return FAILED;
  }

  CheckGlobalDeviceManager();
  int32_t rank = g_device_manager->global_rank();
  int32_t stage = strategy_->GetInputStage();
  local_device_list_ = g_device_manager->global_device_list(stage, rank, repeated_calc_num_);

  return SUCCESS;
}

// if repeated calculation, need to set the repeated_calc_num as the first dimension of dev-matrix,
// only use for infer tensor layout
void OperatorInfo::SetRepeatedCalcDevMatrix() {
  if (repeated_calc_num_ <= 1) {
    return;
  }

  (void)dev_matrix_shape_.insert(dev_matrix_shape_.begin(), repeated_calc_num_);
}

// use for loss repeated calculation
Operator CreateVirtualDivOp(int32_t div_num) {
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
Operator CreateAllReduceOp(const std::string& reduce_op, const std::string& group) {
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

// use for get tensor slice
Operator CreateGetTensorSliceOp(const TensorLayout& tensor_layout) {
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

OperatorVector CreateMirrorOps(const std::string& group_name, size_t dev_num) {
  if ((dev_num == 0) || (dev_num == 1)) {
    MS_LOG(EXCEPTION) << "Invalid dev num: " << dev_num;
  }
  OperatorVector op_for_weight;
  bool mean_flag = ParallelContext::GetInstance()->mirror_mean();

  OperatorName operator_name = MIRROR_OPERATOR;
  ValuePtr attr0_value = MakeValue(group_name);
  ValuePtr attr1_value = MakeValue(dev_num);
  ValuePtr attr2_value = MakeValue(mean_flag);

  Attr attr0 = std::make_pair(GROUP, attr0_value);
  Attr attr1 = std::make_pair(DEV_NUM, attr1_value);
  Attr attr2 = std::make_pair(MEAN_FLAG, attr2_value);

  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);
  operator_attrs.push_back(attr2);

  OperatorParams operator_param;
  OperatorArgs operator_args = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_args);

  op_for_weight.push_back(op);
  MS_LOG(INFO) << "The group name is " << group_name << ", the dev num is " << dev_num << ", the mean flag is "
               << mean_flag;
  return op_for_weight;
}

Status OperatorInfo::CreateGroupByTensorMap(const Shape& tensor_map, std::vector<Group>* group) {
  if (group == nullptr) {
    MS_LOG(ERROR) << "The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int32_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, global_device_list_, dev_matrix_shape_);
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

Status OperatorInfo::CreateGroupByDim(size_t axis, std::vector<Group>* group) {
  if (group == nullptr) {
    MS_LOG(ERROR) << "The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int32_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, global_device_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesAlongDim(SizeToUint(axis), &group_devices) != SUCCESS) {
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

Shape GetSliceShape(const Shape& tensor_shape, const Dimensions& strategy) {
  Shape slice_shape;
  if (std::any_of(strategy.begin(), strategy.end(), [](int32_t value) { return value <= 0; })) {
    MS_LOG(ERROR) << "Invalid strategy: " << ShapeToString(strategy) << ", the element is less than or equal to 0";
    return slice_shape;
  }
  for (size_t i = 0; i < strategy.size(); ++i) {
    slice_shape.push_back(tensor_shape.at(i) / strategy.at(i));
  }
  return slice_shape;
}

Status InferSliceShapeByStrategy(const Strategys& strategys, const Shapes& shapes, Shapes* slice_shapes) {
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

Status OperatorInfo::InferSliceShape(const Strategys& inputs_strategy, const Strategys& outputs_strategy,
                                     Shapes* inputs_slice_shape, Shapes* outputs_slice_shape) {
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
Status OperatorInfo::InitForCostModelWithAutoRepeatCalc(const StrategyPtr& strategy) {
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
  SetDeviceListByStrategy();

  if (InferDevMatrixShape() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferDevMatrixShape failed.";
    return FAILED;
  }

  used_devices_ = std::accumulate(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), 1, std::multiplies<int32_t>());

  // must be after InferDevMatrixShape
  if (InferRepeatedCalcInfo() != SUCCESS) {
    MS_LOG(ERROR) << ": InferRepeatedCalcInfo failed.";
    return FAILED;
  }

  // if repeated calculation, need to set the repeated_calc_num as the first dimension of dev-matrix for layout
  SetRepeatedCalcDevMatrix();

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

// method1: manually insert repeated_calculation_num for dev_matrix_shape in InferDevMatrixShape
Status OperatorInfo::InitForCostModelWithManualRepeatCalc(const StrategyPtr& strategy) {
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
  SetDeviceListByStrategy();

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

Status OperatorInfo::InitWithAutoRepeatCalc(const StrategyPtr& strategy) {
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

Status OperatorInfo::InitWithManualRepeatCalc(const StrategyPtr& strategy) {
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
  for (auto& edge : succ_edges_) {
    if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(RELU) != std::string::npos)) {
      ret.push_back(edge);
    }
  }
  for (auto& edge : succ_edges_) {
    if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(RELU) == std::string::npos)) {
      ret.push_back(edge);
    }
  }
  return ret;
}

std::vector<std::shared_ptr<Edge>> OperatorInfo::GetAlivePrevEdges() {
  std::vector<std::shared_ptr<Edge>> ret;
  for (auto& edge : prev_edges_) {
    if (edge->prev_operator()->is_alive()) {
      ret.push_back(edge);
    }
  }
  return ret;
}

void OperatorInfo::ReplacePreEdge(const std::shared_ptr<OperatorInfo>& op, const std::shared_ptr<Edge>& new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplacePreEdge: the op is null.";
    return;
  }
  for (auto& edge : prev_edges_) {
    if (edge->prev_operator() == op) {
      edge = new_edge;
      return;
    }
  }
  MS_LOG(EXCEPTION) << name_ << ": Replace edge failed: no edge has been replaced";
}

void OperatorInfo::ReplaceSuccEdge(const std::shared_ptr<OperatorInfo>& op, const std::shared_ptr<Edge>& new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplaceSuccEdge: the op is null.";
    return;
  }
  for (auto& edge : succ_edges_) {
    if (edge->next_operator() == op) {
      edge = new_edge;
      return;
    }
  }
  MS_LOG(EXCEPTION) << name_ << ": Replace edge failed: no edge has been replaced";
}

void OperatorInfo::ReplacePreEdges(const std::shared_ptr<OperatorInfo>& op, const std::shared_ptr<Edge>& new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplacePreEdges: the op is null.";
    return;
  }
  std::vector<std::shared_ptr<Edge>> new_pre_edges;
  for (auto& edge : prev_edges_) {
    if (edge->prev_operator() != op) {
      new_pre_edges.push_back(edge);
    }
  }
  new_pre_edges.push_back(new_edge);
  prev_edges_ = new_pre_edges;
}

void OperatorInfo::ReplaceSuccEdges(const std::shared_ptr<OperatorInfo>& op, const std::shared_ptr<Edge>& new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplaceSuccEdges: the op is null";
    return;
  }
  std::vector<std::shared_ptr<Edge>> new_succ_edges;
  for (auto& edge : succ_edges_) {
    if (edge->next_operator() != op) {
      new_succ_edges.push_back(edge);
    }
  }
  new_succ_edges.push_back(new_edge);
  succ_edges_ = new_succ_edges;
}

std::shared_ptr<std::vector<std::vector<int32_t>>> GenerateBatchStrategiesBySplitFlag(
  const Shapes& shapes, const std::vector<bool>& split_flag_list) {
  if (shapes.size() != split_flag_list.size()) {
    MS_LOG(ERROR) << "Split_flag_list do not have the same size as inputs shape, " << split_flag_list.size() << " : "
                  << shapes.size();
    return nullptr;
  }
  CheckGlobalDeviceManager();
  int32_t dev_num = SizeToInt(g_device_manager->GetDeviceListByStageId(0).size());
  std::vector<std::vector<int32_t>> strategy_v;
  for (size_t i = 0; i != shapes.size(); i++) {
    if (shapes[i].empty()) {
      MS_LOG(INFO) << "Elements of shapes is empty.";
      std::vector<int32_t> empty_element;
      strategy_v.push_back(empty_element);
    } else {
      std::vector<int32_t> element(shapes[i].size(), 1);
      if (split_flag_list[i]) {
        element[0] = dev_num;
      }
      strategy_v.push_back(element);
    }
  }
  return std::make_shared<std::vector<std::vector<int32_t>>>(strategy_v);
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

// This is a common method for checking whether the generated stragegy has the correct number of devuces.
Status PrepareStrategyBase(int32_t stage_id, size_t dev_num, const Shapes& inputs_partitions, StrategyPtr* const sp) {
  if (sp == nullptr) {
    MS_LOG(ERROR) << "The strategy is null.";
    return FAILED;
  }
  int32_t product = 1;

  for (auto& input_partition : inputs_partitions) {
    product *= std::accumulate(input_partition.begin(), input_partition.end(), 1, std::multiplies<int>());
  }
  if (NOT_FULLY_USE_DEVICES) {
    if (IntToSize(product) > dev_num) {
      return FAILED;
    }
  } else {
    if ((product != 1) && (IntToSize(product) != dev_num)) {
      return FAILED;
    }
  }
  std::vector<Dimensions> stras(inputs_partitions);
  (*sp) = std::make_shared<Strategy>(stage_id, stras);
  return SUCCESS;
}

std::shared_ptr<std::vector<std::vector<int32_t>>> OperatorInfo::GenerateBatchStrategies() {
  ComputeBatchSplitFlagList();
  return GenerateBatchStrategiesBySplitFlag(inputs_shape_, split_flag_list_);
}

void PrintStrategy(const StrategyPtr& strategy) {
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
Status GenerateStrategiesForTwoEqualInputs(int32_t stage_id, const Shapes& inputs_shape,
                                           const Shapes& splittable_inputs, std::vector<StrategyPtr>* const sp_vector) {
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

  for (auto& sp : *sp_vector) {
    sp->ExpandInputDimFromOneToTwo();
  }

  return SUCCESS;
}

// generate strategies for that input0 and input1 have relevant dimensions, and input0 needs to broadcast
// such as: ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
Status GenerateStrategiesForBroadcastLeft(int32_t stage_id, const Shapes& inputs_shape, const Shapes& splittable_inputs,
                                          std::vector<StrategyPtr>* const sp_vector) {
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
  for (auto& sp : *sp_vector) {
    std::vector<Dimensions> tmp_strategy;
    Dimensions input0_strategy = sp->GetInputDim()[0];
    size_t size_diff = inputs_shape[1].size() - inputs_shape[0].size();

    // erase the unnecessary part
    (void)input0_strategy.erase(input0_strategy.begin(),
                                input0_strategy.begin() + static_cast<different_type>(size_diff));

    // handel the case likes ([1, c, d], [a, b, c, d])
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
Status GenerateStrategiesForBroadcastRight(int32_t stage_id, const Shapes& inputs_shape,
                                           const Shapes& splittable_inputs, std::vector<StrategyPtr>* const sp_vector) {
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
  for (auto& sp : *sp_vector) {
    std::vector<Dimensions> tmp_strategy;
    tmp_strategy.push_back(sp->GetInputDim()[0]);  // input0

    Dimensions input1_strategy = sp->GetInputDim()[1];
    size_t size_diff = inputs_shape[0].size() - inputs_shape[1].size();

    // erase the unnecessary part
    (void)input1_strategy.erase(input1_strategy.begin(),
                                input1_strategy.begin() + static_cast<different_type>(size_diff));

    // handel the case likes ([a, b, c, d], [1, c, d])
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
Status GenerateStrategiesForBroadcastBoth(int32_t stage_id, const Shapes& inputs_shape, const Shapes& splittable_inputs,
                                          std::vector<StrategyPtr>* const sp_vector) {
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
  for (auto& sp : *sp_vector) {
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
Status GenerateStrategiesForIndependentInputs(int32_t stage_id, const Shapes& inputs_shape,
                                              const Shapes& splittable_inputs,
                                              std::vector<StrategyPtr>* const sp_vector) {
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
  std::function<void(uint32_t, size_t)> recursive = [&stage_id, &dev_num, &sp_vector, &combined_inputs_shape,
                                                     &combined_splittable_inputs, &combined_partitions, &recursive,
                                                     &inputs_shape](uint32_t current_index, size_t n) {
    if (current_index == combined_inputs_shape.size()) {
      MS_LOG(DEBUG) << "The value of combined_splittable_inputs.size is: " << combined_splittable_inputs.size();
      Shapes inputs_partitions;
      size_t global_index = 0;
      for (auto& shape : inputs_shape) {
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
        for (uint32_t i = 1; i <= n; i *= 2) {
          if (n % i == 0 && IntToSize(combined_inputs_shape[current_index]) % i == 0) {
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
Status GenerateStrategiesWithBroadcast(int32_t stage_id, const Shapes& inputs_shape, const Shapes& splittable_inputs,
                                       std::vector<StrategyPtr>* const sp_vector) {
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

Status OperatorInfo::SetCostUnderStrategyBase(const StrategyPtr& strategy) {
  if (InitForCostModel(strategy) == FAILED) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Initialization under the strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Initialization under the strategy failed.";
    }
    return FAILED;
  }
  int32_t stage_id = strategy->GetInputStage();
  double computation_cost = cost()->GetForwardComputationCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  double communication_cost = cost()->GetCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  std::shared_ptr<Cost> result = std::make_shared<Cost>(computation_cost, communication_cost);
  result->communication_without_parameter_ =
    cost()->GetForwardCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  result->communication_with_partial_para_ =
    result->communication_without_parameter_ +
    COST_MODEL_GAMMA * (communication_cost - result->communication_without_parameter_);

  // Breaking ties for preferring data parallelization
  BreakingTiesForPerferringDataParallel(strategy, result);
  // refine communication cost calculation for practice
  RefineForPracticalCost(result, false);

  std::shared_ptr<StrategyWithCost> swc =
    std::make_shared<StrategyWithCost>(strategy, inputs_tensor_info_, outputs_tensor_info_);
  swc->cost_list.push_back(result);
  strategy_cost_.emplace_back(swc);

  return SUCCESS;
}

int OperatorInfo::ComputeOpAndPrevEdgeParameterInvolved() {
  if (is_output_parameter_involve_ != -1) {
    return is_output_parameter_involve_;
  }
  is_parameter_involve_ = is_parameter_;
  const auto& prev_edges = this->GetAlivePrevEdges();
  for (auto& p_edge : prev_edges) {
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

  return is_output_parameter_involve_;
}

Status OperatorInfo::set_is_parameter(const std::vector<bool>& is_parameter) {
  if (is_parameter.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << "Is_parameter: " << is_parameter.size()
                  << " do not have the same number of inputs_shape_: " << inputs_shape_.size();
    return FAILED;
  }
  is_parameter_ = is_parameter;
  cost()->set_is_parameter(is_parameter);
  return SUCCESS;
}

int32_t ComputeRepeatDeviceNumByTensorMap(const Shape& dev_matrix_shape, const Shape& tensor_map) {
  int32_t ret = -1;

  // The number of repetitions is equal to the number of all devices divided by the number of devices use for
  // tensor map.
  int32_t device_num = std::accumulate(dev_matrix_shape.begin(), dev_matrix_shape.end(), 1, std::multiplies<int>());
  for (auto& element : tensor_map) {
    // -1 means the corresponding dimension is not split.
    if (element == MAP_NONE) {
      continue;
    } else if ((element < 0) || (IntToSize(element) >= dev_matrix_shape.size())) {
      MS_LOG(ERROR) << "Invalid tensor map: " << ShapeToString(tensor_map) << ", the dev matrix shape is "
                    << ShapeToString(dev_matrix_shape);
      return ret;
    } else {
      size_t index = dev_matrix_shape.size() - IntToSize(element) - 1;
      if (dev_matrix_shape[index] <= 0) {
        MS_LOG(ERROR) << "Invalid dev matrix shape: " << ShapeToString(dev_matrix_shape);
        return ret;
      }
      device_num /= dev_matrix_shape[index];
    }
  }

  return device_num;
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
    as_loss_divisor_ = SizeToInt(global_device_list_.size());
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

Status OperatorInfo::SetInputAndOutputTypeLength(const std::vector<size_t>& input_lengths,
                                                 const std::vector<size_t>& output_lengths) {
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
  cost()->SetInputAndOutputTypeLength(input_lengths, output_lengths);
  return SUCCESS;
}

void OperatorInfo::BreakingTiesForPerferringDataParallel(const StrategyPtr& stra, const CostPtr& cost) {
  if (!stra->GetInputDim().empty() && !stra->GetInputDim()[0].empty()) {
    CheckGlobalDeviceManager();
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stra->GetInputStage()).size();
    if (IntToSize(stra->GetInputDim()[0][0]) == total_device_num) {
      cost->computation_cost_ -= 1.0;
      cost->communication_cost_ -= 1.0;
      cost->communication_with_partial_para_ -= 1.0;
      cost->communication_without_parameter_ -= 1.0;
    }
  }
}

double OperatorInfo::GetForwardMemoryCostFromCNode() {
  return cost()->GetForwardComputationCost(inputs_tensor_info_, outputs_tensor_info_, 0);
}

}  // namespace parallel
}  // namespace mindspore
