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

#include "parallel/ops_info/reshape_info.h"

#include <memory>
#include <vector>

#include "parallel/device_manager.h"
#include "parallel/device_matrix.h"
#include "parallel/step_parallel.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status ReshapeInfo::CheckStrategy(const StrategyPtr& strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    }
    return FAILED;
  }

  size_t strategy_size = strategy->GetInputNumber();
  if (strategy_size != 1) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Invalid strategy size " << strategy_size;
    } else {
      MS_LOG(ERROR) << name_ << ": Invalid strategy size " << strategy_size;
    }
    return FAILED;
  }
  std::vector<Dimensions> stra = strategy->GetInputDim();
  for (size_t i = 0; i < strategy_size; ++i) {
    Shape sub_strategy = stra.at(i);
    size_t strategy_len = sub_strategy.size();
    bool flag = false;
    for (size_t j = 0; j < strategy_len; ++j) {
      int32_t strategy_value = sub_strategy.at(j);
      if (strategy_value > 1) {
        if (flag) {
          if (is_auto_parallel_) {
            MS_LOG(DEBUG) << name_ << ": Only support batch parallel strategy.";
          } else {
            MS_LOG(ERROR) << name_ << ": Only support batch parallel strategy.";
          }
          return FAILED;
        }
        flag = true;
      }
    }
  }
  return SUCCESS;
}

/*
 * support parallel degree smaller than device number, set the duplicate device dimension to the first dimension of
 * device matrix
 * only support batch parallel reshape operator in ReID (batch parallel degree can be smaller than device number)
 */
Status ReshapeInfo::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  input_strategy_ = stra.at(0);
  dev_matrix_shape_.push_back(input_strategy_[0]);
  return SUCCESS;
}

/*
 * there is no Parameter for Reshape Primitive, so no need to do allreduce
 */
Status ReshapeInfo::InferMirrorOps() {
  mirror_ops_.clear();
  Shape input_tensor_map = input_layout_.tensor_map().array();
  std::vector<Group> input_group;
  if (CreateGroupByTensorMap(input_tensor_map, &input_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer MirrorOps failed.";
    return FAILED;
  }

  OperatorVector op_for_input;
  if (input_group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror ops is empty.";
    return SUCCESS;
  }
  if (!input_group.empty()) {
    op_for_input = CreateMirrorOps(input_group[0].name(), input_group[0].GetDevNum());
    std::string group_name = input_group[0].name();
    MS_LOG(INFO) << name_ << ": Create the mirror ops for input_a success, group is " << group_name;
  }
  mirror_ops_.push_back(op_for_input);
  OperatorVector op_for_input_empty;
  mirror_ops_.push_back(op_for_input_empty);

  return SUCCESS;
}

/*
 * there is no reduction dimension for forward computation of Reshape Primitive, so no need to do allreduce
 */
Status ReshapeInfo::InferForwardCommunication() { return SUCCESS; }

/*
 * get shape input of Reshape Primitive
 * the result is saved in parameter_input_v_
 * not support -1
 */
Status ReshapeInfo::GetParameterInput() {
  if (input_value_[1] == nullptr) {
    MS_LOG(ERROR) << name_ << ": input_value_[1] is nullptr.";
    return FAILED;
  }
  std::vector<ValuePtr> elements;
  ValueTuplePtr dim_tuple = input_value_[1]->cast<ValueTuplePtr>();
  if (dim_tuple == nullptr) {
    MS_LOG(ERROR) << name_ << ": Input_value_[1] must be ValueTuplePtr.";
    return FAILED;
  }
  elements = dim_tuple->value();
  if (elements.size() != outputs_shape_[0].size()) {
    MS_LOG(ERROR) << name_ << ": Elements size must equal to outputs shape[0] size.";
    return FAILED;
  }

  for (auto& element : elements) {
    MS_EXCEPTION_IF_NULL(element);
    if (element->isa<Int32Imm>()) {
      int32_t axis = element->cast<Int32ImmPtr>()->value();
      parameter_input_v_.push_back(axis);
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis must be int32.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ReshapeInfo::ComputeReplaceOp() {
  RankList dev_list = global_device_list();
  TensorRedistribution tensor_redistribution(true, true);
  if (tensor_redistribution.Init(input_layout_, output_layout_, dev_list) == FAILED) {
    MS_LOG(ERROR) << name_ << ": tensor_redistribution init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": input " << input_layout_.ToString();
  MS_LOG(INFO) << name_ << ": output " << output_layout_.ToString();
  MS_LOG(INFO) << name_ << ": dev_list " << dev_list.size();
  RedistributionOpListPtr redistribution_oplist_ptr = tensor_redistribution.InferTensorRedistributionOperatorList();
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(ERROR) << name_ << "InferTensorRedistribution failed.";
    return FAILED;
  }
  replace_op_ = redistribution_oplist_ptr->first;
  replace_op_info_ = redistribution_oplist_ptr->second;
  MS_LOG(INFO) << name_ << ": replace op size = " << replace_op_.size();
  return SUCCESS;
}

/*
 * the first dimension of input tensor map and output tensor map is set to the last dimension of device arrangement,
 * all other dimension is set to None
 * only support batch parallel reshape operator in ReID (batch parallel degree can be smaller than device number)
 */
Status ReshapeInfo::InferTensorMap() {
  if ((inputs_shape_.size() != 1) || (outputs_shape_.size() != 1)) {
    MS_LOG(ERROR) << name_ << ": inputs shape and outputs shape size must be 1. inputs shape and outputs shape are "
                  << inputs_shape_.size() << " and " << outputs_shape_.size();
    return FAILED;
  }

  std::vector<int32_t> tensor_map_index_input;
  tensor_map_index_input.push_back(0);

  for (size_t j = 1; j < inputs_shape_[0].size(); ++j) {
    tensor_map_index_input.push_back(MAP_NONE);
  }
  inputs_tensor_map_.push_back(tensor_map_index_input);

  std::vector<int32_t> tensor_map_index_output;
  tensor_map_index_output.push_back(0);

  for (size_t j = 1; j < outputs_shape_[0].size(); ++j) {
    tensor_map_index_output.push_back(MAP_NONE);
  }
  outputs_tensor_map_.push_back(tensor_map_index_output);
  return SUCCESS;
}

/*
 * the output tensor strategy is the same as input tensor strategy
 * only support batch parallel reshape operator in ReID (batch parallel degree can be smaller than device number)
 */
Strategys ReshapeInfo::GetOutputsStrategy() {
  Strategys outputs_strategy;
  std::vector<int32_t> strategy;
  strategy.push_back(input_strategy_[0]);
  for (size_t j = 1; j < outputs_shape_[0].size(); ++j) {
    strategy.push_back(1);
  }
  outputs_strategy.push_back(strategy);
  return outputs_strategy;
}

Status ReshapeInfo::InferTensorLayout(TensorLayouts* inputs_layout, TensorLayouts* outputs_layout) {
  if (inputs_layout == nullptr || outputs_layout == nullptr) {
    MS_LOG(ERROR) << name_ << ": InferTensorLayout: the layout is null.";
    return FAILED;
  }
  Arrangement dev_matrix;
  Status status = dev_matrix.Init(dev_matrix_shape_);
  if (status != Status::SUCCESS) {
    return status;
  }
  // infer input tensor info
  Shape shape_array_in = inputs_shape_.at(0);
  TensorMap tensor_map_array_in = inputs_tensor_map_.at(0);
  TensorLayout tensor_layout_in;
  Map tensor_map_in;
  status = tensor_map_in.Init(tensor_map_array_in);
  if (status != Status::SUCCESS) {
    return status;
  }
  Arrangement shape_in;
  status = shape_in.Init(shape_array_in);
  if (status != Status::SUCCESS) {
    return status;
  }
  (void)tensor_layout_in.Init(dev_matrix, tensor_map_in, shape_in);
  inputs_layout->push_back(tensor_layout_in);
  // infer output tensor info
  Shape shape_array_out = outputs_shape_.at(0);

  TensorMap tensor_map_array_out = outputs_tensor_map_.at(0);
  TensorLayout tensor_layout_out;
  Map tensor_map_out;
  status = tensor_map_out.Init(tensor_map_array_out);
  if (status != Status::SUCCESS) {
    return status;
  }
  Arrangement shape_out;
  status = shape_out.Init(shape_array_out);
  if (status != Status::SUCCESS) {
    return status;
  }
  (void)tensor_layout_out.Init(dev_matrix, tensor_map_out, shape_out);
  outputs_layout->push_back(tensor_layout_out);

  input_layout_ = tensor_layout_in;
  output_layout_ = tensor_layout_out;
  return SUCCESS;
}

Status ReshapeInfo::InferTensorInfo() {
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Strategys outputs_strategy = GetOutputsStrategy();
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }

  TensorLayouts inputs_layout, outputs_layout;
  if (InferTensorLayout(&inputs_layout, &outputs_layout) != SUCCESS) {
    return FAILED;
  }
  TensorLayout tensor_layout_in = inputs_layout.at(0);
  TensorLayout tensor_layout_out = outputs_layout.at(0);
  Shape shape_array_in = inputs_shape_.at(0);
  Shape slice_shape_in = inputs_slice_shape.at(0);
  Shape shape_array_out = outputs_shape_.at(0);
  Shape slice_shape_out = outputs_slice_shape.at(0);
  TensorInfo tensor_info_in(tensor_layout_in, shape_array_in, slice_shape_in);
  TensorInfo tensor_info_out(tensor_layout_out, shape_array_out, slice_shape_out);
  inputs_tensor_info_.push_back(tensor_info_in);
  outputs_tensor_info_.push_back(tensor_info_out);
  return SUCCESS;
}

void ReshapeInfo::InferTensorInfoByLayout() {
  TensorInfo tensor_info_in(input_layout_);
  TensorInfo tensor_info_out(output_layout_);
  inputs_tensor_info_.push_back(tensor_info_in);
  outputs_tensor_info_.push_back(tensor_info_out);
}

/*
 * compute parameter_input_v_ during this method
 */
Status ReshapeInfo::GetAttrs() { return GetParameterInput(); }

void ReshapeInfo::device_number(const StrategyPtr& strategy) {
  int32_t stage = 0;
  if (strategy != nullptr) {
    stage = strategy->GetInputStage();
  }
  CheckGlobalDeviceManager();
  global_device_list_ = g_device_manager->GetDeviceListByStageId(stage);
  dev_num_ = SizeToInt(global_device_list_.size());
  MS_ASSERT(dev_num_ > 0);
}

Status ReshapeInfo::InferDefaultLayout(const Shape& shape, TensorLayout* const layout) {
  std::vector<int32_t> tensor_map_index;
  for (size_t i = 0; i < shape.size(); i++) {
    tensor_map_index.push_back(MAP_NONE);
  }
  Status status = layout->InitFromVector({dev_num_}, tensor_map_index, shape);
  if (status != Status::SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferDefaultLayout failed.";
    return status;
  }
  return Status::SUCCESS;
}

Status ReshapeInfo::Init(const StrategyPtr& strategy) {
  ResetQueueMember();
  device_number(strategy);
  if (strategy) {
    if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Init failed.";
      return FAILED;
    }
  } else {
    if (!input_layout_set_flag_) {
      MS_ASSERT(inputs_shape_.size() == 1);
      Status status = InferDefaultLayout(inputs_shape_.at(0), &input_layout_);
      if (status != SUCCESS) {
        MS_LOG(ERROR) << name_ << ": infer input default layout failed.";
        return status;
      }
    }
    if (!output_layout_set_flag_) {
      MS_ASSERT(output_layout_.size() == 1);
      Status status = InferDefaultLayout(outputs_shape_.at(0), &output_layout_);
      if (status != SUCCESS) {
        MS_LOG(ERROR) << name_ << ": infer output default layout failed.";
        return status;
      }
    }
    inputs_tensor_map_.push_back(input_layout_.tensor_map().array());
    outputs_tensor_map_.push_back(output_layout_.tensor_map().array());
    InferTensorInfoByLayout();
    // change dev_matrix_shape_ to input_layout_ device_arrangement before InferMirrorOps
    dev_matrix_shape_ = input_layout_.device_arrangement().array();
    if (InferMirrorOps() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": InferMirrorOps failed.";
      return FAILED;
    }
    // change dev_matrix_shape_ to output_layout_ device_arrangement before InferVirtualDivOps
    dev_matrix_shape_ = output_layout_.device_arrangement().array();
    if (InferVirtualDivOps() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": InferVirtualDivOps failed.";
      return FAILED;
    }
  }
  Status status = ComputeReplaceOp();
  if (status != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": ComputeReplaceOp failed.";
    return status;
  }
  return SUCCESS;
}

Status ReshapeInfo::InitForCostModel(const StrategyPtr& strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    }
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status ReshapeInfo::SetCostUnderStrategy(const mindspore::parallel::StrategyPtr& strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Set cost under strategy failed.";
    }
    return FAILED;
  }

  return SUCCESS;
}

Status ReshapeInfo::GenerateStrategies(int32_t stage_id) {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GetAttrs failed.";
    return FAILED;
  }
  if ((inputs_shape_.size() != 1) || (outputs_shape_.size() != 1)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size or outputs shape size is wrong, " << inputs_shape_.size() << ", "
                  << outputs_shape_.size();
    return FAILED;
  }
  is_auto_parallel_ = true;
  Shape input0_split(inputs_shape_[0].size(), 0);
  input0_split[0] = 1;
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateStrategiesForIndependentInputs failed.";
    return FAILED;
  }
  size_t success = 0;
  for (auto& sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
