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

#include "frontend/parallel/ops_info/reshape_info.h"

#include <memory>
#include <vector>
#include <utility>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "include/common/utils/convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status ReshapeInfo::CheckStrategy(const StrategyPtr &strategy) { return CheckStrategyValue(strategy, inputs_shape_); }

/*
 * support parallel degree smaller than device number, set the duplicate device dimension to the first dimension of
 * device matrix
 * only support batch parallel reshape operator in ReID (batch parallel degree can be smaller than device number)
 */
Status ReshapeInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  input_strategy_ = stra.at(0);
  dev_matrix_shape_ = stra.at(0);
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
    ReportError(name_ + ": Create group failed.");
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
    MS_LOG(ERROR) << name_ << ": Elements size must be equal to outputs shape[0] size.";
    return FAILED;
  }

  for (auto &element : elements) {
    MS_EXCEPTION_IF_NULL(element);
    if (element->isa<Int64Imm>()) {
      int64_t axis = element->cast<Int64ImmPtr>()->value();
      parameter_input_v_.push_back(axis);
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis must be int32.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ReshapeInfo::ComputeReplaceOp() {
  RankList dev_list = stage_device_list();
  TensorRedistribution tensor_redistribution(!is_generating_costs_, true);
  if (tensor_redistribution.Init(input_layout_, output_layout_, dev_list) == FAILED) {
    if (is_generating_costs_) {
      MS_LOG(DEBUG) << name_ << ": tensor_redistribution init failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": tensor_redistribution init failed.";
    }
    return FAILED;
  }
  MS_LOG(DEBUG) << name_ << ": input " << input_layout_.ToString();
  MS_LOG(DEBUG) << name_ << ": output " << output_layout_.ToString();
  MS_LOG(DEBUG) << name_ << ": dev_list " << dev_list.size();
  if (is_skip_) {
    ConstructOperator constructor;
    replace_op_ = constructor.SkipRedisReshapeOP(output_layout_.slice_shape().array());
    replace_op_info_.clear();
    MS_LOG(INFO) << "skip reshape redistribution and reshape slice_shape is "
                 << ShapeToString(output_layout_.slice_shape().array());
  } else {
    RedistributionOpListPtr redistribution_oplist_ptr = tensor_redistribution.InferTensorRedistributionOperatorList();
    if (redistribution_oplist_ptr == nullptr) {
      if (is_generating_costs_) {
        MS_LOG(DEBUG) << name_ << "InferTensorRedistribution failed.";
      } else {
        MS_LOG(ERROR) << name_ << "InferTensorRedistribution failed.";
      }
      return FAILED;
    }
    replace_op_ = redistribution_oplist_ptr->first;
    replace_op_info_ = redistribution_oplist_ptr->second;
  }
  MS_LOG(DEBUG) << name_ << ": replace op size = " << replace_op_.size();
  if (replace_op_.size() == 1 && replace_op_.front().first == RESHAPE) {
    int64_t shape_dim = 2;
    auto value = replace_op_.front().second.second.front().first.second;
    Shape dst_shape = GetValue<std::vector<int64_t>>(value);
    Shape origin_dst_shape =
      GetValue<std::vector<int64_t>>(cnode_->input(LongToSize(shape_dim))->cast<ValueNodePtr>()->value());
    if (dst_shape.size() == origin_dst_shape.size()) {
      for (size_t i = 0; i < dst_shape.size(); ++i) {
        if (origin_dst_shape[i] != dst_shape[i] && origin_dst_shape[i] != -1) {
          return SUCCESS;
        }
      }
      MS_LOG(INFO) << "The reshape would not change the target shape.";
      replace_op_.front().second.second.front().first.second = MakeValue(origin_dst_shape);
    }
  }
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

  Shape tensor_map_index_input;
  for (size_t j = 0; j < inputs_shape_[0].size(); ++j) {
    tensor_map_index_input.push_back(SizeToLong(inputs_shape_[0].size() - j - 1));
  }
  inputs_tensor_map_.push_back(tensor_map_index_input);

  Shape tensor_map_index_output;
  for (size_t j = 0; j < outputs_shape_[0].size(); ++j) {
    tensor_map_index_output.push_back(MAP_NONE);
  }
  outputs_tensor_map_.push_back(tensor_map_index_output);
  return SUCCESS;
}

/*
 * the output tensor strategy is the same as input tensor strategy
 * only support batch parallel reshape operator in ReID (batch parallel degree can be smaller than device number)
 */
Strategies ReshapeInfo::GetOutputsStrategy() {
  Strategies outputs_strategy;
  Dimensions strategy;
  for (size_t j = 0; j < outputs_shape_[0].size(); ++j) {
    strategy.push_back(1);
  }
  outputs_strategy.push_back(strategy);
  return outputs_strategy;
}

Status ReshapeInfo::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout) {
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
  // skip reshape infer if skip_redistribution is true
  if (is_skip_) {
    TensorLayout layout;
    Shape shape;
    Shape slice_shape;
    layout.set_skip_redistribution(true);
    TensorInfo tensor_info_in(layout, shape, slice_shape);
    inputs_tensor_info_.push_back(tensor_info_in);
    outputs_tensor_info_.push_back(tensor_info_in);
    MS_LOG(DEBUG) << name() << "skip redistribution reshape InferTensorInfo";
    return SUCCESS;
  }

  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategies inputs_strategy = strategy_->GetInputDim();
  Strategies outputs_strategy = GetOutputsStrategy();
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

void ReshapeInfo::device_number() {
  dev_num_ = stage_device_size_;
  MS_ASSERT(dev_num_ > 0);
}

Status ReshapeInfo::InferDefaultLayout(const Shape &shape, TensorLayout *const layout) {
  Shape tensor_map_index;
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

Status ReshapeInfo::Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  auto reshape_skip_redis_iter = attrs_.find(SKIP_REDISTRIBUTION);
  if (reshape_skip_redis_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(reshape_skip_redis_iter->second);
    if (!reshape_skip_redis_iter->second->isa<BoolImm>()) {
      MS_LOG(ERROR) << name_ << ": skip_redistribution is not a bool.";
      return FAILED;
    }
    is_skip_ = reshape_skip_redis_iter->second->cast<BoolImmPtr>()->value();
  }

  ResetQueueMember();
  device_number();
  if (in_strategy) {
    if (InitWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
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

Status ReshapeInfo::SetCostUnderStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

void ReshapeInfo::SetCostForReshapeWithParameter() {
  size_t success = 0;
  for (auto &sp : sp_vector_) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
}

void ReshapeInfo::SetCostForReshape(const mindspore::parallel::StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  int64_t stage_id = strategy->GetInputStage();
  double computation_cost =
    operator_cost()->GetForwardComputationCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  double communication_cost = operator_cost()->GetCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
  std::shared_ptr<Cost> result = std::make_shared<Cost>(computation_cost, communication_cost);
  result->communication_without_parameter_ =
    operator_cost()->GetForwardCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  result->communication_with_partial_para_ =
    result->communication_without_parameter_ + gamma * (communication_cost - result->communication_without_parameter_);

  // Breaking ties for preferring data parallelization
  BreakingTiesForPreferringDataParallel(strategy, result);
  // refine communication cost calculation for practice
  RefineForPracticalCost(result, false);

  std::shared_ptr<StrategyWithCost> swc =
    std::make_shared<StrategyWithCost>(strategy, inputs_tensor_info_, outputs_tensor_info_);
  swc->cost_list.push_back(result);
  strategy_cost_.emplace_back(swc);
}

std::vector<StrategyPtr> ReshapeInfo::GenerateOpStrategies(int64_t stage_id) {
  if (inputs_shape_.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": Inputs shape size or is empty";
  }
  Shape input0_split;
  (void)input0_split.insert(input0_split.cend(), inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  // strategy used only in the input node is parameter,
  // in other case, use the input node's output_layout as input_layout.
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector_) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": GenerateStrategiesForIndependentInputs failed.";
  }

  return sp_vector_;
}

Status ReshapeInfo::GenerateStrategyCosts(
  const std::vector<std::shared_ptr<StrategyWithCost>> &pre_stra_costs,
  std::vector<std::pair<std::vector<std::shared_ptr<StrategyWithCost>>, int64_t>> next_costs_index, int64_t out_index,
  bool is_prev_param, bool is_next_reshape) {
  is_generating_costs_ = true;
  for (auto pre_stra_cost : pre_stra_costs) {
    std::vector<TensorInfo> pre_out_tensor_infos;
    if (is_prev_param) {
      pre_out_tensor_infos = pre_stra_cost->inputs_ptr;
    } else {
      pre_out_tensor_infos = pre_stra_cost->outputs_ptr;
    }
    if (pre_out_tensor_infos.size() <= LongToSize(out_index)) {
      MS_LOG(ERROR) << "out_index is out of range of the tensor_infos in setting reshape's input_layout";
      return FAILED;
    }
    TensorInfo pre_out_tensor_info = pre_out_tensor_infos[LongToSize(out_index)];
    SetInputLayout(pre_out_tensor_info.tensor_layout());
    // infer pre_node output strategy from output_layout.
    Dimensions stra = pre_out_tensor_info.InferStrategy();
    if (stra.empty()) {
      MS_LOG(ERROR) << "Infer strategy by tensor_info failed";
      return FAILED;
    }
    Strategies stra_inputs = {stra};
    StrategyPtr reshape_stra = std::make_shared<Strategy>(pre_stra_cost->strategy_ptr->GetInputStage(), stra_inputs);
    if (is_next_reshape) {
      SetOutputLayout(pre_out_tensor_info.tensor_layout());
      ResetQueueMember();
      InferTensorInfoByLayout();
      SetCostForReshape(reshape_stra);
    } else if (next_costs_index.empty()) {
      if (Init(nullptr, nullptr) == FAILED) {
        MS_LOG(ERROR) << "Failure:operator reshape init failed";
        return FAILED;
      }
      SetCostForReshape(reshape_stra);
      continue;
    }
    for (auto next_cost_index_pair : next_costs_index) {
      auto in_index = next_cost_index_pair.second;
      auto next_stra_costs = next_cost_index_pair.first;
      for (auto next_stra_cost : next_stra_costs) {
        std::vector<TensorInfo> next_in_tensor_infos = next_stra_cost->inputs_ptr;
        if (next_in_tensor_infos.size() <= LongToSize(in_index)) {
          MS_LOG(ERROR) << "in_index is out of range of the tensor_infos in setting reshape's output_layout";
          return FAILED;
        }
        TensorInfo next_in_tensor_info = next_in_tensor_infos[LongToSize(in_index)];

        SetOutputLayout(next_in_tensor_info.tensor_layout());
        ResetQueueMember();
        InferTensorInfoByLayout();
        SetCostForReshape(reshape_stra);
      }
    }
  }
  is_generating_costs_ = false;
  if (strategy_cost_.empty()) {
    return FAILED;
  }
  MS_LOG(INFO) << "Print " << name() << "'s 'strategy_cost':";
  for (auto &swc : strategy_cost_) {
    MS_LOG(INFO) << name() << "'s strategy:";
    PrintStrategy(swc->strategy_ptr);
    MS_LOG(INFO) << "The corresponding cost: " << swc->cost_list[0]->computation_cost_ << ", "
                 << swc->cost_list[0]->communication_cost_ << ", "
                 << swc->cost_list[0]->communication_without_parameter_;
    MS_LOG(INFO) << "Input layout: " << swc->inputs_ptr[0].tensor_layout().ToString();
    MS_LOG(INFO) << "Output layout: " << swc->outputs_ptr[0].tensor_layout().ToString();
  }
  return SUCCESS;
}

int64_t ReshapeInfo::GetSWCIndexByOutputLayoutWithZeroComm(const TensorLayout &output_layout) {
  std::vector<std::pair<int64_t, double>> index_computation;
  for (size_t i = 0; i < strategy_cost_.size(); ++i) {
    const auto &swc = strategy_cost_[i];
    if (swc->outputs_ptr[0].tensor_layout() == output_layout &&
        fabs(swc->cost_list[0]->communication_without_parameter_ - 0.0) < DBL_EPSILON) {
      (void)index_computation.emplace_back(SizeToLong(i), swc->cost_list[0]->computation_cost_);
    }
  }
  if (index_computation.empty()) {
    MS_LOG(WARNING) << "There in no available strategy for zero communication cost for reshape: " << name();
    return -1;
  }
  if (index_computation.size() > 1) {
    MS_LOG(INFO) << "There are multiple strategies available for reshape: " << name();
  }
  std::sort(
    index_computation.begin(), index_computation.end(),
    [](const std::pair<int64_t, double> &a, const std::pair<int64_t, double> &b) { return a.second < b.second; });
  return index_computation[0].first;
}

int64_t ReshapeInfo::GetSWCIndexByOutputLayoutWithMiniComm(const TensorLayout &output_layout) {
  std::vector<std::pair<int64_t, double>> index_comm;
  for (size_t i = 0; i < strategy_cost_.size(); ++i) {
    const auto &swc = strategy_cost_[i];
    if (swc->outputs_ptr[0].tensor_layout() == output_layout) {
      (void)index_comm.emplace_back(SizeToLong(i), swc->cost_list[0]->communication_without_parameter_);
    }
  }
  if (index_comm.empty()) {
    MS_LOG(ERROR) << "There in no available strategy for zero communication cost for reshape: " << name();
    return -1;
  }
  if (index_comm.size() > 1) {
    MS_LOG(INFO) << "There are multiple strategies available for reshape: " << name();
  }
  std::sort(
    index_comm.begin(), index_comm.end(),
    [](const std::pair<int64_t, double> &a, const std::pair<int64_t, double> &b) { return a.second < b.second; });
  return index_comm[0].first;
}

int64_t ReshapeInfo::GetSWCIndexByInputLayoutWithZeroComm(const TensorLayout &input_layout) {
  std::vector<std::pair<int64_t, double>> index_computation;
  for (size_t i = 0; i < strategy_cost_.size(); ++i) {
    const auto &swc = strategy_cost_[i];
    if (swc->inputs_ptr[0].tensor_layout() == input_layout &&
        fabs(swc->cost_list[0]->communication_without_parameter_ - 0.0) < DBL_EPSILON) {
      (void)index_computation.emplace_back(SizeToLong(i), swc->cost_list[0]->computation_cost_);
    }
  }
  if (index_computation.empty()) {
    MS_LOG(WARNING) << "There in no available strategy for zero communication cost for reshape: " << name();
    return -1;
  }
  if (index_computation.size() > 1) {
    MS_LOG(INFO) << "There are multiple strategies available for reshape: " << name();
  }
  std::sort(
    index_computation.begin(), index_computation.end(),
    [](const std::pair<int64_t, double> &a, const std::pair<int64_t, double> &b) { return a.second < b.second; });
  return index_computation[0].first;
}

int64_t ReshapeInfo::GetSWCIndexByInputLayoutWithMiniComm(const TensorLayout &input_layout) {
  std::vector<std::pair<int64_t, double>> index_comm;
  for (size_t i = 0; i < strategy_cost_.size(); ++i) {
    const auto &swc = strategy_cost_[i];
    if (swc->inputs_ptr[0].tensor_layout() == input_layout) {
      (void)index_comm.emplace_back(SizeToLong(i), swc->cost_list[0]->communication_without_parameter_);
    }
  }
  if (index_comm.empty()) {
    MS_LOG(ERROR) << "There in no available strategy for zero communication cost for reshape: " << name();
    return -1;
  }
  if (index_comm.size() > 1) {
    MS_LOG(INFO) << "There are multiple strategies available for reshape: " << name();
  }
  std::sort(
    index_comm.begin(), index_comm.end(),
    [](const std::pair<int64_t, double> &a, const std::pair<int64_t, double> &b) { return a.second < b.second; });
  return index_comm[0].first;
}

bool ReshapeInfo::CheckStrategyConsistencyByOutputLayout(int64_t swc_index, const TensorLayout &output_layout) const {
  if (swc_index == -1 || swc_index >= SizeToLong(strategy_cost_.size())) {
    MS_LOG(ERROR) << "The strategy_index: " << swc_index << " is out of range.";
    return false;
  }
  const auto &swc = strategy_cost_[LongToSize(swc_index)];
  if (swc->outputs_ptr[0].tensor_layout() == output_layout) {
    return true;
  }
  MS_LOG(WARNING) << name_ << "'s desired output layout is: " << output_layout.ToString() << ", while the selected "
                  << "output layout is: " << swc->outputs_ptr[0].tensor_layout().ToString()
                  << " and the input layout is: " << swc->inputs_ptr[0].tensor_layout().ToString();
  return false;
}

bool ReshapeInfo::CheckStrategyConsistencyByInputLayout(int64_t swc_index, const TensorLayout &input_layout) const {
  if (swc_index == -1 || swc_index >= SizeToLong(strategy_cost_.size())) {
    MS_LOG(ERROR) << "The strategy_index: " << swc_index << " is out of range.";
    return false;
  }
  const auto &swc = strategy_cost_[LongToSize(swc_index)];
  if (swc->inputs_ptr[0].tensor_layout() == input_layout) {
    return true;
  }
  MS_LOG(WARNING) << name_ << "'s desired input layout is:" << input_layout.ToString() << ", while the selected "
                  << "input layout is: " << swc->inputs_ptr[0].tensor_layout().ToString()
                  << " and the output layout is: " << swc->outputs_ptr[0].tensor_layout().ToString();
  return false;
}

TensorLayout ReshapeInfo::GetInputLayoutBySWCIndex(int64_t swc_index) const {
  if (swc_index == -1 || swc_index >= SizeToLong(strategy_cost_.size())) {
    MS_LOG(EXCEPTION) << "The strategy_index: " << swc_index << " is out of range.";
  }
  const auto &swc = strategy_cost_[LongToSize(swc_index)];
  return std::move(swc->inputs_ptr[0].tensor_layout());
}

TensorLayout ReshapeInfo::GetOutputLayoutBySWCIndex(int64_t swc_index) const {
  if (swc_index == -1 || swc_index >= SizeToLong(strategy_cost_.size())) {
    MS_LOG(EXCEPTION) << "The strategy_index: " << swc_index << " is out of range.";
  }
  const auto &swc = strategy_cost_[LongToSize(swc_index)];
  return std::move(swc->outputs_ptr[0].tensor_layout());
}

REGISTER(ReshapeInfo);
}  // namespace parallel
}  // namespace mindspore
