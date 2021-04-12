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

#include "frontend/parallel/ops_info/unsorted_segment_op_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/strategy.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
// The operator UnsortedSegment accepts three inputs:
// input0 : vector, the shape is x1,x2,x3,...,xr
// input1 : segment id, the shape is x1,x2,..,xn
// input2 : value, the number of the segments
// For Sum:  r >= n
// For Min:  r >=n, n=1
Status UnsortedSegmentOpInfo::GetAttrs() {
  if (inputs_shape_.size() != UNSORTEDSEGMENTOP_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": inputs shape size must be 2, but is " << inputs_shape_.size();
    return FAILED;
  }
  if (outputs_shape_.size() != UNSORTEDSEGMENTOP_OUTPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": outputs shape size must be 1, but is " << outputs_shape_.size();
    return FAILED;
  }
  if (input_value_.at(2) == nullptr) {
    MS_LOG(ERROR) << name_ << ": the third input value is nullptr, is not a ValueNode!";
    return FAILED;
  }

  if (inputs_shape_.at(0).empty()) {
    MS_LOG(ERROR) << name_ << ": input can not be a scalar!";
    return FAILED;
  }
  auto num_segments = GetValue<int64_t>(input_value_.at(2));
  if (num_segments < 0) {
    MS_LOG(ERROR) << name_ << ": the number of segments should be non negative value.";
    return FAILED;
  }

  return SUCCESS;
}

Status UnsortedSegmentOpInfo::CheckStrategy(const StrategyPtr &strategy) {
  // Check size
  if (inputs_shape_.size() != UNSORTEDSEGMENTOP_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": inputs shape size must be " << UNSORTEDSEGMENTOP_INPUTS_SIZE << ", but is "
                  << inputs_shape_.size();
    return FAILED;
  }
  if (outputs_shape_.size() != UNSORTEDSEGMENTOP_OUTPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": outputs shape size must be " << UNSORTEDSEGMENTOP_OUTPUTS_SIZE << ", but is "
                  << outputs_shape_.size();
    return FAILED;
  }
  // The strategy of the first and the second input should be set.
  if (CheckStrategyValue(strategy, {inputs_shape_.at(0), inputs_shape_.at(1)}) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    return FAILED;
  }
  Strategys stra = strategy->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  Shape input_a_shape = inputs_shape_.at(0);
  Shape input_b_shape = inputs_shape_.at(1);
  // The size of the input b must be equal or smaller than input a
  for (size_t i = 0; i < input_b_shape.size(); ++i) {
    if ((sub_a_strategy[i] != sub_b_strategy[i]) || (input_a_shape[i] != input_b_shape[i])) {
      MS_LOG(ERROR) << name_
                    << " : Invalid strategy. The shape and the strategy of the input0 and input1 "
                       "should be same before the front size of the input[1]";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status UnsortedSegmentOpInfo::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  dev_matrix_shape_ = stra.at(0);
  return SUCCESS;
}

Status UnsortedSegmentOpInfo::InferMirrorOps() {
  mirror_ops_.clear();

  // Only the first input could be parameter.
  Shape tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(tensor_map, &group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create group failed.";
    return FAILED;
  }

  OperatorVector mirror_op;
  OperatorVector op_for_value;
  OperatorVector op_for_value2;
  if (group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror ops is empty.";
    return SUCCESS;
  } else {
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
    mirror_ops_.push_back(op_for_value);
    mirror_ops_.push_back(op_for_value2);
    std::string group_name = group[0].name();
    MS_LOG(INFO) << name_ << " : Create the mirror ops success, the group name is " << group_name;
  }

  return SUCCESS;
}

// As the op converts the vector x1,x2,x3...,xr -> number of segments, xn,..,xr
// the dimension x1,x2,x3,..,xn is eliminated
// suppose the strategy of the inputs is (a,b,c,d), (a,b)
// the tensor map of the input vector is (3,2,1,0), id:(1, 0)
// the output vector is (-1, 1, 0)
Status UnsortedSegmentOpInfo::InferTensorMap() {
  Shape tensor_map_in;
  Shape tensor_map_in_index;
  Shape tensor_map_out;
  size_t input0_size = inputs_shape_.at(0).size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < input0_size; ++i) {
    tensor_map_in.push_back(SizeToInt(input0_size - i - 1));
    tensor_map_in_index.push_back(SizeToInt(input0_size - i - 1));
    tensor_map_out.push_back(SizeToInt(input0_size - i - 1));
  }

  (void)tensor_map_out.erase(tensor_map_out.begin(), tensor_map_out.begin() + inputs_shape_.at(1).size() - 1);
  // A special case: the input vector (a,) id (a,) or input vector (a,b,c), id(a,b,c)
  // The output vector will be a 1-dim vector,
  // These two kinds of situations as row slice.
  tensor_map_out[0] = -1;
  (void)tensor_map_in_index.erase(tensor_map_in_index.begin() + inputs_shape_.at(1).size(), tensor_map_in_index.end());
  if (tensor_map_out.size() != outputs_shape_.at(0).size()) {
    MS_LOG(ERROR) << "Out tensor map size is not equal to output size! Out tensor map size is " << tensor_map_out.size()
                  << " output size is " << outputs_shape_.at(0).size();
    return FAILED;
  }

  inputs_tensor_map_.emplace_back(std::move(tensor_map_in));
  inputs_tensor_map_.emplace_back(std::move(tensor_map_in_index));
  outputs_tensor_map_.emplace_back(std::move(tensor_map_out));
  return SUCCESS;
}

Status UnsortedSegmentOpInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);
  Shape input_index_shape = inputs_shape_.at(1);
  Shape output_shape = outputs_shape_.at(0);

  TensorLayout input_tensor_layout, input_index_layout, output_tensor_layout;
  if ((input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(0), input_shape) != SUCCESS) ||
      (input_index_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(1), input_index_shape) != SUCCESS) ||
      (output_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_.at(0), output_shape) != SUCCESS)) {
    return FAILED;
  }

  TensorInfo input_tensor_info(input_tensor_layout);
  TensorInfo input_index_info(input_index_layout);
  TensorInfo output_tensor_info(output_tensor_layout);

  inputs_tensor_info_.push_back(input_tensor_info);
  inputs_tensor_info_.push_back(input_index_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status UnsortedSegmentOpInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status UnsortedSegmentOpInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

// Set the default strategy
Status UnsortedSegmentOpInfo::GenerateStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, {inputs_shape_.at(0)}, splittable_inputs, &sp_vector) !=
      SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Generate strategies for independent inputs() failed.";
    return FAILED;
  }
  for (auto &sp : sp_vector) {
    Strategys tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    Dimensions second_input_strategy;
    for (size_t i = 0; i < inputs_shape_[1].size(); ++i) {
      second_input_strategy.push_back(first_input_strategy[i]);
    }
    tmp_strategy.push_back(first_input_strategy);
    tmp_strategy.push_back(second_input_strategy);
    sp->ResetInputs(tmp_strategy);
  }
  size_t success = 0;
  for (auto &sp : sp_vector) {
    PrintStrategy(sp);
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << " : Successfully generated " << success << " strategy";
      PrintStrategy(sp);
    }
  }

  return SUCCESS;
}

// if the dimension of the input b is split, we regarded it as the row slice, thus requires a AllReduce
// otherwise it is column slice,
Status UnsortedSegmentOpInfo::InferForwardCommunication() {
  forward_op_.clear();
  std::vector<Group> group_list;
  Shape tmp_group_tensor_map = outputs_tensor_map_.at(0);
  if (repeated_calc_num_ > 1) {
    tmp_group_tensor_map.push_back(0);
  }
  if (CreateGroupByTensorMap(tmp_group_tensor_map, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Infer forward communication, create group failed.";
    return FAILED;
  } else if (group_list.empty()) {
    MS_LOG(INFO) << name_ << " : Forward all reduce is not required.";
    return SUCCESS;
  }

  Operator op;
  op = CreateAllReduceOp(REDUCE_OP_SUM, group_list[0].name());

  forward_op_.push_back(op);
  MS_LOG(INFO) << name_ << " : The group name of forward communication is " << group_list[0].name();
  return SUCCESS;
}

Status UnsortedSegmentOpInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::shared_ptr<Strategys> UnsortedSegmentOpInfo::GenerateBatchStrategies() {
  if (inputs_shape_.size() != UNSORTEDSEGMENTOP_INPUTS_SIZE) {
    MS_LOG(EXCEPTION) << name_ << ": inputs shape size must be " << UNSORTEDSEGMENTOP_INPUTS_SIZE << ", but is "
                      << inputs_shape_.size();
  }
  if (GetAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << "GetAttrs failed!";
  }

  Dimensions strategy_a, strategy_b;
  strategy_a.push_back(stage_device_size_);
  for (size_t i = 1; i < inputs_shape_[0].size(); i++) {
    strategy_a.push_back(1);
  }

  strategy_b.push_back(stage_device_size_);
  for (size_t i = 1; i < inputs_shape_[1].size(); i++) {
    strategy_b.push_back(1);
  }
  Strategys strategy_v = {strategy_a, strategy_b};
  return std::make_shared<Strategys>(strategy_v);
}

// When the index is splited, the graph should be replaced
// a special case is when the shape input equals the shape of ids, we regard it as column slice,
// thus there is no need for repalce graphs
ReplaceGraphPtr UnsortedSegmentMinInfo::replace_graph(const CNodePtr &cnode) {
  auto input_id_strategy = strategy_->GetInputDim().at(1);
  // 1. the two input shapes are same, and the strategy is not all ones
  if (std::any_of(input_id_strategy.begin(), input_id_strategy.end(), [](const int64_t &shard) { return shard > 1; })) {
    if (ComputeReplaceGraph(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
    }
  }
  return replace_graph_;
}

Status UnsortedSegmentMinInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph();
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  // Get the attributes of the UnsortedSegmentMin
  auto num_segments = GetValue<int64_t>(input_value_.at(2));
  // Step1: Output branch
  auto segment_min = gen_g.PushBack({gen_g.NewOpInst(UNSORTED_SEGMENT_MIN), gen_g.virtual_input_node(),
                                     gen_g.virtual_input_node(), CreatInt64Imm(num_segments)});
  auto expandim_output = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), segment_min, CreatInt64Imm(0)});
  auto all_gather_output = gen_g.PushBack({gen_g.NewOpInst(ALL_GATHER), expandim_output});
  auto final_output = gen_g.PushBack({gen_g.NewOpInst(REDUCE_MIN), all_gather_output, CreatInt64Imm(0)});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(segment_min, 1),
                                                             std::make_pair(segment_min, 2)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, final_output));

  return SUCCESS;
}
// The UnsortedSegmentMaxInfo is almost same with UnsortedSegmentMinInfo
// Except the reduceMin op in the ComputeReplaceGraph is replaced with reduceMax op
ReplaceGraphPtr UnsortedSegmentMaxInfo::replace_graph(const CNodePtr &cnode) {
  auto input_id_strategy = strategy_->GetInputDim().at(1);
  // 1. the two input shapes are same, and the strategy is not all ones
  if (std::any_of(input_id_strategy.begin(), input_id_strategy.end(), [](const int64_t &shard) { return shard > 1; })) {
    if (ComputeReplaceGraph(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
    }
  }
  return replace_graph_;
}

Status UnsortedSegmentMaxInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph();
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  // Get the attributes of the UnsortedSegmentMax
  auto num_segments = GetValue<int64_t>(input_value_.at(2));
  auto segment_max = gen_g.PushBack({gen_g.NewOpInst(UNSORTED_SEGMENT_MAX), gen_g.virtual_input_node(),
                                     gen_g.virtual_input_node(), CreatInt64Imm(num_segments)});
  auto expandim_output = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), segment_max, CreatInt64Imm(0)});
  auto all_gather_output = gen_g.PushBack({gen_g.NewOpInst(ALL_GATHER), expandim_output});
  auto final_output = gen_g.PushBack({gen_g.NewOpInst(REDUCE_MAX), all_gather_output, CreatInt64Imm(0)});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(segment_max, 1),
                                                             std::make_pair(segment_max, 2)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, final_output));

  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
