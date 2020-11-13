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

#include "frontend/parallel/ops_info/range_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
float RangeInfo::GetRangeAttr(const std::string &arg) {
  auto iter = attrs_.find(arg);
  if (iter == attrs_.end()) {
    MS_LOG(EXCEPTION) << name_ << ": Can not find the attr for " << arg;
  }

  MS_EXCEPTION_IF_NULL(iter->second);
  if (!iter->second->isa<FP32Imm>()) {
    MS_LOG(EXCEPTION) << name_ << ": The type of attr is not float, the attr is " << arg;
  }

  return iter->second->cast<FP32ImmPtr>()->value();
}

Status RangeInfo::GetAttrs() {
  start_ = GetRangeAttr(START);
  limit_ = GetRangeAttr(LIMIT);
  delta_ = GetRangeAttr(DELTA);
  MS_LOG(INFO) << name_ << ": The start is " << start_ << ", the limit is " << limit_ << ", the delta is " << delta_;
  return SUCCESS;
}

Status RangeInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }
  return SUCCESS;
}

Status RangeInfo::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  dev_matrix_shape_ = stra[0];
  split_num_ = stra[0][0];
  return SUCCESS;
}

Status RangeInfo::InferMirrorOps() { return SUCCESS; }

Status RangeInfo::InferForwardCommunication() { return SUCCESS; }

Status RangeInfo::InferTensorMap() {
  TensorMap input_tensor_map = {0}, output_tensor_map = {0};

  inputs_tensor_map_.push_back(input_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
  return SUCCESS;
}

Status RangeInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }

  TensorLayout input_layout, output_layout;
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    // infer tensor layout
    if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[i], inputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed.";
      return FAILED;
    }
    TensorInfo input_tensor_info(input_layout);
    inputs_tensor_info_.push_back(input_tensor_info);
  }

  if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], outputs_shape_[0]) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed.";
    return FAILED;
  }
  TensorInfo output_tensor_info(output_layout);
  outputs_tensor_info_.push_back(output_tensor_info);

  for (auto &tensor_info : inputs_tensor_info_) {
    MS_LOG(INFO) << name_ << ": The input layout: " << tensor_info.tensor_layout().ToString();
  }
  MS_LOG(INFO) << name_ << ": The output layout: " << outputs_tensor_info_[0].tensor_layout().ToString();
  return SUCCESS;
}

Status RangeInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init success";
  return SUCCESS;
}

Status RangeInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success";
  return SUCCESS;
}

Status RangeInfo::InferNewAttr() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->rank_index_in_stage();

  // If repeated calculation and repeated num as the last dimension of dev-matrix,
  // the dev-matrix is [split_num_, repeated_calc_num_], so from rank 0 to rank repeated_calc_num_
  // are repeated calculation, and these rank have the same 'new_start_'.
  // If repeated calculation and repeated num as the first dimension of dev-matrix,
  // the dev-matrix is [repeated_calc_num_, split_num_], so rank 0 and rank split_num_ and so on
  // are repeated calculation, and these rank have the same 'new_start_'.
  float start_bias = inputs_shape_[0][0] / split_num_ * delta_;
  if (repeated_num_in_dev_matrix_right_) {
    new_start_ = start_ + start_bias * (rank / repeated_calc_num_);
  } else {
    new_start_ = start_ + start_bias * (rank % split_num_);
  }

  new_limit_ = new_start_ + start_bias;
  MS_LOG(INFO) << name_ << ": The new start is " << new_start_ << ", the new limit is " << new_limit_;
  return SUCCESS;
}

Status RangeInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph();
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateGraph Init failed";
    return FAILED;
  }

  if (InferNewAttr() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer new attr failed";
    return FAILED;
  }

  Attr attr_start = std::make_pair(START, MakeValue(new_start_));
  Attr attr_limit = std::make_pair(LIMIT, MakeValue(new_limit_));
  Attr attr_delta = std::make_pair(DELTA, MakeValue(delta_));
  OperatorAttrs attrs = {attr_start, attr_limit, attr_delta};
  auto new_range_op = gen_g.PushBack({gen_g.NewOpInst(RANGE, attrs), gen_g.virtual_input_node()});
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(new_range_op, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, new_range_op));

  return SUCCESS;
}

ReplaceGraphPtr RangeInfo::replace_graph(const CNodePtr &cnode) {
  if (ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
  }
  return replace_graph_;
}

Status RangeInfo::GenerateStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Generate strategies for independent inputs() failed.";
    return FAILED;
  }
  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

Status RangeInfo::SetCostUnderStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}
}  // namespace parallel
}  // namespace mindspore
