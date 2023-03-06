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

#include "frontend/parallel/ops_info/batch_parallel_info.h"

#include <memory>
#include <utility>

#include "ir/value.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
Status BatchParallelInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  size_t strategy_size = strategy->GetInputNumber();
  Strategies stra = strategy->GetInputDim();
  for (size_t i = 0; i < strategy_size; ++i) {
    Shape sub_strategy = stra.at(i);
    size_t strategy_len = sub_strategy.size();
    bool flag = false;
    for (size_t j = 0; j < strategy_len; ++j) {
      int64_t strategy_value = sub_strategy.at(j);
      if (strategy_value > 1) {
        if (flag || strategy_value != stage_device_size_) {
          MS_LOG(ERROR) << name_ << " : It is not a valid data parallel strategy.";
          return FAILED;
        }
        flag = true;
      }
    }
  }
  return SUCCESS;
}

Status BatchParallelInfo::InferDevMatrixShape() {
  dev_matrix_shape_.push_back(stage_device_size_);

  if (need_replace_input_ && !inputs_shape_.empty()) {
    replace_shape_ = inputs_shape_[0];
    if (!replace_shape_.empty()) {
      replace_shape_[0] /= stage_device_size_;
    }
  }

  return SUCCESS;
}

Status BatchParallelInfo::InferForwardCommunication() { return SUCCESS; }

Status BatchParallelInfo::InferTensorMap() {
  auto strategy = strategy_->GetInputDim();
  if (strategy.empty()) {
    MS_LOG(INFO) << name_ << ": the strategy is empty";
    return SUCCESS;
  }

  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    Shape tensor_map_index;
    for (size_t j = 0; j < inputs_shape_[i].size(); ++j) {
      if (strategy[i][j] > 1 && j == 0) {
        tensor_map_index.push_back(0);
      } else {
        tensor_map_index.push_back(MAP_NONE);
      }
    }
    inputs_tensor_map_.push_back(tensor_map_index);
  }
  for (size_t i = 0; i < outputs_shape_.size(); i++) {
    Shape tensor_map_index;
    for (size_t j = 0; j < outputs_shape_[i].size(); ++j) {
      if (i == 0 && j == 0) {
        tensor_map_index.push_back(0);
      } else {
        tensor_map_index.push_back(MAP_NONE);
      }
    }
    outputs_tensor_map_.push_back(tensor_map_index);
  }
  return SUCCESS;
}

Status BatchParallelInfo::GetAttrs() {
  // if the operator's input is a shape(is not a tensor), need to assign the shape value to inputs_shape_
  if (!inputs_shape_.empty()) {
    return SUCCESS;
  }

  if (input_value_.empty()) {
    return SUCCESS;
  }

  auto shape_ptr = input_value_[0]->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);

  inputs_shape_.push_back(GetValue<Shape>(shape_ptr));
  need_replace_input_ = true;
  return SUCCESS;
}

Status BatchParallelInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> BatchParallelInfo::GenerateOpStrategies(int64_t stage_id) {
  StrategyPtr sp;
  Strategies strategy;
  ComputeBatchSplitFlagList();

  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    Shape temp(inputs_shape_[i].size(), 1);
    if (split_flag_list_[i]) {
      temp[0] = stage_device_size_;
    }
    strategy.push_back(temp);
  }
  sp = std::make_shared<Strategy>(stage_id, strategy);
  std::vector<StrategyPtr> sp_vector;
  sp_vector.push_back(sp);
  return sp_vector;
}

Status BatchParallelInfo::InferAsLossDivisor() {
  as_loss_divisor_ = 1;
  return SUCCESS;
}

void BatchParallelInfo::ReplaceNodeInputOrAttrs() {
  if (!need_replace_input_) {
    return;
  }

  for (auto &cnode : cnodes_) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() != 2) {
      MS_LOG(EXCEPTION) << name_ << ": The size of tile cnode's inputs must be 2";
    }

    if (!IsValueNode<ValueTuple>(cnode->input(1))) {
      MS_LOG(EXCEPTION) << name_ << ": The input[1] of tile cnode is not ValueTuple.";
    }

    auto func_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);

    ValuePtr replace_shape = MakeValue(replace_shape_);
    AnfNodePtr val = NewValueNode(replace_shape);
    cnode->set_input(kIndex1, val);
  }
}

void SparseSoftmaxCrossEntropyWithLogitsInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    split_flag_list_[i] = true;
  }
}

Status CheckValidInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  if (stra[0][1] != 1) {
    MS_LOG(ERROR) << name_ << ": The second dimension of the first input can not be split, but got " << stra[0][1];
    return FAILED;
  }

  if (stra[1][0] != 1) {
    MS_LOG(ERROR) << name_ << ": The second input can not be split, but got " << stra[1][0];
    return FAILED;
  }
  return SUCCESS;
}

Status CheckValidInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  dev_matrix_shape_.push_back(stra[0][0]);
  return SUCCESS;
}

void CheckValidInfo::ReComputeBatchSplitFlagList() { split_flag_list_[0] = true; }

REGISTER(BatchParallelInfo);
REGISTER(SparseSoftmaxCrossEntropyWithLogitsInfo);
REGISTER(CheckValidInfo);  // has not bprop
}  // namespace parallel
}  // namespace mindspore
