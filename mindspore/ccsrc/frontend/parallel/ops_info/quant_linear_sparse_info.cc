/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/quant_linear_sparse_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "primitive/sequence_ops.h"
#include "ir/value.h"
#include "include/frontend/parallel/graph_costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {

namespace {
constexpr size_t kQuantLinearSparseInputX = 0;
constexpr size_t kQuantLinearSparseInputWeight = 1;
constexpr size_t kQuantLinearSparseInputDeqScale = 2;
constexpr size_t kQuantLinearSparseInputCompressIdx = 3;
constexpr size_t kQuantLinearSparseInputBias = 4;
constexpr size_t kQuantLinearSparseOutput = 0;
constexpr size_t kQuantLinearSparseInputNum = 5;
}  // namespace

Status QuantLinearSparseInfo::GetAttrs() { return SUCCESS; }

Status QuantLinearSparseInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();

  if ((stra[kQuantLinearSparseInputWeight][0] != 1 || stra[kQuantLinearSparseInputCompressIdx][0] != 1)) {
    MS_LOG(ERROR) << name_ << ": weight's and compressIdx's strategy should be (1,)";
    return FAILED;
  }

  if ((stra[kQuantLinearSparseInputDeqScale][0] != stra[kQuantLinearSparseInputBias][0])) {
    MS_LOG(ERROR) << name_ << ": deqScale's and bias's strategy should be same";
    return FAILED;
  }

  return SUCCESS;
}

Status QuantLinearSparseInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  dev_matrix_shape_ = {stra[0][0], stra[0][1], stra[2][0]};
  MS_LOG(DEBUG) << name_ << ": The dev matrix shape is " << dev_matrix_shape_;
  return SUCCESS;
}

Status QuantLinearSparseInfo::InferTensorMap() {
  inputs_tensor_map_.push_back({2, 1});
  inputs_tensor_map_.push_back({-1});
  inputs_tensor_map_.push_back({0});
  inputs_tensor_map_.push_back({-1});
  inputs_tensor_map_.push_back({0});
  TensorMap out_tensor_map = {2, 0};
  outputs_tensor_map_.push_back(out_tensor_map);
  return SUCCESS;
}

Status QuantLinearSparseInfo::InferForwardCommunication() {
  if (is_layout_config_) {
    return SUCCESS;
  }
  forward_op_.clear();
  size_t dimension = dev_matrix_shape_.size();
  size_t relevant_dimension_index = SECOND_FROM_END(dimension);
  // Relevant dimension is not split and all reduce is not required,
  // need to use origin_dev_matrix_shape_ here, since the dev_matrix_shape_ will be changed if repeated calculation.
  if (dev_matrix_shape_.at(relevant_dimension_index) == MIN_SLICE_NUM) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required.";
    return SUCCESS;
  }

  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    // if repeated calculation and repeated num in the left of dev matrix, the index of relevant dimension should add 1
    relevant_dimension_index += 1;
  }

  std::vector<Group> group_list;
  if (CreateGroupByDim(relevant_dimension_index, &group_list) != SUCCESS) {
    ReportError(name_ + ": Infer forward communication, create group failed.");
    return FAILED;
  } else if (group_list.empty() || group_list.front().GetDevNum() <= kSizeOne) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required.";
    return SUCCESS;
  }
  Operator op;
  op = CreateAllReduceOp(REDUCE_OP_SUM, group_list[0].name());
  forward_op_.push_back(op);
  MS_LOG(INFO) << name_ << ": The group name of forward communication is " << group_list[0].name();
  return SUCCESS;
}

std::vector<StrategyPtr> QuantLinearSparseInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape split_flag;
  for (size_t i = 0; i < inputs_shape_[0].size() - 1; ++i) {
    split_flag.push_back(1);
  }

  Shapes splittable_input = {split_flag};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy";
  }

  return sp_vector;
}

std::shared_ptr<Strategies> QuantLinearSparseInfo::GenerateBatchStrategies() {
  MS_EXCEPTION_IF_ZERO("device_num", stage_device_size_);
  Strategies strategy_v = {};
  Dimensions batch_strategy_x1(inputs_shape_[kQuantLinearSparseInputX].size(), 1);
  batch_strategy_x1[0] = stage_device_size_;
  strategy_v.emplace_back(batch_strategy_x1);

  for (size_t i = kQuantLinearSparseInputWeight; i < inputs_shape_.size(); i++) {
    Dimensions strategy_dimensions(inputs_shape_[i].size(), 1);
    strategy_v.emplace_back(strategy_dimensions);
  }
  return std::make_shared<Strategies>(strategy_v);
}

Status QuantLinearSparseInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

REGISTER(QuantLinearSparseInfo);
}  // namespace parallel
}  // namespace mindspore
