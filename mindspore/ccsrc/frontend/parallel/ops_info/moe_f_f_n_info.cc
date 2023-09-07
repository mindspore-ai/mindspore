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

#include "frontend/parallel/ops_info/moe_f_f_n_info.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include <numeric>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/redistribution_operator_infer.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
// MoeFFN five inputs
// x:       (bs * seq, h)
// expert:  (16)
// weight1: (epert_dim, h, ffn_h)
// bias1:   (epert_dim, ffn_h)
// weight2: (expert_dim, ffn_h, h)
// ------------------------------
// output:  (bs * seq, h)

// split strategy
// bs * seq is able to split.
// h is not able to split.
// ffn_h is able to split.
// expert_dim is able to split.

Status MoeFFNInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto input_strategys = strategy->GetInputDim();
  auto strategy_x = input_strategys.at(0);       // (1,1)
  auto strategy_expert = input_strategys.at(1);  // (1)
  auto strategy_w1 = input_strategys.at(2);      // (1,1,4)
  auto strategy_bias1 = input_strategys.at(3);   // (1,1,4)
  auto strategy_w2 = input_strategys.at(4);      // (1,4,1)

  if (strategy_expert.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The expert can't be shard, but got"
                  << " expert's strategy: " << strategy_expert;
    return FAILED;
  }

  // hidden_size dim must be 1, not able to split.
  if (strategy_x.at(1) != 1 || strategy_x.at(1) != strategy_w1.at(1) || strategy_w1.at(1) != strategy_w2.at(2)) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The hidden size dim must be 1 at the same time, but got"
                  << " x's strategy: " << strategy_x << " weight1's strategy: " << strategy_w1
                  << " weight2's strategy: " << strategy_w2;
    return FAILED;
  }

  // expert_dim must be the same strategy.
  if (strategy_w1.at(0) != 1 || strategy_w1.at(0) != strategy_bias1.at(0) || strategy_w1.at(0) != strategy_w2.at(0)) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The expert dim must be 1 at the same time, but got"
                  << " strategy_w1's strategy: " << strategy_w1 << " strategy_bias1's strategy: " << strategy_bias1
                  << " strategy_w2's strategy: " << strategy_w2;
    return FAILED;
  }

  // ffn_hidden_size dim must be the same strategy.
  if (strategy_w1.at(2) != strategy_bias1.at(1) || strategy_w1.at(2) != strategy_w2.at(1)) {
    MS_LOG(ERROR) << name_ << " Invalid strategy: The expert dim must be shard at the same time, but got"
                  << " strategy_w1's strategy: " << strategy_w1 << " strategy_bias1's strategy: " << strategy_bias1
                  << " strategy_w2's strategy: " << strategy_w2;
    return FAILED;
  }
  return SUCCESS;
}

Status MoeFFNInfo::InferDevMatrixShape() {
  auto input_strategys = strategy()->GetInputDim();
  auto strategy_x = input_strategys.at(0);
  auto strategy_w1 = input_strategys.at(2);

  // (epert_dim, bs * seq_len, h, ffn_h)
  // (3,         2,            1,     0)
  dev_matrix_shape_ = {strategy_w1.at(0), strategy_x.at(0), strategy_w1.at(1), strategy_w1.at(2)};
  origin_dev_matrix_shape_ = dev_matrix_shape_;
  return SUCCESS;
}

Status MoeFFNInfo::InferTensorMap() {
  // x: [bs * seq_length, hidden]
  Shape x_tensor_map{2, 1};
  Shape expert_tensor_map{-1};
  // w1: [expert, hidden, ffn_hidden_size]
  Shape weight1_tensor_map{3, 1, 0};
  // b1: [expert, ffn_hidden_size]
  Shape bias1_tensor_map{3, 0};
  // w2: [expert, ffn_hidden_size, hidden]
  Shape weight2_tensor_map{3, 0, 1};

  inputs_tensor_map_.emplace_back(x_tensor_map);
  inputs_tensor_map_.emplace_back(expert_tensor_map);
  inputs_tensor_map_.emplace_back(weight1_tensor_map);
  inputs_tensor_map_.emplace_back(bias1_tensor_map);
  inputs_tensor_map_.emplace_back(weight2_tensor_map);

  // out: [bs * seq_length, hidden]
  Shape out_tensor_map{2, 1};
  outputs_tensor_map_.emplace_back(out_tensor_map);
  return SUCCESS;
}

Status MoeFFNInfo::InferForwardCommunication() {
  forward_op_.clear();
  size_t dimension = origin_dev_matrix_shape_.size();
  size_t relevant_dimension_index = LAST_INDEX(dimension);
  // Relevant dimension is not split and all reduce is not required,
  // need to use origin_dev_matrix_shape_ here, since the dev_matrix_shape_ will be changed if repeated calculation.
  if (origin_dev_matrix_shape_.at(relevant_dimension_index) == MIN_SLICE_NUM) {
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
  } else if (group_list.empty()) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required.";
    return SUCCESS;
  }

  Operator op;
  op = CreateAllReduceOp(REDUCE_OP_SUM, group_list[0].name());

  forward_op_.push_back(op);
  MS_LOG(INFO) << name_ << ": The group name of forward communication is " << group_list[0].name();
  return SUCCESS;
}

Status MoeFFNInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> MoeFFNInfo::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp_vector;
  return sp_vector;
}

REGISTER(MoeFFNInfo);
}  // namespace parallel
}  // namespace mindspore
