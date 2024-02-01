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

#include "frontend/parallel/ops_info/f_f_n_info.h"

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
namespace {
// FFN inputs
// x:       (bs * seq, h)
// weight1: (expert_dim, h, ffn_h)
// weight2: (expert_dim, ffn_h, h)
// expert:  (16)
// bias1:   (expert_dim, ffn_h)
// bias2:   (expert_dim, h)
// ------------------------------
// output:  (bs * seq, h)

// split strategy
// bs * seq is able to split.
// h is not able to split.
// ffn_h is able to split.
// expert_dim is able to split.

enum FFNInputIndex : size_t {
  kInputIndexX = 0,
  kInputIndexW1,
  kInputIndexW2,
  kInputIndexExpert,
  kInputIndexBias1,
  kInputIndexBias2,
  kInputScale,
  kInputOffset,
  kInputDeqScale1,
  kInputDeqScale2,
  kInputAntiquantScale1,
  kInputAntiquantScale2,
  kInputAntiquantOffset1,
  kInputAntiquantOffset2,
};

auto GetStrategy = [](const Dimensions &strategy, int64_t org_dim) {
  auto dim = org_dim;
  if (dim < 0) {
    dim += strategy.size();
  }
  MS_EXCEPTION_IF_CHECK_FAIL(
    dim < SizeToLong(strategy.size()),
    "Failed to get strategy, dim index " + std::to_string(org_dim) + ", size " + std::to_string(strategy.size()));
  return strategy[dim];
};
}  // namespace

void FFNInfo::InitInputsExist() {
  if (cnode_ == nullptr) {
    return;
  }
  if (!inputs_exist_.empty()) {
    return;
  }
  std::vector<AnfNodePtr> all_inputs = cnode_->inputs();
  size_t inputs_size = all_inputs.size();
  for (size_t i = 1; i < inputs_size; ++i) {
    Shapes input_shapes;
    AnfNodePtr input = all_inputs[i];
    if (HasAbstractMonad(input)) {
      continue;
    }
    if (IsValueNode<None>(input)) {
      inputs_exist_.push_back(false);
    } else {
      inputs_exist_.push_back(true);
    }
  }
}

bool FFNInfo::IsInputExist(size_t index) {
  InitInputsExist();
  if (index >= inputs_exist_.size()) {
    return false;
  }
  return inputs_exist_[index];
}

size_t FFNInfo::GetStrategyRealIndex(size_t index) {
  InitInputsExist();
  if (index >= inputs_exist_.size()) {
    return UINT32_MAX;
  }
  if (!inputs_exist_[index]) {
    return UINT32_MAX;
  }
  size_t real_index = index;
  for (size_t i = 0; i < index; i++) {
    if (!inputs_exist_[i]) {
      real_index--;
    }
  }
  return real_index;
}

Status FFNInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto input_strategies = strategy->GetInputDim();
  auto strategy_x = input_strategies.at(kInputIndexX);    // (bs * seq, h) (1,1)
  auto strategy_w1 = input_strategies.at(kInputIndexW1);  // (expert_dim, h, ffn_h) (1,1,4)
  auto strategy_w2 = input_strategies.at(kInputIndexW2);  // (expert_dim, ffn_h, h) (1,4,1)
  // hidden_size dim must be 1, not able to split.
  if (GetStrategy(strategy_x, -1) != 1 || GetStrategy(strategy_x, -1) != GetStrategy(strategy_w1, -2) ||
      GetStrategy(strategy_w1, -2) != GetStrategy(strategy_w2, -1)) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The hidden size dim must be 1 at the same time, but got"
                  << " x's strategy: " << strategy_x << " weight1's strategy: " << strategy_w1
                  << " weight2's strategy: " << strategy_w2;
    return FAILED;
  }
  if (std::any_of(strategy_x.begin(), strategy_x.end(), [](auto dim) { return dim != 1; })) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy, all dimension of x's strategy must be 1,but got " << strategy_x;
    return FAILED;
  }

  // ffn_hidden_size dim must be the same strategy.
  if (GetStrategy(strategy_w1, -2) != GetStrategy(strategy_w2, -1)) {
    MS_LOG(ERROR) << name_ << " Invalid strategy: The ffn hidden size dim must be shard at the same time, but got"
                  << " strategy_w1's strategy: " << strategy_w1 << " strategy_w2's strategy: " << strategy_w2;
    return FAILED;
  }
  bool with_moe = IsInputExist(kInputIndexExpert);
  size_t expert_index = GetStrategyRealIndex(kInputIndexExpert);
  if (input_strategies.size() > expert_index) {  // with_moe
    // expert_dim must be the same strategy.
    if (GetStrategy(strategy_w1, 0) != 1 || GetStrategy(strategy_w1, 0) != GetStrategy(strategy_w2, 0)) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy: The expert dim must be 1 at the same time, but got"
                    << " strategy_w1's strategy: " << strategy_w1 << " strategy_w2's strategy: " << strategy_w2;
      return FAILED;
    }
    auto strategy_expert = input_strategies.at(kInputIndexExpert);  // (1)
    if (GetStrategy(strategy_expert, 0) != 1) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy: The expert can't be shard, but got"
                    << " expert's strategy: " << strategy_expert;
      return FAILED;
    }
  }
  size_t bias1_index = GetStrategyRealIndex(kInputIndexBias1);
  if (input_strategies.size() > bias1_index) {
    auto strategy_bias1 = input_strategies.at(bias1_index);  // (expert_dim, ffn_h) (1,4)
    if (with_moe && GetStrategy(strategy_w1, 0) != GetStrategy(strategy_bias1, 0)) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy: The expert dim must be 1 at the same time, but got"
                    << " strategy_w1's strategy: " << strategy_w1 << " strategy_bias1's strategy: " << strategy_bias1;
      return FAILED;
    }
    if (GetStrategy(strategy_w1, -1) != GetStrategy(strategy_bias1, -1)) {
      MS_LOG(ERROR) << name_ << " Invalid strategy: The ffn hidden size dim must be shard at the same time, but got"
                    << " strategy_w1's strategy: " << strategy_w1 << " strategy_bias1's strategy: " << strategy_bias1;
      return FAILED;
    }
  }
  size_t bias2_index = GetStrategyRealIndex(kInputIndexBias2);
  if (input_strategies.size() > bias2_index) {
    auto strategy_bias2 = input_strategies.at(bias2_index);  // (expert_dim, h) (1,1)
    if (with_moe && GetStrategy(strategy_w2, 0) != GetStrategy(strategy_bias2, 0)) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy: The expert dim must be 1 at the same time, but got"
                    << " strategy_w2's strategy: " << strategy_w2 << " strategy_bias2's strategy: " << strategy_bias2;
      return FAILED;
    }
    if (GetStrategy(strategy_w2, -1) != GetStrategy(strategy_bias2, -1)) {
      MS_LOG(ERROR) << name_ << " Invalid strategy: The hidden size dim must be shard at the same time, but got"
                    << " strategy_w2's strategy: " << strategy_w2 << " strategy_bias2's strategy: " << strategy_bias2;
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FFNInfo::InferDevMatrixShape() {
  auto input_strategies = strategy()->GetInputDim();
  auto strategy_x = input_strategies.at(0);
  auto strategy_w1 = input_strategies.at(1);

  // (expert_dim, bs * seq_len, h, ffn_h)  or (bs * seq_len, h, ffn_h)
  // (3,          2,            1,     0)  or (2,            1,     0)
  //
  dev_matrix_shape_.clear();
  bool with_moe = IsInputExist(kInputIndexExpert);
  if (with_moe) {
    dev_matrix_shape_.push_back(GetStrategy(strategy_w1, 0));
  }
  for (size_t i = 0; i < strategy_x.size(); i++) {
    dev_matrix_shape_.push_back(strategy_x[i]);
  }
  dev_matrix_shape_.push_back(GetStrategy(strategy_w1, -1));
  origin_dev_matrix_shape_ = dev_matrix_shape_;
  return SUCCESS;
}

Status FFNInfo::InferTensorMap() {
  size_t expert_index = GetStrategyRealIndex(kInputIndexExpert);
  size_t bias1_index = GetStrategyRealIndex(kInputIndexBias1);
  size_t bias2_index = GetStrategyRealIndex(kInputIndexBias2);
  size_t antiquant_scale1 = GetStrategyRealIndex(kInputAntiquantScale1);
  size_t antiquant_scale2 = GetStrategyRealIndex(kInputAntiquantScale2);
  size_t antiquant_offset1 = GetStrategyRealIndex(kInputAntiquantOffset1);
  size_t antiquant_offset2 = GetStrategyRealIndex(kInputAntiquantOffset2);
  // x: [bs * seq_length, hidden]
  Shape x_tensor_map;  // [...,5,4,3,2,1]
  for (size_t i = inputs_shape_[0].size(); i > 0; i--) {
    x_tensor_map.push_back(i);
  }
  if (inputs_shape_.size() > expert_index) {
    int64_t expert_pos = SizeToLong(x_tensor_map.size()) + 1;
    // w1: [expert, hidden, ffn_hidden_size]
    Shape weight1_tensor_map{expert_pos, 1, 0};
    // w2: [expert, ffn_hidden_size, hidden]
    Shape weight2_tensor_map{expert_pos, 0, 1};
    Shape expert_tensor_map{-1};
    // b1: [expert, ffn_hidden_size]
    Shape bias1_tensor_map{expert_pos, 0};
    // b1: [expert, hidden_size]
    Shape bias2_tensor_map{expert_pos, 1};
    // aq scale1: [expert, ffn_hidden_size]
    Shape antiquant_scale1_tensor_map{expert_pos, 0};
    // aq scale2: [expert, hidden_size]
    Shape antiquant_scale2_tensor_map{expert_pos, 1};
    // aq offset1: [expert, ffn_hidden_size]
    Shape antiquant_offset1_tensor_map{expert_pos, 0};
    // aq offset2: [expert, hidden_size]
    Shape antiquant_offset2_tensor_map{expert_pos, 1};

    inputs_tensor_map_.emplace_back(x_tensor_map);
    inputs_tensor_map_.emplace_back(weight1_tensor_map);
    inputs_tensor_map_.emplace_back(weight2_tensor_map);
    inputs_tensor_map_.emplace_back(expert_tensor_map);
    if (inputs_shape_.size() > bias1_index) {
      inputs_tensor_map_.emplace_back(bias1_tensor_map);
    }
    if (inputs_shape_.size() > bias2_index) {
      inputs_tensor_map_.emplace_back(bias2_tensor_map);
    }
    if (inputs_shape_.size() > antiquant_scale1) {
      inputs_tensor_map_.emplace_back(antiquant_scale1_tensor_map);
    }
    if (inputs_shape_.size() > antiquant_scale2) {
      inputs_tensor_map_.emplace_back(antiquant_scale2_tensor_map);
    }
    if (inputs_shape_.size() > antiquant_offset1) {
      inputs_tensor_map_.emplace_back(antiquant_offset1_tensor_map);
    }
    if (inputs_shape_.size() > antiquant_offset2) {
      inputs_tensor_map_.emplace_back(antiquant_offset2_tensor_map);
    }
  } else {
    // w1: [hidden, ffn_hidden_size]
    Shape weight1_tensor_map{1, 0};
    // w2: [ffn_hidden_size, hidden]
    Shape weight2_tensor_map{0, 1};
    Shape expert_tensor_map{-1};
    // b1: [ffn_hidden_size]
    Shape bias1_tensor_map{0};
    // b1: [hidden_size]
    Shape bias2_tensor_map{1};

    inputs_tensor_map_.emplace_back(x_tensor_map);
    inputs_tensor_map_.emplace_back(weight1_tensor_map);
    inputs_tensor_map_.emplace_back(weight2_tensor_map);
    if (inputs_shape_.size() > bias1_index) {
      inputs_tensor_map_.emplace_back(bias1_tensor_map);
    }
    if (inputs_shape_.size() > bias2_index) {
      inputs_tensor_map_.emplace_back(bias2_tensor_map);
    }
  }
  // out: [bs * seq_length, hidden]
  Shape out_tensor_map = x_tensor_map;
  outputs_tensor_map_.emplace_back(out_tensor_map);
  return SUCCESS;
}

Status FFNInfo::InferForwardCommunication() {
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

Status FFNInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> FFNInfo::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp_vector;
  return sp_vector;
}

REGISTER(FFNInfo);
}  // namespace parallel
}  // namespace mindspore
