/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/scatter_math_ops_info.h"

#include <memory>
#include <vector>
#include <functional>
#include <utility>
#include <algorithm>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
// The strategy of (input, indices) equal to Gather, the strategy of updates equal to Add(Gather, updates)
// Thus, it can support row-split/column-split/row-column-split
Status ScatterMathOpsInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();

  if (stra[0].size() - 1 != stra[2].size() - stra[1].size()) {
    MS_LOG(ERROR) << name_ << ": updates.strategy size must be equal to (indices.strategy + input.strategy[1:]) size";
    return FAILED;
  }

  if (!stra[1].empty() && std::accumulate(stra[1].begin(), stra[1].end(), 1, std::multiplies<int64_t>()) != 1) {
    MS_LOG(ERROR) << name_ << ": The indices can not be split";
    return FAILED;
  }

  for (size_t j = 0; j < stra[1].size(); ++j) {
    if (stra[2][j] != 1) {
      MS_LOG(ERROR) << name_ << ": updates.strategy must be equal to indices.strategy + input.strategy[1:]";
      return FAILED;
    }
  }

  for (size_t i = 1; i < stra[0].size(); ++i) {
    if (stra[0][i] != stra[2][stra[1].size() + i - 1]) {
      MS_LOG(ERROR) << name_ << ": updates.strategy must be equal to indices.strategy + input.strategy[1:]";
      return FAILED;
    }
  }

  if (stra[0][0] > 1) {
    do_replace_graph_ = true;
  }
  return SUCCESS;
}

Status ScatterMathOpsInfo::InferTensorMap() {
  if (inputs_shape_.size() != 3) {
    MS_LOG(ERROR) << name_ << "The size of inputs shape must be 3";
    return FAILED;
  }

  TensorMap input_tensor_map, updates_tensor_map;
  TensorMap indices_tensor_map(inputs_shape_[1].size(), MAP_NONE);

  // cannot use dev_matrix_shape_ replace inputs_shape_[0], because it may not be fully split in all devices.
  int64_t size = SizeToLong(inputs_shape_[0].size());
  for (int64_t i = 0; i < size; ++i) {
    input_tensor_map.push_back(size - i - 1);
  }

  // updates_tensor_map = indices_tensor_map + input_tensor_map[1:]
  updates_tensor_map = indices_tensor_map;
  for (size_t i = 1; i < input_tensor_map.size(); ++i) {
    updates_tensor_map.push_back(input_tensor_map[i]);
  }
  inputs_tensor_map_.push_back(input_tensor_map);    // input
  inputs_tensor_map_.push_back(indices_tensor_map);  // indices
  inputs_tensor_map_.push_back(updates_tensor_map);  // updates

  outputs_tensor_map_.push_back(input_tensor_map);
  return SUCCESS;
}

Status ScatterMathOpsInfo::InferBias() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->rank_index_in_stage();
  auto input_shape = inputs_shape_.at(0);
  auto input_strategy = strategy_->GetInputDim().at(0);
  // axis don't split
  if (input_strategy[0] == 1) {
    bias_ = 0;
    return SUCCESS;
  }

  slice_size_ = input_shape.at(0) / input_strategy.at(0);
  int64_t input_shard_num =
    std::accumulate(input_strategy.begin(), input_strategy.end(), 1, std::multiplies<int64_t>());
  int64_t input_column_shard_num = input_shard_num / input_strategy[0];
  // if repeated calculation, because the repeated num in the right of dev-matrix, so rank need to div repeated num
  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      rank = rank / repeated_calc_num_;
    } else {
      rank = rank % input_shard_num;
    }
  }

  bias_ = rank / input_column_shard_num * slice_size_;
  return SUCCESS;
}

ReplaceGraphPtr ScatterMathOpsInfo::replace_graph(const CNodePtr &cnode) {
  if (ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " replace graph failed";
  }
  return replace_graph_;
}

Status ScatterMathOpsInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  if (!do_replace_graph_) {
    return SUCCESS;
  }
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  if (InferBias() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer Bias failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": The rank is " << g_device_manager->rank_index_in_stage() << ", the bias is " << bias_;
  auto sub = gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), CreateInt32Tensor(bias_)});
  auto relu = gen_g.PushBack({gen_g.NewOpInst(RELU), sub});
  auto minimum = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu, CreateInt32Tensor(slice_size_ - 1)});
  auto equal = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub, minimum});
  auto dtype = gen_g.PushBack({gen_g.NewOpInst(DTYPE), gen_g.virtual_input_node()});
  auto cast = gen_g.PushBack({gen_g.NewOpInst(CAST), equal, dtype});
  std::vector<int64_t> mask_shape = inputs_shape_[1];
  (void)mask_shape.insert(mask_shape.end(), inputs_shape_[2].size() - inputs_shape_[1].size(), 1);
  auto reshape = gen_g.PushBack({gen_g.NewOpInst(RESHAPE), cast, NewValueNode(MakeValue(mask_shape))});
  auto sub_mask =
    gen_g.PushBack({gen_g.NewOpInst(SUB), NewValueNode(std::make_shared<tensor::Tensor>(1.0, kFloat32)), reshape});
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), gen_g.virtual_input_node(), reshape});
  auto add_mask = gen_g.PushBack({gen_g.NewOpInst(ADD), mul, sub_mask});
  auto info_position = name_.find("Info");
  if (info_position == std::string::npos) {
    MS_LOG(EXCEPTION) << "The name " << name_ << " dose not contain 'Info'";
  }
  auto node_name = name_.substr(0, info_position);
  auto scatter_ops = gen_g.PushBack({gen_g.NewOpInst(node_name), gen_g.virtual_input_node(), minimum, add_mask});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(scatter_ops, 1), std::make_pair(sub, 2),
                                                             std::make_pair(mul, 3), std::make_pair(dtype, 3)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, scatter_ops));

  return SUCCESS;
}

Status ScatterAddInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  if (!do_replace_graph_) {
    return SUCCESS;
  }
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  if (InferBias() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer Bias failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": The rank is " << g_device_manager->rank_index_in_stage() << ", the bias is " << bias_;
  auto sub = gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), CreateInt32Tensor(bias_)});
  auto relu = gen_g.PushBack({gen_g.NewOpInst(RELU), sub});
  auto minimum = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu, CreateInt32Tensor(slice_size_ - 1)});
  auto equal = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub, minimum});
  auto dtype = gen_g.PushBack({gen_g.NewOpInst(DTYPE), gen_g.virtual_input_node()});
  auto cast = gen_g.PushBack({gen_g.NewOpInst(CAST), equal, dtype});
  std::vector<int64_t> mask_shape = inputs_shape_[1];
  (void)mask_shape.insert(mask_shape.end(), inputs_shape_[2].size() - inputs_shape_[1].size(), 1);
  auto reshape = gen_g.PushBack({gen_g.NewOpInst(RESHAPE), cast, NewValueNode(MakeValue(mask_shape))});
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), gen_g.virtual_input_node(), reshape});
  auto info_position = name_.find("Info");
  if (info_position == std::string::npos) {
    MS_LOG(EXCEPTION) << "The name " << name_ << " dose not contain 'Info'";
  }
  auto node_name = name_.substr(0, info_position);
  auto scatter_ops = gen_g.PushBack({gen_g.NewOpInst(node_name), gen_g.virtual_input_node(), minimum, mul});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(scatter_ops, 1), std::make_pair(sub, 2),
                                                             std::make_pair(mul, 3), std::make_pair(dtype, 3)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, scatter_ops));

  return SUCCESS;
}

std::vector<StrategyPtr> ScatterMathOpsInfo::GenerateOpStrategies(int64_t stage_id) {
  // to generate the first input's strategy
  Shape input_split(inputs_shape_[0].size(), 1);
  Shapes splittable_input = {input_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  // the others strategies are equal to the first input's strategy
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }
    Strategies tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    Dimensions indices_strategy(inputs_shape_[1].size(), 1);
    // updates_strategy = indices_strategy + input_strategy[1:]
    Dimensions updates_strategy = indices_strategy;
    for (size_t i = 1; i < first_input_strategy.size(); ++i) {
      updates_strategy.push_back(first_input_strategy[i]);
    }

    tmp_strategy.push_back(first_input_strategy);  // input
    tmp_strategy.push_back(indices_strategy);      // indices
    tmp_strategy.push_back(updates_strategy);      // updates
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

Status ScatterMathOpsInfo::InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  if (InitForCostModelWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    }
    return FAILED;
  }
  auto param_strategy = strategy_->GetInputDim().at(0);
  // cost model set axis and strategy
  auto scatter_ops_cost = std::dynamic_pointer_cast<ScatterMathOpsCost>(operator_cost());
  scatter_ops_cost->set_is_split_axis((param_strategy[0] > 1));
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

REGISTER(ScatterAddInfo);
REGISTER(ScatterSubInfo);
REGISTER(ScatterMulInfo);
REGISTER(ScatterDivInfo);
}  // namespace parallel
}  // namespace mindspore
