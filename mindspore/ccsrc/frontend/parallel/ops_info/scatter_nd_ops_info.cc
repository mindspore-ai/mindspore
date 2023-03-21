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

#include "frontend/parallel/ops_info/scatter_nd_ops_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"
#include "frontend/parallel/tensor_layout/shape_util.h"

namespace mindspore {
namespace parallel {
// The shape of input:   [A, B, C, D], the strategy of input: (a, b, c, d)
// The shape of indices: [Q, W, 2], the strategy of indices: (1, 1, 1)
// the look dim of indices whose value (x, y) should satisfy '0 <= x < A, 0 <= y < B'
// here the 2 respect to the size of [A, B]
// The shape of updates: [Q, W, C, D], the strategy of updates: (1, 1, c, d)
// The shape of output:  [A, B, C, D], the strategy of output: (a, b, c, d)
// The dev matrix: (a, b, 1, 1)
Status ScatterNdOpsInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 3) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 3";
    return FAILED;
  }

  auto indices_shape = inputs_shape_[1];
  gather_dims_size_ = indices_shape.back();
  if (CheckInputStrategy(stra[0]) != SUCCESS) {
    return FAILED;
  }
  if (!stra[1].empty() && std::accumulate(stra[1].begin(), stra[1].end(), 1, std::multiplies<int64_t>()) != 1) {
    MS_LOG(ERROR) << name_ << ": The indices can not be split";
    return FAILED;
  }

  if (stra[2].empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy[2] is empty";
    return FAILED;
  }

  if (std::accumulate(stra[2].begin(), stra[2].begin() + static_cast<different_type>(stra[1].size() - 1), 1,
                      std::multiplies<int64_t>()) != 1) {
    MS_LOG(ERROR) << name_ << ": The first " << (stra[1].size() - 1) << " dimensions of updates can not be split";
    return FAILED;
  }

  for (size_t i = 0; i < stra[0].size() - gather_dims_size_; ++i) {
    if (stra[0][i + gather_dims_size_] != stra[2][stra[1].size() - 1 + i]) {
      MS_LOG(ERROR)
        << name_ << ": updates.strategy must be equal to indices.strategy[:-1] + input.strategy[indices.strategy[-1]:]";
      return FAILED;
    }
  }

  for (size_t i = 0; i < gather_dims_size_; ++i) {
    if (stra[0][i] > 1) {
      do_replace_graph_ = true;
      break;
    }
  }

  return SUCCESS;
}

// The shape of input:   [A, B.., C, D], the strategy of input: (1, 1.., c, d)
Status ScatterNdUpdateInfo::CheckInputStrategy(const Dimensions &strategy_item) {
  for (size_t i = 0; i < gather_dims_size_; ++i) {
    if (strategy_item[i] != 1) {
      MS_LOG(ERROR) << "For " << name_
                    << ", "
                       "the input cannot be shard at gather dims input[:"
                    << gather_dims_size_ << "].";
      return FAILED;
    }
  }
  return SUCCESS;
}

// The shape of input:   [A, B.., C, D], the strategy of input: (1, 1.., c, d)
Status TensorScatterUpdateInfo::CheckInputStrategy(const Dimensions &strategy_item) {
  for (size_t i = 0; i < gather_dims_size_; ++i) {
    if (strategy_item[i] != 1) {
      MS_LOG(ERROR) << "For ScatterNdUpdate/TensorScatterUpdate, "
                       "the input cannot be shard at gather dims input[:"
                    << gather_dims_size_ << "].";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ScatterNdOpsInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

// The shape of input:   [A, B, C, D], the strategy of input: (a, b, c, d), tensor_map: (3, 2, 1, 0)
// The shape of indices: [Q, W, 2], the strategy of indices: (1, 1, 1), tensor_map: (-1, -1, -1)
// The shape of updates: [Q, W, C, D], the strategy of updates: (1, 1, c, d), tensor_map: (-1, -1, 1, 0)
// The shape of output:  [A, B, C, D], the strategy of output: (a, b, c, d), tensor_map: (3, 2, 1, 0)
// The dev matrix: (a, b, c, d)
Status ScatterNdOpsInfo::InferTensorMap() {
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

  // updates_tensor_map = indices_tensor_map[:-1] + input_tensor_map[gather_dim_size_:]
  updates_tensor_map = indices_tensor_map;
  updates_tensor_map.pop_back();
  for (size_t i = gather_dims_size_; i < input_tensor_map.size(); ++i) {
    updates_tensor_map.push_back(input_tensor_map[i]);
  }
  inputs_tensor_map_.push_back(input_tensor_map);    // input
  inputs_tensor_map_.push_back(indices_tensor_map);  // indices
  inputs_tensor_map_.push_back(updates_tensor_map);  // updates

  outputs_tensor_map_.push_back(input_tensor_map);
  return SUCCESS;
}

void ScatterNdOpsInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    split_flag_list_[i] = false;  // when set no strategy, going stand alone mode.
  }
}

ReplaceGraphPtr ScatterNdOpsInfo::replace_graph(const CNodePtr &cnode) {
  if (ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " replace graph failed";
  }
  return replace_graph_;
}

// The shape of input:   [A, B, C, D], the strategy of input: (a, b, c, d)
// The shape of indices: [Q, W, 2], the strategy of indices: (1, 1, 1)
// the look dim of indices whose value (x, y) should satisfy '0 <= x < A, 0 <= y < B'
// The shape of updates: [Q, W, C, D], the strategy of updates: (1, 1, c, d)
// when splitting [A, B], doing replace graph
Status ScatterNdOpsInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  if (!do_replace_graph_) {
    return SUCCESS;
  }
  if (gen_g_.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  auto anf_node_list = PrepareReplaceGraph();
  // {indices_sub, mul, div, dtype};
  auto indices_sub = anf_node_list[0];
  auto mul = anf_node_list[1];
  auto div = anf_node_list[2];
  auto dtype = anf_node_list[3];
  auto info_position = name_.find("Info");
  if (info_position == std::string::npos) {
    MS_LOG(EXCEPTION) << "The name " << name_ << " dose not contain 'Info'";
  }
  auto node_name = name_.substr(0, info_position);
  auto scatter_ops = gen_g_.PushBack({gen_g_.NewOpInst(node_name), gen_g_.virtual_input_node(), indices_sub, mul});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(scatter_ops, 1), std::make_pair(div, 2),
                                                             std::make_pair(indices_sub, 2), std::make_pair(mul, 3),
                                                             std::make_pair(dtype, 3)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, scatter_ops));

  return SUCCESS;
}

// The shape of input:   [A, B, C, D], the strategy of input: (a, b, c, d)
// The shape of indices: [Q, W, 2], the strategy of indices: (1, 1, 1)
// the look dim of indices whose value (x, y) should satisfy '0 <= x < A, 0 <= y < B'
// The shape of updates: [Q, W, C, D], the strategy of updates: (1, 1, c, d)
// when splitting [A, B], doing replace graph
Status ScatterNdMulDivBaseInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  if (!do_replace_graph_) {
    return SUCCESS;
  }
  if (gen_g_.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  auto anf_node_list = PrepareReplaceGraph();
  // {indices_sub, mul, div, dtype};
  auto indices_sub = anf_node_list[0];
  auto mul = anf_node_list[1];
  auto div = anf_node_list[2];
  auto dtype = anf_node_list[3];
  auto reshape_updates_mask = anf_node_list[4];
  auto reverse_sub = gen_g_.PushBack(
    {gen_g_.NewOpInst(SUB), NewValueNode(std::make_shared<tensor::Tensor>(1.0, kFloat32)), reshape_updates_mask});
  auto add_mask = gen_g_.PushBack({gen_g_.NewOpInst(ADD), mul, reverse_sub});
  auto info_position = name_.find("Info");
  if (info_position == std::string::npos) {
    MS_LOG(EXCEPTION) << "The name " << name_ << " dose not contain 'Info'";
  }
  auto node_name = name_.substr(0, info_position);
  auto scatter_ops = gen_g_.PushBack({gen_g_.NewOpInst(node_name), gen_g_.virtual_input_node(), indices_sub, add_mask});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(scatter_ops, 1), std::make_pair(div, 2),
                                                             std::make_pair(indices_sub, 2), std::make_pair(mul, 3),
                                                             std::make_pair(dtype, 3)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, scatter_ops));

  return SUCCESS;
}

// The shape of input:   [A, B, C, D], the strategy of input: (a, b, c, d)
// The shape of indices: [Q, W, 2], the strategy of indices: (1, 1, 1)
// the look dim of indices whose value (x, y) should satisfy '0 <= x < A, 0 <= y < B'
// The shape of updates: [Q, W, C, D], the strategy of updates: (1, 1, c, d)
// when splitting [A, B], doing replace graph
std::vector<AnfNodePtr> ScatterNdOpsInfo::PrepareReplaceGraph() {
  auto rank_in_stage = g_device_manager->rank_index_in_stage();
  MS_LOG(INFO) << name_ << ": The rank is " << rank_in_stage;
  auto input_slice_shape = inputs_tensor_info_[0].slice_shape();
  Shape indices_slice_value;
  (void)std::copy(input_slice_shape.begin(), input_slice_shape.begin() + static_cast<different_type>(gather_dims_size_),
                  std::back_inserter(indices_slice_value));
  auto indices_slice_value_tensor = std::make_shared<mindspore::tensor::Tensor>(indices_slice_value, kInt32);
  Shape indices_shape_size(inputs_shape_[1].size(), 1);
  indices_shape_size[indices_shape_size.size() - 1] = -1;
  auto reshape_indices_slice =
    gen_g_.PushBack({gen_g_.NewOpInst(RESHAPE), ValuePtrToAnfNodePtr(indices_slice_value_tensor),
                     NewValueNode(MakeValue(indices_shape_size))});
  auto div = gen_g_.PushBack({gen_g_.NewOpInst(FLOORDIV), gen_g_.virtual_input_node(), reshape_indices_slice});
  Shape dev_accum_shape;
  (void)ShapeToAccumulateProductReverse(dev_matrix_shape_, &dev_accum_shape);
  MS_LOG(INFO) << "The dev_matrix is :" << dev_matrix_shape_ << ", dev_accum_shape is :" << dev_accum_shape;
  size_t gather_begin_position = 0;
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    gather_begin_position = 1;
  }
  Shape accum_value;
  (void)std::copy(dev_accum_shape.begin() + static_cast<different_type>(gather_begin_position),
                  dev_accum_shape.begin() + static_cast<different_type>(gather_begin_position + gather_dims_size_),
                  std::back_inserter(accum_value));
  auto delta_value =
    std::accumulate(dev_accum_shape.begin() + static_cast<different_type>(gather_begin_position + gather_dims_size_),
                    dev_accum_shape.end(), 0, std::plus<int64_t>());
  auto accum_value_tensor = std::make_shared<mindspore::tensor::Tensor>(accum_value, kInt32);
  auto reshape_accum_value = gen_g_.PushBack(
    {gen_g_.NewOpInst(RESHAPE), ValuePtrToAnfNodePtr(accum_value_tensor), NewValueNode(MakeValue(indices_shape_size))});
  auto rank_mul = gen_g_.PushBack({gen_g_.NewOpInst(MUL), div, reshape_accum_value});
  auto indices_mul = gen_g_.PushBack({gen_g_.NewOpInst(MUL), div, reshape_indices_slice});
  auto indices_sub = gen_g_.PushBack({gen_g_.NewOpInst(SUB), gen_g_.virtual_input_node(), indices_mul});
  auto reduce_sum = gen_g_.PushBack({gen_g_.NewOpInst(REDUCE_SUM), rank_mul, CreateInt32Tensor(-1)});
  auto sub = gen_g_.PushBack({gen_g_.NewOpInst(SUB), CreateInt32Tensor(rank_in_stage), reduce_sum});
  auto relu = gen_g_.PushBack({gen_g_.NewOpInst(RELU), sub});
  auto minimum = gen_g_.PushBack({gen_g_.NewOpInst(MINIMUM), relu, CreateInt32Tensor(delta_value)});
  auto equal = gen_g_.PushBack({gen_g_.NewOpInst(EQUAL), sub, minimum});
  auto dtype = gen_g_.PushBack({gen_g_.NewOpInst(DTYPE), gen_g_.virtual_input_node()});
  auto cast = gen_g_.PushBack({gen_g_.NewOpInst(CAST), equal, dtype});
  Shape update_shapes(inputs_shape_[2]);
  for (size_t i = inputs_shape_[1].size() - 1; i < update_shapes.size(); ++i) {
    update_shapes[i] = 1;
  }
  auto reshape_updates_mask =
    gen_g_.PushBack({gen_g_.NewOpInst(RESHAPE), cast, NewValueNode(MakeValue(update_shapes))});
  auto mul = gen_g_.PushBack({gen_g_.NewOpInst(MUL), gen_g_.virtual_input_node(), reshape_updates_mask});
  return {indices_sub, mul, div, dtype, reshape_updates_mask};
}

Status ScatterNdOpsInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

// The shape of input:   [A, B, C, D], the strategy of input: (a, b, c, d)
// The shape of indices: [Q, W, 2], the strategy of indices: (1, 1, 1)
// The shape of updates: [Q, W, C, D], the strategy of updates: (1, 1, c, d)
// The shape of output:  [A, B, C, D], the strategy of output: (a, b, c, d)
// The dev matrix: (a, b, 1, 1)
std::vector<StrategyPtr> ScatterNdOpsInfo::GenerateOpStrategies(int64_t stage_id) {
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
    // updates_strategy = indices_strategy[:-1] + input_strategy[gather_dims_size_:]
    Dimensions updates_strategy = indices_strategy;
    updates_strategy.pop_back();
    for (size_t i = gather_dims_size_; i < first_input_strategy.size(); ++i) {
      updates_strategy.push_back(first_input_strategy[i]);
    }

    tmp_strategy.push_back(first_input_strategy);  // input
    tmp_strategy.push_back(indices_strategy);      // indices
    tmp_strategy.push_back(updates_strategy);      // updates
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

Status ScatterNdOpsInfo::InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
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
  auto scatter_nd_ops_cost = std::dynamic_pointer_cast<ScatterMathOpsCost>(operator_cost());
  MS_EXCEPTION_IF_NULL(scatter_nd_ops_cost);
  bool is_split_axis = std::find_if(param_strategy.begin(), param_strategy.begin() + gather_dims_size_,
                                    [](auto dim) { return dim > 1; }) != param_strategy.end();
  scatter_nd_ops_cost->set_is_split_axis(is_split_axis);
  scatter_nd_ops_cost->set_coefficient(1, 5, 2);
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

REGISTER(ScatterNdAddInfo);
REGISTER(ScatterNdSubInfo);
REGISTER(ScatterNdUpdateInfo);
REGISTER(TensorScatterUpdateInfo);
REGISTER(TensorScatterAddInfo);
REGISTER(TensorScatterSubInfo);
REGISTER(TensorScatterMulInfo);
REGISTER(TensorScatterDivInfo);
REGISTER(TensorScatterMaxInfo);
REGISTER(TensorScatterMinInfo);
}  // namespace parallel
}  // namespace mindspore
