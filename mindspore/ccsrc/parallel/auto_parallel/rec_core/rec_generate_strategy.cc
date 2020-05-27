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

#include "parallel/auto_parallel/rec_core/rec_generate_strategy.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "ir/value.h"
#include "parallel/auto_parallel/rec_core/rec_parse_graph.h"
#include "parallel/auto_parallel/rec_core/rec_partition.h"
#include "parallel/ops_info/operator_info.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
void GenerateStrategy(std::shared_ptr<Graph> graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                      const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list,
                      const std::vector<std::vector<std::string>> &input_tensor_names,
                      const std::shared_ptr<std::vector<size_t>> index_list) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(eli_list);
  MS_EXCEPTION_IF_NULL(index_list);
  GeneratePartitionedOperatorStrategy(graph, ops, index_list);
  std::shared_ptr<std::vector<size_t>> no_stra_op_list(new std::vector<size_t>);
  GenerateEliminatedOperatorStrategyForward(graph, ops, eli_list, input_tensor_names, index_list, no_stra_op_list);
  GenerateEliminatedOperatorStrategyBackward(ops, input_tensor_names, no_stra_op_list);
}

std::vector<std::vector<int32_t>> PrepareMatMul(const std::shared_ptr<Graph> &graph,
                                                const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const size_t iter_graph, const size_t iter_ops) {
  std::vector<std::vector<int32_t>> strategies;
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    std::vector<int32_t> s;
    auto attrs = ops[iter_ops]->attrs();
    bool transpose_a = attrs[TRANSPOSE_A]->cast<BoolImmPtr>()->value();
    bool transpose_b = attrs[TRANSPOSE_B]->cast<BoolImmPtr>()->value();
    if (transpose_a && (iter_op_inputs == 0)) {
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
    } else if (transpose_b && (iter_op_inputs == 1)) {
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
    } else {
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    }
    strategies.push_back(s);
  }
  return strategies;
}

std::vector<std::vector<int32_t>> PrepareVirtualDataset(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                        const size_t iter_ops) {
  std::vector<std::vector<int32_t>> strategies = MakeDataParallelStrategy(ops, iter_ops);
  strategies[1][0] = strategies[0][0];
  return strategies;
}

std::vector<std::vector<int32_t>> PrepareScalarInputOperator(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                             const size_t iter_ops, std::vector<int32_t> s) {
  std::vector<std::vector<int32_t>> strategies;

  auto dev_num = g_device_manager->DeviceNum();
  size_t cut_num = 1;
  for (size_t iter_s = 0; iter_s < s.size(); iter_s++) {
    cut_num *= s[iter_s];
  }
  if (cut_num != dev_num) {
    std::vector<int32_t> s_max = s;
    for (size_t dim = 0; dim < (size_t)ops[iter_ops]->inputs_tensor_info()[0].shape().size(); dim++) {
      size_t shape = ops[iter_ops]->inputs_tensor_info()[0].shape()[dim] / s[dim];
      while (cut_num < dev_num && shape % 2 == 0) {
        shape = shape / 2;
        s_max[dim] = s_max[dim] * 2;
        cut_num = cut_num * 2;
      }
      if (cut_num == dev_num) {
        break;
      }
    }
    s = s_max;
  }

  strategies.push_back(s);
  std::vector<int32_t> s_biasadd;
  s_biasadd.push_back(s[1]);
  strategies.push_back(s_biasadd);

  return strategies;
}

std::vector<std::vector<int32_t>> PrepareOneHot(std::vector<int32_t> s) {
  std::vector<std::vector<int32_t>> strategies;
  std::vector<int32_t> s_empty = {};
  strategies.push_back(s);
  strategies.push_back(s_empty);
  strategies.push_back(s_empty);
  return strategies;
}

std::vector<std::vector<int32_t>> MakeRecSearchStrategy(const std::shared_ptr<Graph> &graph,
                                                        const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                        const size_t iter_graph, const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  StrategyPtr origin_strategy = ops[iter_ops]->strategy();

  std::vector<std::vector<int32_t>> strategies;
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    if (iter_op_inputs >= origin_strategy->GetInputDim().size()) {
      MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
    }

    // size_t output_size = ops[iter_ops]->outputs_tensor_info()[0].shape().size();
    size_t output_size = origin_strategy->GetInputDim()[iter_op_inputs].size();

    std::vector<int32_t> s;
    if (output_size == 4) {
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_n));
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_c));
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (output_size == 2) {
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (output_size == 1) {
      s.push_back(
        static_cast<int32_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (output_size == 0) {
      s = {};
    } else {
      MS_LOG(ERROR) << "Tensor's output size is unexcepted.";
    }

    strategies.push_back(s);
  }
  return strategies;
}

std::vector<std::vector<int32_t>> MakeDataParallelStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                           const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  StrategyPtr origin_strategy = ops[iter_ops]->strategy();

  std::vector<std::vector<int32_t>> strategies;
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    if (iter_op_inputs >= origin_strategy->GetInputDim().size()) {
      MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
    }

    std::vector<int32_t> s;
    size_t input_size = origin_strategy->GetInputDim()[iter_op_inputs].size();
    for (size_t dim = 0; dim < input_size; dim++) {
      if (input_size == 1 || input_size == 2 || input_size == 4) {
        if (dim == 0) {
          size_t max_device_num = g_device_manager->DeviceNum();
          size_t target_tensor_batch = ops[iter_ops]->outputs_tensor_info()[0].shape()[0];
          s.push_back(std::min(max_device_num, target_tensor_batch));
        } else {
          s.push_back(1);
        }
      } else {
        MS_LOG(ERROR) << "Tensor's shape is unknown.";
      }
    }

    strategies.push_back(s);
  }
  return strategies;
}

std::vector<std::vector<int32_t>> PrepareStrategy(const std::shared_ptr<Graph> &graph,
                                                  const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                  const size_t iter_graph, const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  auto type = ops[iter_ops]->type();
  if (type == VIRTUAL_DATA_SET) {
    return PrepareVirtualDataset(ops, iter_ops);
  }
  auto idx = DictOpType.find(type);
  if (idx == DictOpType.end()) {
    return MakeDataParallelStrategy(ops, iter_ops);
  }

  if (type == MATMUL) {
    return PrepareMatMul(graph, ops, iter_graph, iter_ops);
  } else if (type == RESHAPE) {
    return MakeDataParallelStrategy(ops, iter_ops);
  } else {
    return MakeRecSearchStrategy(graph, ops, iter_graph, iter_ops);
  }
}

void GeneratePartitionedOperatorStrategy(const std::shared_ptr<Graph> graph,
                                         const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                         const std::shared_ptr<std::vector<size_t>> index_list) {
  for (size_t iter_ops = 0; iter_ops < (size_t)index_list->size(); iter_ops++) {
    std::vector<std::vector<int32_t>> strategies;
    size_t iter_graph = index_list->at(iter_ops);
    if (iter_graph == SIZE_MAX) {
      StrategyPtr sp = std::make_shared<Strategy>(0, strategies);
      ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
      continue;
    }
    strategies = PrepareStrategy(graph, ops, iter_graph, iter_ops);
    StrategyPtr sp = std::make_shared<Strategy>(0, strategies);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }
}

int FindIndexOfOperatorIncoming(const std::vector<std::vector<std::string>> &input_tensor_names,
                                const size_t iter_ops) {
  int incoming_op_index = -1;
  for (size_t i = 1; i < (size_t)input_tensor_names[iter_ops].size(); i++) {
    for (size_t j = 0; j < (size_t)input_tensor_names.size(); j++) {
      if (input_tensor_names[iter_ops][i] == input_tensor_names[j][0]) {
        incoming_op_index = j;
        break;
      }
    }
    if (incoming_op_index != -1) {
      break;
    }
  }
  return incoming_op_index;
}

std::vector<int32_t> CopyIncomingOperatorOutputStrategy(const std::shared_ptr<Graph> graph,
                                                        const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                        const size_t iter_ops, const size_t iter_graph) {
  std::vector<int32_t> s;
  for (auto input : ops[iter_ops]->inputs_tensor_info()) {
    auto input_stra_dim = input.shape().size();
    if (input_stra_dim == 0) {
      continue;
    }
    if (input_stra_dim == 1) {
      s.push_back(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_w);
    } else if (input_stra_dim == 2) {
      s.push_back(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_h);
      s.push_back(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_w);
    } else if (input_stra_dim == 4) {
      s.push_back(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_n);
      s.push_back(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_c);
      s.push_back(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_h);
      s.push_back(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_w);
    } else {
      MS_LOG(ERROR) << "Tensor's shape is unknown.";
    }
    break;
  }
  return s;
}

std::vector<int32_t> PrepareIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                          const int incoming_op_index) {
  std::vector<int32_t> s;
  auto strategy = ops[incoming_op_index]->selected_strategy();
  if (strategy->GetInputNumber() == 0) {
    return s;
  }
  for (size_t i = 0; i < (size_t)ops[incoming_op_index]->inputs_tensor_info().size(); i++) {
    if (ops[incoming_op_index]->inputs_tensor_info()[i].shape().size() == 0) {
      continue;
    }
    for (size_t j = 0; j < ops[incoming_op_index]->inputs_tensor_info()[i].shape().size(); ++j) {
      s.push_back(strategy->GetInputDim()[i][j]);
    }
    break;
  }
  return s;
}

std::vector<int32_t> GetAxisList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const int iter_ops) {
  std::vector<int32_t> axis_list;
  auto axis_param = ops[iter_ops]->attrs().find(AXIS)->second;
  std::vector<ValuePtr> elements;
  if (axis_param->isa<ValueTuple>()) {
    elements = axis_param->cast<ValueTuplePtr>()->value();
  } else if (axis_param->isa<ValueList>()) {
    elements = axis_param->cast<ValueListPtr>()->value();
  } else {
    MS_LOG(EXCEPTION) << "Failure: Axis type is invalid, neither tuple nor list." << std::endl;
  }
  for (auto &element : elements) {
    if (!element->isa<Int32Imm>()) {
      MS_LOG(EXCEPTION) << "Failure: Dimension indexes is not Int32." << std::endl;
    }
    auto axis = element->cast<Int32ImmPtr>()->value();
    axis_list.push_back(axis);
  }
  return axis_list;
}

std::vector<int32_t> ModifyStrategyIfSqueezeIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                     const int incoming_op_index, std::vector<int32_t> s) {
  std::vector<int32_t> s_Squeeze;
  std::vector<int32_t> stra_dim_list;
  for (size_t i = 0; i < s.size(); i++) {
    stra_dim_list.push_back(i);
  }
  auto axis_list = GetAxisList(ops, incoming_op_index);
  for (auto axis : axis_list) {
    auto it = find(stra_dim_list.begin(), stra_dim_list.end(), axis);
    if (it == stra_dim_list.end()) {
      MS_LOG(EXCEPTION) << "Failure: Can not find dimension indexes in Axis." << std::endl;
    }
    if (ops[incoming_op_index]->inputs_tensor_info()[0].shape()[axis] != 1) {
      MS_LOG(EXCEPTION) << "Failure: Removed dimension's shape is not 1." << std::endl;
    }
    stra_dim_list.erase(it);
  }
  for (size_t i = 0; i < (size_t)stra_dim_list.size(); i++) {
    s_Squeeze.push_back(s[stra_dim_list[i]]);
  }
  return s_Squeeze;
}

std::vector<int32_t> GetDimList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops) {
  std::vector<int32_t> dim_list;
  bool keep_dims;
  if (!ops[iter_ops]->attrs().find(KEEP_DIMS)->second->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << "Failure: Parameter keep_dims is not a boolean value." << std::endl;
  }
  keep_dims = ops[iter_ops]->attrs().find(KEEP_DIMS)->second->cast<BoolImmPtr>()->value();
  if (keep_dims != false) {
    return dim_list;
  }
  auto input_value = ops[iter_ops]->input_value();
  auto input_dim = ops[iter_ops]->inputs_tensor_info()[0].shape().size();
  if (input_value.back()->isa<ValueTuple>()) {
    auto attr_axis = GetValue<std::vector<int>>(input_value.back());
    if (attr_axis.empty()) {
      MS_LOG(EXCEPTION) << "Failure: This output is a 0-D tensor." << std::endl;
    }
    for (auto &axis : attr_axis) {
      axis < 0 ? dim_list.push_back(axis + SizeToInt(input_dim)) : dim_list.push_back(axis);
    }
  } else if (input_value.back()->isa<Int32Imm>()) {
    int axis = GetValue<int>(input_value.back());
    axis < 0 ? dim_list.push_back(axis + SizeToInt(input_dim)) : dim_list.push_back(axis);
  } else {
    MS_LOG(EXCEPTION) << "Failure: Axis type is invalid." << std::endl;
  }
  return dim_list;
}

std::vector<int32_t> ModifyStrategyIfReduceIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                    const int incoming_op_index, std::vector<int32_t> s) {
  std::vector<int32_t> s_Reduce;
  std::vector<int32_t> axis_list;
  for (size_t i = 0; i < s.size(); i++) {
    axis_list.push_back(i);
  }
  auto dim_list = GetDimList(ops, incoming_op_index);
  for (auto axis : dim_list) {
    auto it = find(axis_list.begin(), axis_list.end(), axis);
    if (it == axis_list.end()) {
      MS_LOG(EXCEPTION) << "Failure: Can not find dimension indexes in Axis." << std::endl;
    }
    axis_list.erase(it);
  }
  for (size_t i = 0; i < (size_t)axis_list.size(); i++) {
    s_Reduce.push_back(s[axis_list[i]]);
  }
  return s_Reduce;
}

std::vector<int32_t> CopyIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                       const int incoming_op_index, const size_t iter_ops,
                                                       const std::shared_ptr<std::vector<size_t>> no_stra_op_list) {
  std::vector<int32_t> s;
  s = PrepareIncomingOperatorInputStrategy(ops, incoming_op_index);
  if (s.size() != 0) {
    if (ops[incoming_op_index]->type() == SQUEEZE) {
      s = ModifyStrategyIfSqueezeIncoming(ops, incoming_op_index, s);
    }
    if (ops[incoming_op_index]->type() == REDUCE_SUM || ops[incoming_op_index]->type() == REDUCE_MAX ||
        ops[incoming_op_index]->type() == REDUCE_MIN || ops[incoming_op_index]->type() == REDUCE_MEAN) {
      s = ModifyStrategyIfReduceIncoming(ops, incoming_op_index, s);
    }
  }
  return s;
}

std::vector<std::vector<int32_t>> GenerateStrategiesFromStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                                 const size_t iter_ops, std::vector<int32_t> s) {
  std::vector<int32_t> s_empty = {};
  std::vector<std::vector<int32_t>> stra;

  if (s.size() == 0) {
    for (size_t iter_op_inputs = 0; iter_op_inputs < (size_t)ops[iter_ops]->inputs_tensor_info().size();
         iter_op_inputs++) {
      stra.push_back(s);
    }
    return stra;
  }

  MS_EXCEPTION_IF_NULL(ops[iter_ops]);
  if (ops[iter_ops]->type() == BIAS_ADD || ops[iter_ops]->type() == PRELU) {
    return PrepareScalarInputOperator(ops, iter_ops, s);
  }
  if (ops[iter_ops]->type() == ONEHOT) {
    return PrepareOneHot(s);
  }

  auto dev_num = g_device_manager->DeviceNum();
  for (size_t iter_op_inputs = 0; iter_op_inputs < (size_t)ops[iter_ops]->inputs_tensor_info().size();
       iter_op_inputs++) {
    if (ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape().size() == 0) {
      stra.push_back(s_empty);
      continue;
    }

    size_t cut_num = 1;
    for (size_t iter_s = 0; iter_s < s.size(); iter_s++) {
      cut_num *= s[iter_s];
    }
    if (cut_num == dev_num) {
      std::vector<int32_t> s_1 = s;
      bool modified = false;
      for (size_t j = 0; j < (size_t)ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape().size(); j++) {
        if (ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape()[j] == 1) {
          s_1[j] = 1;
          modified = true;
        }
      }
      if (modified) {
        stra.push_back(s_1);
      } else {
        stra.push_back(s);
      }
      continue;
    }

    std::vector<int32_t> s_max = s;
    for (size_t dim = 0; dim < (size_t)ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape().size(); dim++) {
      size_t shape = ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape()[dim] / s[dim];
      while (cut_num < dev_num && shape % 2 == 0) {
        shape = shape / 2;
        s_max[dim] = s_max[dim] * 2;
        cut_num = cut_num * 2;
      }
      if (cut_num == dev_num) {
        break;
      }
    }

    stra.push_back(s_max);
  }
  return stra;
}

void GenerateEliminatedOperatorStrategyForward(const std::shared_ptr<Graph> graph,
                                               const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                               const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list,
                                               const std::vector<std::vector<std::string>> &input_tensor_names,
                                               const std::shared_ptr<std::vector<size_t>> index_list,
                                               const std::shared_ptr<std::vector<size_t>> no_stra_op_list) {
  for (int eli_index = eli_list->size() - 1; eli_index >= 0; eli_index--) {
    size_t iter_ops = eli_list->at(eli_index)[0];
    std::vector<std::vector<int32_t>> stra;
    std::vector<int32_t> s;
    int incoming_op_index = FindIndexOfOperatorIncoming(input_tensor_names, iter_ops);
    if (incoming_op_index != -1) {
      auto iter_graph = index_list->at(incoming_op_index);
      if (iter_graph != SIZE_MAX) {
        s = CopyIncomingOperatorOutputStrategy(graph, ops, iter_ops, iter_graph);
      } else {
        s = CopyIncomingOperatorInputStrategy(ops, incoming_op_index, iter_ops, no_stra_op_list);
      }
    }

    if (s.size() == 0) {
      no_stra_op_list->push_back(iter_ops);
    } else {
      stra = GenerateStrategiesFromStrategy(ops, iter_ops, s);
    }

    StrategyPtr sp = std::make_shared<Strategy>(0, stra);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }
}

std::vector<int32_t> ModifyStrategyIfSqueezeOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                     const size_t iter_ops, std::vector<int32_t> s) {
  std::vector<int32_t> s_Squeeze;
  auto axis_list = GetAxisList(ops, iter_ops);
  size_t s_index = 0;
  size_t axis_list_index = 0;
  for (size_t i = 0; i < (size_t)(s.size() + axis_list.size()); i++) {
    if ((i) == (size_t)axis_list[axis_list_index]) {
      s_Squeeze.push_back(1);
      axis_list_index++;
    } else {
      s_Squeeze.push_back(s[s_index]);
      s_index++;
    }
  }
  return s_Squeeze;
}

std::vector<int32_t> ModifyStrategyIfReduceOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                    const size_t iter_ops, std::vector<int32_t> s) {
  std::vector<int32_t> dim_list = GetDimList(ops, iter_ops);
  if (dim_list.size() == 0) {
    return s;
  }
  std::vector<int32_t> s_Reduce;
  size_t s_index = 0;
  size_t dim_list_index = 0;
  for (size_t i = 0; i < (size_t)(s.size() + dim_list.size()); i++) {
    if (i == (size_t)dim_list[dim_list_index]) {
      s_Reduce.push_back(1);
      dim_list_index++;
    } else {
      s_Reduce.push_back(s[s_index]);
      s_index++;
    }
  }
  return s_Reduce;
}

std::vector<int32_t> CopyOutgoingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                       const std::vector<std::vector<std::string>> &input_tensor_names,
                                                       const size_t iter_ops) {
  std::vector<int32_t> s;
  bool found = false;
  for (size_t i = 0; i < (size_t)input_tensor_names.size(); i++) {
    for (size_t j = 1; j < (size_t)input_tensor_names[i].size(); j++) {
      if (input_tensor_names[i][j] == input_tensor_names[iter_ops][0]) {
        for (size_t k = 0; k < ops[i]->selected_strategy()->GetInputDim()[j - 1].size(); ++k) {
          s.push_back(ops[i]->selected_strategy()->GetInputDim()[j - 1][k]);
        }
        found = true;
        break;
      }
    }
    if (found) break;
  }
  return s;
}

void GenerateEliminatedOperatorStrategyBackward(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const std::vector<std::vector<std::string>> &input_tensor_names,
                                                const std::shared_ptr<std::vector<size_t>> no_stra_op_list) {
  MS_EXCEPTION_IF_NULL(no_stra_op_list);
  for (int iter_list = no_stra_op_list->size() - 1; iter_list >= 0; iter_list--) {
    auto iter_ops = no_stra_op_list->at(iter_list);
    std::vector<std::vector<int32_t>> stra;
    std::vector<int32_t> s = CopyOutgoingOperatorInputStrategy(ops, input_tensor_names, iter_ops);
    if (s.size() == 0) {
      for (size_t i = 0; i < ops[iter_ops]->inputs_tensor_info()[0].shape().size(); i++) {
        s.push_back(1);
      }
    }
    if (ops[iter_ops]->type() == SQUEEZE) {
      s = ModifyStrategyIfSqueezeOutgoing(ops, iter_ops, s);
    }
    if (ops[iter_ops]->type() == REDUCE_SUM || ops[iter_ops]->type() == REDUCE_MAX ||
        ops[iter_ops]->type() == REDUCE_MIN || ops[iter_ops]->type() == REDUCE_MEAN) {
      s = ModifyStrategyIfReduceOutgoing(ops, iter_ops, s);
    }
    stra = GenerateStrategiesFromStrategy(ops, iter_ops, s);
    StrategyPtr sp = std::make_shared<Strategy>(0, stra);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }
}

}  // namespace parallel
}  // namespace mindspore
