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
  for (size_t i = 0; i < eli_list->size(); i++) {
    no_stra_op_list->push_back(eli_list->at(i)[0]);
  }
  GenerateEliminatedOperatorStrategyForward(graph, ops, input_tensor_names, index_list, no_stra_op_list);
  GenerateEliminatedOperatorStrategyBackward(ops, input_tensor_names, no_stra_op_list);
  GenerateRemainingOperatorStrategy(graph, ops, input_tensor_names, index_list, no_stra_op_list);
}

std::vector<std::vector<int32_t>> PrepareMatMul(const std::shared_ptr<Graph> &graph,
                                                const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const size_t iter_graph, const size_t iter_ops) {
  std::vector<std::vector<int32_t>> strategies;
  auto attrs = ops[iter_ops]->attrs();
  bool transpose_a = attrs[TRANSPOSE_A]->cast<BoolImmPtr>()->value();
  bool transpose_b = attrs[TRANSPOSE_B]->cast<BoolImmPtr>()->value();

  // HCCL does not support multi-dimension partition, and the hardware does not support excessive
  // number of EVENT, so we temporarily disable matmul's multi-dimension partition function.
  auto max_cut = 1.0 / g_device_manager->DeviceNum();
  if (graph->nodes[iter_graph].apply.arguments[0].tensor_str.str_h != max_cut &&
      graph->nodes[iter_graph].apply.arguments[1].tensor_str.str_w != max_cut) {
    graph->nodes[iter_graph].apply.arguments[0].tensor_str.str_h = 1.0;
    graph->nodes[iter_graph].apply.arguments[0].tensor_str.str_w = 1.0;
    graph->nodes[iter_graph].apply.arguments[1].tensor_str.str_h = 1.0;
    graph->nodes[iter_graph].apply.arguments[1].tensor_str.str_w = 1.0;
    graph->nodes[iter_graph].tensor_parm.tensor_str.str_h = 1.0;
    graph->nodes[iter_graph].tensor_parm.tensor_str.str_w = 1.0;

    auto shape_1 = ops[iter_ops]->inputs_tensor_info()[0].shape()[0];
    if (transpose_a) {
      shape_1 = ops[iter_ops]->inputs_tensor_info()[0].shape()[1];
    }
    auto shape_4 = ops[iter_ops]->inputs_tensor_info()[1].shape()[1];
    if (transpose_b) {
      shape_4 = ops[iter_ops]->inputs_tensor_info()[1].shape()[0];
    }

    bool already_cut = false;
    if (shape_1 >= shape_4) {
      if (shape_1 % g_device_manager->DeviceNum() == 0) {
        graph->nodes[iter_graph].apply.arguments[0].tensor_str.str_h = max_cut;
        graph->nodes[iter_graph].tensor_parm.tensor_str.str_h = max_cut;
        already_cut = true;
      }
      if (!already_cut && shape_4 % g_device_manager->DeviceNum() == 0) {
        graph->nodes[iter_graph].apply.arguments[1].tensor_str.str_w = max_cut;
        graph->nodes[iter_graph].tensor_parm.tensor_str.str_w = max_cut;
        already_cut = true;
      }
    } else {
      if (shape_4 % g_device_manager->DeviceNum() == 0) {
        graph->nodes[iter_graph].apply.arguments[1].tensor_str.str_w = max_cut;
        graph->nodes[iter_graph].tensor_parm.tensor_str.str_w = max_cut;
        already_cut = true;
      }
      if (!already_cut && shape_1 % g_device_manager->DeviceNum() == 0) {
        graph->nodes[iter_graph].apply.arguments[0].tensor_str.str_h = max_cut;
        graph->nodes[iter_graph].tensor_parm.tensor_str.str_h = max_cut;
        already_cut = true;
      }
    }

    if (!already_cut) {
      MS_LOG(EXCEPTION) << "Failure: MatMul's shape is invalid.";
    }
  }

  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    std::vector<int32_t> s;
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

std::vector<std::vector<int32_t>> PreparePReLU(const std::shared_ptr<Graph> &graph,
                                               const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                               const size_t iter_graph, const size_t iter_ops) {
  std::vector<std::vector<int32_t>> strategies = MakeDataParallelStrategy(graph, ops, iter_graph, iter_ops);
  strategies[1][0] = 1;
  return strategies;
}

std::vector<std::vector<int32_t>> PrepareBatchNorm(const std::shared_ptr<Graph> &graph,
                                                   const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                   const size_t iter_graph, const size_t iter_ops) {
  std::vector<std::vector<int32_t>> strategies = MakeDataParallelStrategy(graph, ops, iter_graph, iter_ops);
  for (size_t i = 1; i < strategies.size(); i++) {
    strategies[i][0] = strategies[0][1];
  }
  strategies[1][0] = 1;
  return strategies;
}

std::vector<std::vector<int32_t>> PrepareSoftmaxWithLogits(const std::shared_ptr<Graph> &graph,
                                                           const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                           const size_t iter_graph, const size_t iter_ops) {
  std::vector<std::vector<int32_t>> strategies = MakeDataParallelStrategy(graph, ops, iter_graph, iter_ops);
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_w = graph->nodes[iter_graph].tensor_parm.tensor_str.str_h;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_h = graph->nodes[iter_graph].tensor_parm.tensor_str.str_c;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_c = graph->nodes[iter_graph].tensor_parm.tensor_str.str_n;
  return strategies;
}

std::vector<std::vector<int32_t>> PrepareBiasAdd(const std::shared_ptr<std::vector<int32_t>> &s) {
  std::vector<std::vector<int32_t>> strategies;
  strategies.push_back(*s);
  std::vector<int32_t> s_biasadd;
  s_biasadd.push_back(s->at(1));
  strategies.push_back(s_biasadd);
  return strategies;
}

std::vector<std::vector<int32_t>> PrepareOneHot(const std::shared_ptr<std::vector<int32_t>> &s) {
  std::vector<std::vector<int32_t>> strategies;
  std::vector<int32_t> s_empty = {};
  strategies.push_back(*s);
  strategies.push_back(s_empty);
  strategies.push_back(s_empty);
  return strategies;
}

std::vector<std::vector<int32_t>> PrepareGatherV2(const std::shared_ptr<std::vector<int32_t>> &s) {
  std::vector<std::vector<int32_t>> strategies;
  strategies.push_back(*s);
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

std::vector<std::vector<int32_t>> MakeDataParallelStrategy(const std::shared_ptr<Graph> &graph,
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
  size_t max_device_num = g_device_manager->DeviceNum();
  size_t target_tensor_batch = ops[iter_ops]->outputs_tensor_info()[0].shape()[0];
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    if (iter_op_inputs >= origin_strategy->GetInputDim().size()) {
      MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
    }

    std::vector<int32_t> s;
    size_t input_size = origin_strategy->GetInputDim()[iter_op_inputs].size();
    for (size_t dim = 0; dim < input_size; dim++) {
      if (input_size == 1 || input_size == 2 || input_size == 4) {
        if (dim == 0) {
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

  graph->nodes[iter_graph].tensor_parm.tensor_str.str_n = 1.0;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_c = 1.0;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_h = 1.0;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_w = 1.0;
  if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 1) {
    graph->nodes[iter_graph].tensor_parm.tensor_str.str_w = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 2) {
    graph->nodes[iter_graph].tensor_parm.tensor_str.str_h = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 4) {
    graph->nodes[iter_graph].tensor_parm.tensor_str.str_n = 1.0 / std::min(max_device_num, target_tensor_batch);
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
  MS_EXCEPTION_IF_NULL(ops[iter_ops]);

  auto type = ops[iter_ops]->type();
  auto idx = DictOpType.find(type);
  if (idx == DictOpType.end()) {
    return MakeDataParallelStrategy(graph, ops, iter_graph, iter_ops);
  }

  if (type == MATMUL) {
    return PrepareMatMul(graph, ops, iter_graph, iter_ops);
  } else if (type == PRELU) {
    return PreparePReLU(graph, ops, iter_graph, iter_ops);
  } else if (type == BATCH_NORM) {
    return PrepareBatchNorm(graph, ops, iter_graph, iter_ops);
  } else if (type == SOFTMAX_CROSS_ENTROPY_WITH_LOGITS) {
    return PrepareSoftmaxWithLogits(graph, ops, iter_graph, iter_ops);
  } else if (type == SOFTMAX || type == LOG_SOFTMAX || type == SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS) {
    return MakeDataParallelStrategy(graph, ops, iter_graph, iter_ops);
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
    if (iter_graph != SIZE_MAX) {
      strategies = PrepareStrategy(graph, ops, iter_graph, iter_ops);
    }
    StrategyPtr sp = std::make_shared<Strategy>(0, strategies);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }
}

size_t FindIndexOfOperatorIncoming(const std::vector<std::vector<std::string>> &input_tensor_names,
                                   const size_t iter_ops) {
  size_t incoming_op_index = SIZE_MAX;
  for (size_t i = 1; i < input_tensor_names[iter_ops].size(); i++) {
    for (size_t j = 0; j < input_tensor_names.size(); j++) {
      if (input_tensor_names[iter_ops][i] == input_tensor_names[j][0]) {
        incoming_op_index = j;
        break;
      }
    }
    if (incoming_op_index != SIZE_MAX) {
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
                                                          const size_t incoming_op_index) {
  std::vector<int32_t> s;
  if (ops[incoming_op_index]->type() == RESHAPE || ops[incoming_op_index]->type() == GATHERV2) {
    return s;
  }
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
                                                     const size_t incoming_op_index, std::vector<int32_t> s) {
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
                                                    const size_t incoming_op_index, std::vector<int32_t> s) {
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
                                                       const size_t iter_ops, const size_t incoming_op_index) {
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
                                                                 const size_t iter_ops,
                                                                 std::vector<int32_t> basic_stra) {
  std::vector<int32_t> s_empty = {};
  std::vector<std::vector<int32_t>> stra;
  MS_EXCEPTION_IF_NULL(ops[iter_ops]);

  if (basic_stra.size() == 0) {
    for (size_t iter_op_inputs = 0; iter_op_inputs < (size_t)ops[iter_ops]->inputs_tensor_info().size();
         iter_op_inputs++) {
      stra.push_back(basic_stra);
    }
    return stra;
  }

  auto s_ptr = std::make_shared<std::vector<int32_t>>(basic_stra);
  if (ops[iter_ops]->type() == BIAS_ADD) {
    return PrepareBiasAdd(s_ptr);
  }
  if (ops[iter_ops]->type() == ONEHOT) {
    return PrepareOneHot(s_ptr);
  }
  if (ops[iter_ops]->type() == GATHERV2) {
    return PrepareGatherV2(s_ptr);
  }

  for (size_t iter_op_inputs = 0; iter_op_inputs < (size_t)ops[iter_ops]->inputs_tensor_info().size();
       iter_op_inputs++) {
    if (ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape().size() == 0) {
      stra.push_back(s_empty);
      continue;
    }

    std::vector<int32_t> tmp_stra = basic_stra;
    bool modified = false;
    for (size_t j = 0; j < (size_t)ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape().size(); j++) {
      if (ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape()[j] == 1) {
        tmp_stra[j] = 1;
        modified = true;
      }
    }
    if (modified) {
      stra.push_back(tmp_stra);
    } else {
      stra.push_back(basic_stra);
    }
  }
  return stra;
}

void GenerateEliminatedOperatorStrategyForward(const std::shared_ptr<Graph> graph,
                                               const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                               const std::vector<std::vector<std::string>> &input_tensor_names,
                                               const std::shared_ptr<std::vector<size_t>> index_list,
                                               const std::shared_ptr<std::vector<size_t>> no_stra_op_list) {
  if (no_stra_op_list->size() == 0) {
    return;
  }
  std::vector<size_t> no_stra_op_list_bis;

  for (size_t iter_list = no_stra_op_list->size(); iter_list > 0; iter_list--) {
    size_t iter_ops = no_stra_op_list->at(iter_list - 1);
    std::vector<std::vector<int32_t>> stra;
    std::vector<int32_t> s;
    size_t incoming_op_index = FindIndexOfOperatorIncoming(input_tensor_names, iter_ops);
    if (incoming_op_index != SIZE_MAX && ops[iter_ops]->type() != ONEHOT) {
      auto iter_graph = index_list->at(incoming_op_index);
      if (iter_graph != SIZE_MAX) {
        s = CopyIncomingOperatorOutputStrategy(graph, ops, iter_ops, iter_graph);
      } else {
        s = CopyIncomingOperatorInputStrategy(ops, iter_ops, incoming_op_index);
      }
    }

    if (s.size() == 0) {
      no_stra_op_list_bis.push_back(iter_ops);
    } else {
      stra = GenerateStrategiesFromStrategy(ops, iter_ops, s);
    }

    StrategyPtr sp = std::make_shared<Strategy>(0, stra);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }

  no_stra_op_list->clear();
  for (size_t i = 0; i < no_stra_op_list_bis.size(); i++) {
    no_stra_op_list->push_back(no_stra_op_list_bis[i]);
  }
}

std::vector<int32_t> ModifyStrategyIfSqueezeOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                     const size_t iter_ops, std::vector<int32_t> s) {
  std::vector<int32_t> s_Squeeze;
  auto axis_list = GetAxisList(ops, iter_ops);
  size_t s_index = 0;
  size_t axis_list_index = 0;
  for (size_t i = 0; i < (size_t)(s.size() + axis_list.size()); i++) {
    if (i == (size_t)axis_list[axis_list_index]) {
      s_Squeeze.push_back(1);
      axis_list_index++;
    } else {
      s_Squeeze.push_back(s[s_index]);
      s_index++;
    }
  }

  size_t cut = 1;
  for (size_t i = 0; i < s_Squeeze.size(); i++) {
    cut *= s_Squeeze[i];
  }
  if (cut != g_device_manager->DeviceNum()) {
    s_Squeeze.clear();
  }

  return s_Squeeze;
}

std::vector<int32_t> CopyOutgoingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                       const std::vector<std::vector<std::string>> &input_tensor_names,
                                                       const size_t iter_ops) {
  std::vector<int32_t> s;
  if (ops[iter_ops]->type() == REDUCE_MAX || ops[iter_ops]->type() == REDUCE_MIN ||
      ops[iter_ops]->type() == REDUCE_SUM || ops[iter_ops]->type() == REDUCE_MEAN || ops[iter_ops]->type() == RESHAPE ||
      ops[iter_ops]->type() == GATHERV2) {
    return s;
  }

  bool found = false;
  size_t outgoing_op_index = SIZE_MAX;
  size_t iter_op_inputs = SIZE_MAX;
  for (size_t i = 0; i < input_tensor_names.size(); i++) {
    for (size_t j = 1; j < input_tensor_names[i].size(); j++) {
      if (input_tensor_names[i][j] == input_tensor_names[iter_ops][0] &&
          ops[i]->selected_strategy()->GetInputNumber() != 0) {
        outgoing_op_index = i;
        iter_op_inputs = j - 1;
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }

  if (outgoing_op_index != SIZE_MAX && iter_op_inputs != SIZE_MAX) {
    for (size_t k = 0; k < ops[outgoing_op_index]->selected_strategy()->GetInputDim()[iter_op_inputs].size(); ++k) {
      s.push_back(ops[outgoing_op_index]->selected_strategy()->GetInputDim()[iter_op_inputs][k]);
    }
  }
  return s;
}

void GenerateEliminatedOperatorStrategyBackward(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const std::vector<std::vector<std::string>> &input_tensor_names,
                                                const std::shared_ptr<std::vector<size_t>> no_stra_op_list) {
  if (no_stra_op_list->size() == 0) {
    return;
  }
  std::vector<size_t> no_stra_op_list_bis;

  for (size_t iter_list = no_stra_op_list->size(); iter_list > 0; iter_list--) {
    auto iter_ops = no_stra_op_list->at(iter_list - 1);
    std::vector<std::vector<int32_t>> stra;
    std::vector<int32_t> s = CopyOutgoingOperatorInputStrategy(ops, input_tensor_names, iter_ops);

    if (s.size() != 0 && ops[iter_ops]->type() == SQUEEZE) {
      s = ModifyStrategyIfSqueezeOutgoing(ops, iter_ops, s);
    }
    if (s.size() != 0) {
      stra = GenerateStrategiesFromStrategy(ops, iter_ops, s);
    } else {
      no_stra_op_list_bis.push_back(iter_ops);
    }

    StrategyPtr sp = std::make_shared<Strategy>(0, stra);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }

  no_stra_op_list->clear();
  for (size_t i = 0; i < no_stra_op_list_bis.size(); i++) {
    no_stra_op_list->push_back(no_stra_op_list_bis[i]);
  }
}

void GenerateRemainingOperatorStrategy(const std::shared_ptr<Graph> graph,
                                       const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                       const std::vector<std::vector<std::string>> &input_tensor_names,
                                       const std::shared_ptr<std::vector<size_t>> index_list,
                                       const std::shared_ptr<std::vector<size_t>> no_stra_op_list) {
  if (no_stra_op_list->size() == 0) {
    return;
  }

  size_t no_stra_op_list_size;
  do {
    no_stra_op_list_size = no_stra_op_list->size();
    GenerateEliminatedOperatorStrategyForward(graph, ops, input_tensor_names, index_list, no_stra_op_list);
    GenerateEliminatedOperatorStrategyBackward(ops, input_tensor_names, no_stra_op_list);
  } while (no_stra_op_list_size > no_stra_op_list->size());

  for (size_t iter_list = 0; iter_list < no_stra_op_list->size(); iter_list++) {
    auto iter_ops = no_stra_op_list->at(iter_list);
    std::vector<std::vector<int32_t>> stra;
    std::vector<int32_t> s;

    size_t max_dim_num = 0;
    for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
      if (ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape().size() > max_dim_num) {
        max_dim_num = ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape().size();
      }
    }
    for (size_t i = 0; i < max_dim_num; i++) {
      s.push_back(1);
    }

    stra = GenerateStrategiesFromStrategy(ops, iter_ops, s);
    StrategyPtr sp = std::make_shared<Strategy>(0, stra);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }
}
}  // namespace parallel
}  // namespace mindspore
