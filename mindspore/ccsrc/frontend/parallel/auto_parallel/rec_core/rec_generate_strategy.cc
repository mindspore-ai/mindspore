/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/auto_parallel/rec_core/rec_generate_strategy.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_parse_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_partition.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
void GenerateStrategy(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                      const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                      const std::vector<std::vector<std::string>> &input_tensor_names,
                      const std::shared_ptr<std::vector<size_t>> &index_list, bool is_training,
                      const std::vector<std::vector<size_t>> &shared_tensors_ops, const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(eli_list);
  MS_EXCEPTION_IF_NULL(index_list);
  GeneratePartitionedOperatorStrategy(graph, ops, index_list);

  std::shared_ptr<std::vector<size_t>> no_stra_op_list = std::make_shared<std::vector<size_t>>();
  for (size_t i = 0; i < eli_list->size(); i++) {
    no_stra_op_list->push_back(eli_list->at(i)[0]);
  }
  GenerateEliminatedOperatorStrategyForward(graph, ops, input_tensor_names, index_list, no_stra_op_list);
  GenerateEliminatedOperatorStrategyBackward(ops, input_tensor_names, no_stra_op_list);
  GenerateRemainingOperatorStrategy(graph, ops, input_tensor_names, index_list, no_stra_op_list);
  ModifyParamSharingOpsStrategy(ops, shared_tensors_ops);

  for (auto &op : ops) {
    // Set user-defined strategy
    auto attrs = op->attrs();
    if (StrategyFound(attrs)) {
      StrategyPtr user_defined_stra = parallel::ExtractStrategy(attrs[IN_STRATEGY]);
      op->SetSelectedStrategyAndCost(user_defined_stra, op->selected_cost());
    }
    // Set back to raw strategy for special node in predict/eval
    if (!is_training) {
      if ((op->is_last_node()) || (op->type() == VIRTUAL_DATA_SET)) {
        SetBackToRawStrategy(op);
      }
    }
  }
}

Dimensions PrepareMatMulStrategy(const std::shared_ptr<Graph> &graph, const size_t iter_graph, bool transpose_a,
                                 bool transpose_b, size_t iter_op_inputs) {
  Dimensions s;
  if (transpose_a && (iter_op_inputs == 0)) {
    s.push_back(static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    s.push_back(static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
  } else if (transpose_b && (iter_op_inputs == 1)) {
    s.push_back(static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    s.push_back(static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
  } else {
    s.push_back(static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
    s.push_back(static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
  }
  return s;
}

Strategies PrepareMatMul(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                         const size_t iter_graph, const size_t iter_ops) {
  Strategies strategies;
  auto attrs = ops[iter_ops]->attrs();
  bool transpose_a = attrs[TRANSPOSE_A]->cast<BoolImmPtr>()->value();
  bool transpose_b = attrs[TRANSPOSE_B]->cast<BoolImmPtr>()->value();

  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    Dimensions s = PrepareMatMulStrategy(graph, iter_graph, transpose_a, transpose_b, iter_op_inputs);
    strategies.push_back(s);
  }
  return strategies;
}

Strategies PrepareBiasAdd(const std::shared_ptr<Dimensions> &s) {
  Strategies strategies;
  strategies.push_back(*s);
  Dimensions s_biasadd;
  s_biasadd.push_back(s->at(1));
  strategies.push_back(s_biasadd);
  return strategies;
}

Strategies PrepareStridedSlice(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                               Dimensions basic_stra) {
  Strategies stra;

  auto begin = GetValue<std::vector<int64_t>>(ops[iter_ops]->input_value().at(1));
  auto end = GetValue<std::vector<int64_t>>(ops[iter_ops]->input_value().at(2));
  auto strides = GetValue<std::vector<int64_t>>(ops[iter_ops]->input_value().at(3));

  for (size_t i = 0; i < strides.size(); ++i) {
    if ((strides[i] != 1) && (basic_stra[i] > 1)) {
      basic_stra[i] = 1;
    }
  }

  for (size_t i = 0; i < begin.size(); ++i) {
    bool no_fully_fetch = ((begin[i] != 0) || (end[i] < ops[iter_ops]->inputs_tensor_info()[0].shape()[i]));
    if (no_fully_fetch && (basic_stra[i] != 1)) {
      basic_stra[i] = 1;
    }
  }

  stra.push_back(basic_stra);
  return stra;
}

Strategies PrepareSoftMax(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                          const Dimensions &basic_stra) {
  Strategies strategies;
  strategies.push_back(basic_stra);
  std::vector<int64_t> axis_list;
  string axis_name = AXIS;

  auto iter = ops[iter_ops]->attrs().find(axis_name);
  if (iter != ops[iter_ops]->attrs().end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int64Imm>()) {
      axis_list.push_back(iter->second->cast<Int64ImmPtr>()->value());
    } else if (iter->second->isa<ValueTuple>()) {
      ValueTuplePtr value_tuple = iter->second->cast<ValueTuplePtr>();
      if (value_tuple == nullptr) {
        MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": The value_tuple is nullptr.";
      }

      std::vector<ValuePtr> value_vector = value_tuple->value();
      (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(axis_list),
                           [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
    } else {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": The value of axis is not int64_t or tuple int64_t.";
    }
  } else {
    axis_list.push_back(-1);
  }

  for (auto &axis : axis_list) {
    if (axis < 0) {
      int64_t input_dim = SizeToLong(ops[iter_ops]->inputs_tensor_info()[0].shape().size());
      axis = input_dim + axis;
    }
    if (axis >= SizeToLong(strategies[0].size()) || axis < 0) {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": axis value is out of range.";
    }
    if (strategies[0][LongToSize(axis)] != 1) {
      strategies[0][LongToSize(axis)] = 1;
      MS_LOG(INFO) << ops[iter_ops]->name() << ": adjust strategy to 1 on axis " << axis;
    }
  }

  return strategies;
}

Strategies PrepareOneHot(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops, Dimensions s) {
  Strategies strategies;

  // The Dimension size of the first input tensor of OneHot should be 2 even its Shape size is 1. Using the division of
  // the number of devices and the partition parts of the first dimension.

  size_t s_second = 1;

  if (s[0] != 0) {
    s_second = g_device_manager->stage_device_num() / LongToSize(s[0]);
  }

  if (s.size() == 1) {
    s.push_back(s_second);
  }

  // Partition number should not exceed the number of devices
  for (size_t i = 0; i < ops[iter_ops]->outputs_tensor_info()[0].shape().size(); i++) {
    if (s[i] > ops[iter_ops]->outputs_tensor_info()[0].shape()[i]) {
      s[i] = 1;
    }
  }

  strategies.push_back(s);

  // Push two empty Dimensions for the other two input tensors.
  Dimensions s_empty = {};
  strategies.push_back(s_empty);
  strategies.push_back(s_empty);

  return strategies;
}

Strategies PrepareGatherV2(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops, Dimensions s) {
  Strategies strategies;

  auto output_shape = ops[iter_ops]->outputs_tensor_info()[0].shape();
  Dimensions index(output_shape.size() - 1, 0);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = SizeToLong(i);
  }
  std::sort(index.begin(), index.end(), [&output_shape](const int64_t &a, const int64_t &b) {
    return (output_shape[LongToSize(a + 1)] > output_shape[LongToSize(b + 1)]);
  });
  (void)std::transform(std::begin(index), std::end(index), std::begin(index), [](int64_t x) { return x + 1; });
  (void)index.insert(index.cbegin(), 0);

  Dimensions strategie(output_shape.size(), 1);
  size_t num_device = g_device_manager->stage_device_num();
  size_t cut = 1;
  for (size_t i = 0; i < index.size(); i++) {
    size_t index_i = LongToSize(index[i]);
    while (output_shape[index_i] % 2 == 0 && output_shape[index_i] > 0 && cut < num_device) {
      output_shape[index_i] /= 2;
      cut *= 2;
      strategie[index_i] *= 2;  // We apply 2-parts partitioning for Gather.
    }
    if (cut == num_device) {
      break;
    }
  }

  auto axis_input = GetValue<int64_t>(ops[iter_ops]->input_value().at(2));
  if (axis_input < 0) {
    axis_input += SizeToLong(ops[iter_ops]->inputs_tensor_info()[0].shape().size());
  }
  int64_t axis = axis_input;
  if (axis >= SizeToLong(s.size())) {
    MS_LOG(EXCEPTION) << "Failure: GatherV2' axis out of range.";
  }
  if (axis == 0) {
    s.clear();
    s.push_back(1);
    for (size_t i = 1; i < ops[iter_ops]->inputs_tensor_info()[0].shape().size(); i++) {
      s.push_back(strategie[ops[iter_ops]->inputs_tensor_info()[1].shape().size() - 1 + i]);
    }
    strategies.push_back(s);
    s.clear();
    for (size_t i = 0; i < ops[iter_ops]->inputs_tensor_info()[1].shape().size(); i++) {
      s.push_back(strategie[i]);
    }
    strategies.push_back(s);
  } else if (axis == 1) {
    s.clear();
    s.push_back(strategie[0]);
    s.push_back(1);
    strategies.push_back(s);
    s.clear();
    for (size_t i = 0; i < ops[iter_ops]->inputs_tensor_info()[1].shape().size(); i++) {
      s.push_back(strategie[ops[iter_ops]->inputs_tensor_info()[0].shape().size() - 1 + i]);
    }
    strategies.push_back(s);
  } else {
    MS_LOG(EXCEPTION) << "Failure: GatherV2's axis is neither 0 nor 1.";
  }

  return strategies;
}

Dimensions PrepareGatherV2OutputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                         const size_t incoming_op_index) {
  auto output_shape = ops[incoming_op_index]->outputs_tensor_info()[0].shape();
  Dimensions index(output_shape.size() - 1, 0);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = SizeToLong(i);
  }
  std::sort(index.begin(), index.end(),
            [&output_shape](const size_t &a, const size_t &b) { return (output_shape[a + 1] > output_shape[b + 1]); });
  (void)std::transform(std::begin(index), std::end(index), std::begin(index), [](int64_t x) { return x + 1; });
  (void)index.insert(index.cbegin(), 0);

  Dimensions strategie(output_shape.size(), 1);
  size_t num_device = g_device_manager->stage_device_num();
  size_t cut = 1;
  for (size_t i = 0; i < index.size(); i++) {
    size_t index_i = LongToSize(index[i]);
    while (output_shape[index_i] % 2 == 0 && output_shape[index_i] > 0 && cut < num_device) {
      output_shape[index_i] /= 2;
      cut *= 2;
      strategie[index_i] *= 2;  // We apply 2-parts partitioning for Gather.
    }
    if (cut == num_device) {
      break;
    }
  }

  return strategie;
}

Strategies PrepareL2Normalize(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                              Dimensions s) {
  int64_t axis = 0;
  auto iter = ops[iter_ops]->attrs().find(AXIS);
  if (iter != ops[iter_ops]->attrs().end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<ValueSequence>()) {
      axis = GetValue<std::vector<int64_t>>(iter->second)[0];
    } else {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << " : The value of axis is not int64_t.";
    }
  }

  int64_t axis_index = axis;
  if (axis < 0) {
    size_t input_dim = ops[iter_ops]->inputs_tensor_info()[0].shape().size();
    axis_index = static_cast<int64_t>(input_dim) + axis;
  }

  s[LongToSize(axis_index)] = 1;

  Strategies strategies;
  strategies.push_back(s);
  return strategies;
}

Strategies PrepareAxisRelatedStrategy(const std::shared_ptr<Graph> &graph,
                                      const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_graph,
                                      const size_t iter_ops) {
  Strategies strategies = MakeRecSearchStrategy(graph, ops, iter_graph, iter_ops);
  if (strategies.size() < 1) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": get empty Strategy.";
  }

  std::vector<int64_t> axis_list;
  string axis_name = AXIS;
  int64_t default_axis = -1;
  if (ops[iter_ops]->type() == LAYER_NORM) {
    axis_name = "begin_norm_axis";
    default_axis = 1;
  }

  auto iter = ops[iter_ops]->attrs().find(axis_name);
  if (iter != ops[iter_ops]->attrs().end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int64Imm>()) {
      axis_list.push_back(iter->second->cast<Int64ImmPtr>()->value());
    } else if (iter->second->isa<ValueTuple>()) {
      ValueTuplePtr value_tuple = iter->second->cast<ValueTuplePtr>();
      if (value_tuple == nullptr) {
        MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": The value_tuple is nullptr.";
      }
      std::vector<ValuePtr> value_vector = value_tuple->value();
      (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(axis_list),
                           [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
    } else {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": The value of axis is not int64_t or tuple int64_t.";
    }
  } else {
    axis_list.push_back(default_axis);
  }

  for (auto &axis : axis_list) {
    if (axis < 0) {
      int64_t input_dim = SizeToLong(ops[iter_ops]->inputs_tensor_info()[0].shape().size());
      axis = input_dim + axis;
    }
    if (axis >= SizeToLong(strategies[0].size()) || axis < 0) {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": axis value is out of range.";
    }
    if (strategies[0][LongToSize(axis)] != 1) {
      strategies[0][LongToSize(axis)] = 1;
      MS_LOG(INFO) << ops[iter_ops]->name() << ": adjust strategy to 1 on axis " << axis;
    }
  }
  return strategies;
}

Strategies MakeRecSearchStrategy(const std::shared_ptr<Graph> &graph,
                                 const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_graph,
                                 const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }
  if (graph->nodes[iter_graph].apply.op_type == kRecUnsortedSegmentOp) {
    return MakeDataParallelStrategy(graph, ops, iter_graph, iter_ops);
  }

  StrategyPtr origin_strategy = ops[iter_ops]->strategy();
  Strategies strategies;
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    if (iter_op_inputs >= origin_strategy->GetInputDim().size()) {
      MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
    }

    size_t output_size = origin_strategy->GetInputDim()[iter_op_inputs].size();
    Dimensions s;
    if (output_size == 4) {
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_n));
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_c));
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (output_size == 3) {
      // Experimental support for 3D data.
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_c));
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (output_size == 2) {
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_h));
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (output_size == 1) {
      s.push_back(
        static_cast<int64_t>(1.0 / graph->nodes[iter_graph].apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (output_size == 0) {
      s = {};
    } else {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Tensor's output size is unexcepted.";
    }
    strategies.push_back(s);
  }
  return strategies;
}

Strategies MakeDataParallelStrategy(const std::shared_ptr<Graph> &graph,
                                    const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_graph,
                                    const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  StrategyPtr origin_strategy = ops[iter_ops]->strategy();
  Strategies strategies;
  size_t max_device_num = g_device_manager->stage_device_num();
  size_t target_tensor_batch = LongToUlong(ops[iter_ops]->inputs_tensor_info()[0].shape()[0]);
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    if (iter_op_inputs >= origin_strategy->GetInputDim().size()) {
      MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
    }

    Dimensions s;
    size_t input_size = origin_strategy->GetInputDim()[iter_op_inputs].size();
    for (size_t dim = 0; dim < input_size; dim++) {
      // Experimental support for 3D data (input_size == 3).
      if (input_size >= 1 && input_size <= 4) {
        if (dim == 0) {
          s.push_back(std::min(max_device_num, target_tensor_batch));
        } else {
          s.push_back(1);
        }
      } else if (input_size == 0) {
        s = {};
      } else {
        MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Tensor shape " << input_size << " is unexpected.";
      }
    }
    strategies.push_back(s);
  }
  // Set default strategy.
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_n = 1.0;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_c = 1.0;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_h = 1.0;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_w = 1.0;

  // Update data parallel strategy.
  if (ops[iter_ops]->outputs_tensor_info().size() == 0) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << " output tensor info is empty.";
  }
  if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 1) {
    graph->nodes[iter_graph].tensor_parm.tensor_str.str_w = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 2) {
    graph->nodes[iter_graph].tensor_parm.tensor_str.str_h = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 3) {
    // Experimental support for 3D data.
    graph->nodes[iter_graph].tensor_parm.tensor_str.str_c = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else if (ops[iter_ops]->outputs_tensor_info()[0].shape().size() == 4) {  // Experimental support for 4D data.
    graph->nodes[iter_graph].tensor_parm.tensor_str.str_n = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else {
    MS_LOG(INFO) << ops[iter_ops]->name() << " output tensor shape is unexpected, using default value instead.";
  }

  return strategies;
}

Strategies MakeFullBatchStrategy(const std::shared_ptr<Graph> &graph,
                                 const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_graph,
                                 const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  StrategyPtr origin_strategy = ops[iter_ops]->strategy();
  Strategies strategies;
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    if (iter_op_inputs >= origin_strategy->GetInputDim().size()) {
      MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
    }
    Dimensions s;
    size_t input_size = origin_strategy->GetInputDim()[iter_op_inputs].size();
    for (size_t dim = 0; dim < input_size; dim++) {
      if (input_size >= 1 && input_size <= 4) {
        s.push_back(1);
      } else if (input_size == 0) {
        s = {};
      } else {
        MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Tensor shape " << input_size << " is unexpected.";
      }
    }
    strategies.push_back(s);
  }
  // Update the output strategy of Rec Graph
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_n = 1.0;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_c = 1.0;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_h = 1.0;
  graph->nodes[iter_graph].tensor_parm.tensor_str.str_w = 1.0;

  return strategies;
}

void SetBackToRawStrategy(const std::shared_ptr<OperatorInfo> &op) {
  StrategyPtr origin_strategy = op->strategy();
  Strategies strategies;

  for (size_t iter_strategy = 0; iter_strategy < origin_strategy->GetInputDim().size(); iter_strategy++) {
    Dimensions s;
    size_t strategy_size = origin_strategy->GetInputDim()[iter_strategy].size();
    for (size_t dim = 0; dim < strategy_size; dim++) {
      if (strategy_size >= 1 && strategy_size <= 4) {
        s.push_back(1);
      } else if (strategy_size == 0) {
        s = {};
      } else {
        MS_LOG(EXCEPTION) << op->name() << ": Strategy size " << strategy_size << " is unmatched.";
      }
    }
    strategies.push_back(s);
  }

  StrategyPtr sp = std::make_shared<Strategy>(0, strategies);
  op->SetSelectedStrategyAndCost(sp, op->selected_cost());
}

Strategies PrepareStrategy(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                           const size_t iter_graph, const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }
  MS_EXCEPTION_IF_NULL(ops[iter_ops]);

  auto type = ops[iter_ops]->type();
  if (type == MATMUL) {
    return PrepareMatMul(graph, ops, iter_graph, iter_ops);
  } else if (type == LAYER_NORM) {
    return PrepareAxisRelatedStrategy(graph, ops, iter_graph, iter_ops);
  } else if (type == SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS) {
    return MakeDataParallelStrategy(graph, ops, iter_graph, iter_ops);
  } else if (type == VIRTUAL_DATA_SET) {
    if (ParallelContext::GetInstance()->full_batch()) {
      return MakeFullBatchStrategy(graph, ops, iter_graph, iter_ops);
    } else {
      return MakeDataParallelStrategy(graph, ops, iter_graph, iter_ops);
    }
  } else {
    return MakeRecSearchStrategy(graph, ops, iter_graph, iter_ops);
  }
}

void GeneratePartitionedOperatorStrategy(const std::shared_ptr<Graph> &graph,
                                         const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                         const std::shared_ptr<std::vector<size_t>> &index_list) {
  for (size_t iter_ops = 0; iter_ops < index_list->size(); iter_ops++) {
    Strategies strategies;
    size_t iter_graph = index_list->at(iter_ops);
    if (iter_graph != SIZE_MAX && ops[iter_ops]->type() != GET_NEXT) {
      strategies = PrepareStrategy(graph, ops, iter_graph, iter_ops);
    }
    StrategyPtr sp = std::make_shared<Strategy>(0, strategies);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }
}

void ModifyParamSharingOpsStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                   const std::vector<std::vector<size_t>> &param_users_ops_index) {
  for (auto tensor : param_users_ops_index) {
    for (auto op_i : tensor) {
      if (ops[op_i]->type() == GATHERV2) {
        for (auto op_j : tensor) {
          if (op_i != op_j) {
            Dimensions str_j;
            if (ops[op_j]->type() == CAST) {
              str_j = ops[op_j]->selected_strategy()->GetInputDim()[0];
            } else if (ops[op_j]->type() == MATMUL) {
              str_j = ops[op_j]->selected_strategy()->GetInputDim()[1];
            } else {
              continue;
            }
            Strategies strategies;
            Dimensions str1, str2;
            str1 = str_j;
            size_t num_device_used = 1;
            for (size_t i = 0; i < str_j.size(); i++) {
              num_device_used *= LongToSize(str_j[i]);
            }
            str2.push_back(g_device_manager->stage_device_num() / num_device_used);
            str2.push_back(1);
            strategies.push_back(str1);
            strategies.push_back(str2);
            StrategyPtr sp = std::make_shared<Strategy>(0, strategies);
            MS_LOG(INFO) << "Changing strategy of " << ops[op_i]->name() << " with " << ops[op_j]->name();
            ops[op_i]->SetSelectedStrategyAndCost(sp, ops[op_i]->selected_cost());
          }
        }
      }
    }
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

float CheckVirtualDatasetStrategy(const std::shared_ptr<Graph> &graph, const size_t iter_graph) {
  // The values for str can only be 1.0, 0.5, 0.25, 0.125…
  // We want to find out the first str that is smaller than 1
  if (graph->nodes[iter_graph].tensor_parm.tensor_str.str_n < 0.9) {
    return graph->nodes[iter_graph].tensor_parm.tensor_str.str_n;
  }
  if (graph->nodes[iter_graph].tensor_parm.tensor_str.str_c < 0.9) {
    return graph->nodes[iter_graph].tensor_parm.tensor_str.str_c;
  }
  if (graph->nodes[iter_graph].tensor_parm.tensor_str.str_h < 0.9) {
    return graph->nodes[iter_graph].tensor_parm.tensor_str.str_h;
  }
  if (graph->nodes[iter_graph].tensor_parm.tensor_str.str_w < 0.9) {
    return graph->nodes[iter_graph].tensor_parm.tensor_str.str_w;
  }
  return 1.0;
}

Dimensions CopyVirtualDataset(const std::shared_ptr<Graph> &graph,
                              const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                              const size_t iter_graph, float epsilon = 0.00005f) {
  Dimensions s;
  auto input_stra_dim = ops[iter_ops]->inputs_tensor_info()[0].shape().size();
  auto virtual_dataset_str = CheckVirtualDatasetStrategy(graph, iter_graph);
  if (input_stra_dim == 0) {
    return s;
  } else {
    if (std::fabs(virtual_dataset_str) < epsilon) {
      s.push_back(1);
    } else {
      s.push_back(FloatToLong(1 / virtual_dataset_str));
    }
    for (size_t i = 1; i < input_stra_dim; i++) {
      s.push_back(1);
    }
  }
  return s;
}

Dimensions CopyIncomingOperatorOutputStrategy(const std::shared_ptr<Graph> &graph,
                                              const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                              const size_t iter_ops, const size_t iter_graph,
                                              const size_t incoming_op_index) {
  Dimensions s;

  if (ops[incoming_op_index]->type() == VIRTUAL_DATA_SET) {
    s = CopyVirtualDataset(graph, ops, iter_ops, iter_graph);
    return s;
  }

  for (auto input : ops[iter_ops]->inputs_tensor_info()) {
    auto input_stra_dim = input.shape().size();
    if (input_stra_dim == 0) {
      continue;
    }
    if (input_stra_dim == 1) {
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_w));
    } else if (input_stra_dim == 2) {
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_h));
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_w));
    } else if (input_stra_dim == 3) {
      // Experimental support for 3D data.
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_c));
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_h));
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_w));
    } else if (input_stra_dim == 4) {
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_n));
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_c));
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_h));
      s.push_back(FloatToLong(1 / graph->nodes[iter_graph].tensor_parm.tensor_str.str_w));
    } else {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Tensor's shape is unknown.";
    }
    break;
  }
  return s;
}

Dimensions PrepareReshapeOutputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                        const size_t incoming_op_index) {
  Dimensions s;
  auto output_shape = ops[incoming_op_index]->outputs_tensor_info()[0].shape();
  auto input_shape = ops[incoming_op_index]->inputs_tensor_info()[0].shape();
  auto strategy = ops[incoming_op_index]->selected_strategy();
  std::vector<int64_t> mapping;
  int64_t tmp_prod = 1;
  int64_t tmp_index = 0;

  // mapping is using to store the correspondence between the input dims and output dims.
  // mapping stores the output's corresponded input's position.
  // If the length of the output shape is bigger than that of input,
  // e.g. input_shape [2,4,2] output_shape [2,2,2,2], the mapping is [0,1,1,2,-1].
  // If the length of the output shape is smaller than that of input,
  // e.g. input_shape [2,2,2,2] output_shape [2,4,2], the mapping is [0,2,3,-1].
  if (output_shape.size() >= input_shape.size()) {
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (input_shape[i] < output_shape[LongToSize(tmp_index)]) {
        break;
      }
      for (size_t j = LongToSize(tmp_index); j < output_shape.size(); j++) {
        tmp_prod *= output_shape[j];
        tmp_index++;
        if (input_shape[i] == tmp_prod) {
          tmp_prod = 1;
          mapping.push_back(i);
          break;
        }
        mapping.push_back(i);
      }
    }
    mapping.push_back(-1);
    tmp_index = 0;
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (mapping[i] == tmp_index) {
        s.push_back(strategy->GetInputDim()[0][LongToSize(mapping[i])]);
        tmp_index++;
      } else {
        s.push_back(1);
      }
    }
    return s;
  }

  for (size_t i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] < input_shape[LongToSize(tmp_index)]) {
      break;
    }
    for (size_t j = LongToSize(tmp_index); j < input_shape.size(); j++) {
      tmp_prod *= input_shape[j];
      if (output_shape[i] == tmp_prod) {
        tmp_prod = 1;
        mapping.push_back(tmp_index);
        tmp_index++;
        break;
      }
      tmp_index++;
    }
  }
  mapping.push_back(-1);
  tmp_index = 0;
  tmp_prod = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    if (mapping[i] == -1) {
      mapping.push_back(-1);
      s.push_back(1);
    } else {
      for (size_t j = LongToSize(tmp_index); j < input_shape.size(); j++) {
        tmp_prod *= strategy->GetInputDim()[0][j];
        tmp_index++;
        if (mapping[i] == SizeToLong(j)) {
          s.push_back(tmp_prod);
          tmp_prod = 1;
          break;
        }
      }
    }
  }
  return s;
}

Dimensions PrepareTransposeOutputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                          const size_t incoming_op_index) {
  Dimensions s;
  auto permutation = GetValue<std::vector<int64_t>>(ops[incoming_op_index]->input_value().at(1));
  auto strategy = ops[incoming_op_index]->selected_strategy();
  // The strategies are assigned according to the order in permutation (user defined).
  for (size_t i = 0; i < permutation.size(); i++) {
    s.push_back(strategy->GetInputDim()[0][LongToSize(permutation[i])]);
  }
  return s;
}

Dimensions PrepareExpandDimsOutputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                           const size_t incoming_op_index) {
  Dimensions s;

  auto axis_input = GetValue<int64_t>(ops[incoming_op_index]->input_value().at(1));
  auto strategy = ops[incoming_op_index]->selected_strategy();
  bool already_expand = false;

  // The strategy of the expanded dimesion will be assigned 1, the others take the strategies of corresponding
  // dimensions.
  for (size_t i = 0; i < ops[incoming_op_index]->inputs_tensor_info()[0].shape().size() + 1; i++) {
    if (UlongToLong(i) == axis_input) {
      s.push_back(1);
      already_expand = true;
    } else if (UlongToLong(i) != axis_input && !already_expand) {
      s.push_back(strategy->GetInputDim()[0][i]);
    } else {
      if (i < 1) {
        MS_LOG(EXCEPTION) << "The index i -1 is less than 0. Please check the situation.";
      }
      s.push_back(strategy->GetInputDim()[0][i - 1]);
    }
  }

  return s;
}

Dimensions PrepareIncompingArithmeticOpeartorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                           const size_t incoming_op_index) {
  Dimensions s;
  size_t max = 0;
  for (size_t i = 1; i < ops[incoming_op_index]->inputs_tensor_info().size(); i++) {
    if (ops[incoming_op_index]->inputs_tensor_info()[i].shape().size() >
        ops[incoming_op_index]->inputs_tensor_info()[max].shape().size()) {
      max = i;
    }
  }

  for (size_t j = 0; j < ops[incoming_op_index]->inputs_tensor_info()[max].shape().size(); j++) {
    s.push_back(ops[incoming_op_index]->selected_strategy()->GetInputDim()[max][j]);
  }

  return s;
}

Dimensions PrepareIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const size_t incoming_op_index) {
  Dimensions s;

  if (ops[incoming_op_index]->type() == GATHERV2) {
    auto pos = ops[incoming_op_index]->name().find("Info");
    if (pos == std::string::npos) {
      return s;
    }
    auto name = ops[incoming_op_index]->name().substr(0, pos);
    if (name == "Gather") {
      return PrepareGatherV2OutputStrategy(ops, incoming_op_index);
    } else {
      MS_LOG(EXCEPTION) << "Failure: Unknown type of GatherV2." << std::endl;
    }
  }
  auto strategy = ops[incoming_op_index]->selected_strategy();
  if (strategy->GetInputNumber() == 0) {
    return s;
  }

  if (ops[incoming_op_index]->type() == MUL || ops[incoming_op_index]->type() == SUB ||
      ops[incoming_op_index]->type() == ADD || ops[incoming_op_index]->type() == BIAS_ADD) {
    s = PrepareIncompingArithmeticOpeartorInputStrategy(ops, incoming_op_index);
    return s;
  }

  if (ops[incoming_op_index]->type() == RESHAPE) {
    return PrepareReshapeOutputStrategy(ops, incoming_op_index);
  } else if (ops[incoming_op_index]->type() == TRANSPOSE) {
    return PrepareTransposeOutputStrategy(ops, incoming_op_index);
  } else if (ops[incoming_op_index]->type() == EXPAND_DIMS) {
    return PrepareExpandDimsOutputStrategy(ops, incoming_op_index);
  }
  for (size_t i = 0; i < ops[incoming_op_index]->inputs_tensor_info().size(); i++) {
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

Dimensions GetAxisList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const int64_t iter_ops) {
  Dimensions axis_list;
  auto axis_param = ops[LongToSize(iter_ops)]->attrs().find(AXIS)->second;
  std::vector<ValuePtr> elements;
  if (axis_param->isa<ValueTuple>()) {
    elements = axis_param->cast<ValueTuplePtr>()->value();
  } else if (axis_param->isa<ValueList>()) {
    elements = axis_param->cast<ValueListPtr>()->value();
  } else {
    MS_LOG(EXCEPTION) << "Failure: Axis type is invalid, neither tuple nor list." << std::endl;
  }

  for (auto &element : elements) {
    if (!element->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "Failure: Dimension indexes is not Int32." << std::endl;
    }
    auto axis = element->cast<Int64ImmPtr>()->value();
    axis_list.push_back(axis);
  }
  return axis_list;
}

Dimensions ModifyStrategyIfSqueezeIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                           const size_t incoming_op_index, Dimensions s) {
  Dimensions s_Squeeze;
  Dimensions stra_dim_list;
  for (size_t i = 0; i < s.size(); i++) {
    stra_dim_list.push_back(SizeToLong(i));
  }

  auto axis_list = GetAxisList(ops, SizeToLong(incoming_op_index));
  for (auto axis : axis_list) {
    auto it = find(stra_dim_list.begin(), stra_dim_list.end(), axis);
    if (it == stra_dim_list.end()) {
      MS_LOG(EXCEPTION) << "Failure: Can not find dimension indexes in Axis." << std::endl;
    }
    if (ops[incoming_op_index]->inputs_tensor_info()[0].shape()[LongToSize(axis)] != 1) {
      MS_LOG(EXCEPTION) << "Failure: Removed dimension's shape is not 1." << std::endl;
    }
    (void)stra_dim_list.erase(it);
  }

  for (size_t i = 0; i < stra_dim_list.size(); i++) {
    s_Squeeze.push_back(s[LongToSize(stra_dim_list[i])]);
  }
  return s_Squeeze;
}

bool GetKeepDims(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops) {
  bool keepdims = false;
  auto keep_dims_iter = ops[iter_ops]->attrs().find(KEEP_DIMS);
  if (keep_dims_iter == ops[iter_ops]->attrs().end()) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Don't have attr keep_dims.";
  }
  MS_EXCEPTION_IF_NULL(keep_dims_iter->second);
  if (!keep_dims_iter->second->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Keep_dims is not a bool.";
  }
  keepdims = keep_dims_iter->second->cast<BoolImmPtr>()->value();
  return keepdims;
}

Dimensions GetDimList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops) {
  Dimensions dim_list;
  bool keep_dims = GetKeepDims(ops, iter_ops);
  if (keep_dims != false) {
    return dim_list;
  }
  auto input_value = ops[iter_ops]->input_value();
  auto input_dim = ops[iter_ops]->inputs_tensor_info()[0].shape().size();
  if (input_value.back()->isa<ValueTuple>()) {
    auto attr_axis = GetValue<std::vector<int64_t>>(input_value.back());
    if (attr_axis.empty()) {
      for (size_t i = 0; i < input_dim; i++) {
        dim_list.push_back(SizeToLong(i));
      }
    } else {
      for (auto &axis : attr_axis) {
        axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
      }
    }
  } else if (input_value.back()->isa<Int64Imm>()) {
    int64_t axis = GetValue<int64_t>(input_value.back());
    axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
  } else {
    MS_LOG(EXCEPTION) << "Failure: Axis type is invalid." << std::endl;
  }
  return dim_list;
}

Dimensions ModifyStrategyIfReduceIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                          const size_t incoming_op_index, Dimensions s) {
  Dimensions s_Reduce;
  Dimensions axis_list;
  for (size_t i = 0; i < s.size(); i++) {
    axis_list.push_back(SizeToLong(i));
  }

  auto dim_list = GetDimList(ops, incoming_op_index);
  for (auto axis : dim_list) {
    auto it = find(axis_list.begin(), axis_list.end(), axis);
    if (it == axis_list.end()) {
      MS_LOG(EXCEPTION) << "Failure: Can not find dimension indexes in Axis." << std::endl;
    }
    (void)axis_list.erase(it);
  }

  for (size_t i = 0; i < axis_list.size(); i++) {
    s_Reduce.push_back(s[LongToSize(axis_list[i])]);
  }
  return s_Reduce;
}

Dimensions GetDimListFromAttrs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops) {
  Dimensions dim_list;
  auto iter = ops[iter_ops]->attrs().find(AXIS);
  if (iter == ops[iter_ops]->attrs().end()) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Don't have attr axis.";
  }
  auto input_dim = ops[iter_ops]->inputs_tensor_info()[0].shape().size();
  MS_EXCEPTION_IF_NULL(iter->second);
  if (iter->second->isa<ValueTuple>()) {
    auto attr_axis = GetValue<std::vector<int64_t>>(iter->second);
    if (attr_axis.empty()) {
      for (size_t i = 0; i < input_dim; ++i) {
        dim_list.push_back(SizeToLong(i));
      }
    } else {
      for (auto &axis : attr_axis) {
        axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
      }
    }
  } else if (iter->second->isa<Int64Imm>()) {
    int64_t axis = GetValue<int64_t>(iter->second);
    axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
  } else {
    MS_LOG(EXCEPTION) << "Axis type is invalid.";
  }
  return dim_list;
}

Dimensions ModifyStrategyIfArgIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                       const size_t incoming_op_index, Dimensions s) {
  bool keepdims = GetKeepDims(ops, incoming_op_index);
  if (keepdims) {
    return s;
  }

  Dimensions s_Arg;
  Dimensions axis_list;
  for (size_t i = 0; i < s.size(); i++) {
    axis_list.push_back(SizeToLong(i));
  }

  auto dim_list = GetDimListFromAttrs(ops, incoming_op_index);
  for (auto axis : dim_list) {
    auto it = find(axis_list.begin(), axis_list.end(), axis);
    if (it == axis_list.end()) {
      MS_LOG(EXCEPTION) << "Failure: Can not find dimension indexes in Axis." << std::endl;
    }
    (void)axis_list.erase(it);
  }

  for (size_t i = 0; i < axis_list.size(); i++) {
    s_Arg.push_back(s[LongToSize(axis_list[i])]);
  }
  return s_Arg;
}

Dimensions CopyIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                             const size_t iter_ops, const size_t incoming_op_index) {
  Dimensions s;
  if (ops[iter_ops]->type() == ONEHOT) {
    return s;
  }
  s = PrepareIncomingOperatorInputStrategy(ops, incoming_op_index);
  if (s.size() != 0) {
    if (ops[incoming_op_index]->type() == SQUEEZE) {
      s = ModifyStrategyIfSqueezeIncoming(ops, incoming_op_index, s);
    }
    if (ops[incoming_op_index]->type() == REDUCE_SUM || ops[incoming_op_index]->type() == REDUCE_MAX ||
        ops[incoming_op_index]->type() == REDUCE_MIN || ops[incoming_op_index]->type() == REDUCE_MEAN) {
      s = ModifyStrategyIfReduceIncoming(ops, incoming_op_index, s);
    }
    if (ops[incoming_op_index]->type() == ARGMAXWITHVALUE || ops[incoming_op_index]->type() == ARGMINWITHVALUE) {
      s = ModifyStrategyIfArgIncoming(ops, incoming_op_index, s);
    }
  }
  return s;
}

Strategies GenerateStrategiesFromStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                          Dimensions basic_stra) {
  Strategies stra;
  MS_EXCEPTION_IF_NULL(ops[iter_ops]);

  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  if (basic_stra.size() == 0) {
    for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
      stra.push_back(basic_stra);
    }
    return stra;
  }

  auto s_ptr = std::make_shared<Dimensions>(basic_stra);
  if (ops[iter_ops]->type() == BIAS_ADD) {
    return PrepareBiasAdd(s_ptr);
  }
  if (ops[iter_ops]->type() == STRIDED_SLICE) {
    return PrepareStridedSlice(ops, iter_ops, basic_stra);
  }
  if (ops[iter_ops]->type() == GATHERV2) {
    auto pos = ops[iter_ops]->name().find("Info");
    auto name = ops[iter_ops]->name().substr(0, pos);
    if (name == "Gather") {
      return PrepareGatherV2(ops, iter_ops, basic_stra);
    } else {
      MS_LOG(EXCEPTION) << "Failure: Unknown type of GatherV2." << std::endl;
    }
  }
  if (ops[iter_ops]->type() == ONEHOT) {
    return PrepareOneHot(ops, iter_ops, basic_stra);
  }
  if (ops[iter_ops]->type() == L2_NORMALIZE) {
    return PrepareL2Normalize(ops, iter_ops, basic_stra);
  }
  if (ops[iter_ops]->type() == ADD || ops[iter_ops]->type() == SUB || ops[iter_ops]->type() == MUL ||
      ops[iter_ops]->type() == DIV) {
    return CheckBroadcast(ops, iter_ops, basic_stra);
  }
  if (ops[iter_ops]->type() == SOFTMAX || ops[iter_ops]->type() == LOG_SOFTMAX) {
    return PrepareSoftMax(ops, iter_ops, basic_stra);
  }
  return CheckDivisible(ops, iter_ops, basic_stra);
}

// Function to deal with ops with broadcasting, like TensorAdd/Sub/Mul/Div etc.
Strategies CheckBroadcast(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                          const Dimensions &s) {
  Strategies stra;

  size_t first_tensor_dim = ops[iter_ops]->inputs_tensor_info()[0].shape().size();
  size_t second_tensor_dim = ops[iter_ops]->inputs_tensor_info()[1].shape().size();
  size_t s_dim = s.size();
  // Do Broadcasting in the second tensor.
  if (second_tensor_dim < first_tensor_dim) {
    bool broadcast_first_tensor = false;
    // Push back the first tensor's strategy.
    if (s_dim == first_tensor_dim) {
      stra.push_back(s);
    } else {
      Dimensions broadcast_revise_s(first_tensor_dim, 1);
      stra.push_back(broadcast_revise_s);
    }
    // Push back the second tensor's strategy after applying broadcast.
    stra.push_back(ApplyBroadcast(ops, iter_ops, s, first_tensor_dim, second_tensor_dim, broadcast_first_tensor));
  } else if (second_tensor_dim > first_tensor_dim) {  // Do Broadcasting in the first tensor.
    bool broadcast_first_tensor = true;
    // Push back the first tensor's strategy after applying broadcast.
    stra.push_back(ApplyBroadcast(ops, iter_ops, s, first_tensor_dim, second_tensor_dim, broadcast_first_tensor));
    // Push back the second tensor's strategy.
    if (s_dim == second_tensor_dim) {
      stra.push_back(s);
    } else {
      Dimensions broadcast_revise_s(second_tensor_dim, 1);
      stra.push_back(broadcast_revise_s);
    }
  } else {  // Broadcasting can be ignored or No broadcasting needs to be applied.
    stra = CheckDivisible(ops, iter_ops, s);
  }
  // Strategy protection to avoid that partition number is larger than the shape of related dimension.
  for (size_t i = 0; i < ops[iter_ops]->inputs_tensor_info().size(); i++) {
    for (size_t j = 0; j < ops[iter_ops]->inputs_tensor_info()[i].shape().size(); j++) {
      if (stra[i][j] > ops[iter_ops]->inputs_tensor_info()[i].shape()[j] ||
          ops[iter_ops]->inputs_tensor_info()[i].shape()[j] % stra[i][j] != 0) {
        stra[i][j] = 1;
      }
    }
  }

  return stra;
}

Dimensions ApplyBroadcast(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops, Dimensions s,
                          size_t first_tensor_dim, size_t second_tensor_dim, bool broadcast_first_tensor) {
  Dimensions s_empty = {};
  Dimensions s_broadcast;
  size_t target_tensor_index = 0;
  size_t refer_tensor_index = 0;
  size_t target_tensor_dim;
  size_t refer_tensor_dim;

  // Indexing target and refer tensor.
  if (broadcast_first_tensor) {
    target_tensor_index = 0;
    refer_tensor_index = 1;
    target_tensor_dim = first_tensor_dim;
    refer_tensor_dim = second_tensor_dim;
  } else {
    target_tensor_index = 1;
    refer_tensor_index = 0;
    target_tensor_dim = second_tensor_dim;
    refer_tensor_dim = first_tensor_dim;
  }

  // When target tensor with an empty dim.
  if (target_tensor_dim == 0) {
    return s_empty;
  } else if (target_tensor_dim == 1) {  // When target tensor with a single dim.
    bool broadcast_dim_found = false;
    for (size_t iter = 0; iter < refer_tensor_dim; iter++) {
      // Find and copy that dim's strategy from the refer tensor.
      if ((ops[iter_ops]->inputs_tensor_info()[refer_tensor_index].shape()[iter] ==
           ops[iter_ops]->inputs_tensor_info()[target_tensor_index].shape()[0]) &&
          (ops[iter_ops]->inputs_tensor_info()[refer_tensor_index].shape()[iter] > 1) &&
          (refer_tensor_dim == s.size())) {
        s_broadcast.push_back(s.at(iter));
        broadcast_dim_found = true;
        break;
      }
    }
    // Cannot decide which dim it is, push back one.
    if (broadcast_dim_found == false) {
      s_broadcast.push_back(1);
    }
  } else {
    // Cannot decide which dim needs to do broadcast, push back one(s).
    for (size_t iter = 0; iter < target_tensor_dim; iter++) {
      s_broadcast.push_back(1);
    }
  }

  return s_broadcast;
}

// Check whether the operator can be divided by the current strategy.
Strategies CheckDivisible(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                          const Dimensions &basic_stra) {
  Dimensions s_empty = {};
  Strategies stra;

  // For all the input tensors.
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
    // If input tensor is empty, return strategy as void.
    if (ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape().size() == 0) {
      stra.push_back(s_empty);
      continue;
    }

    Dimensions tmp_stra = basic_stra;
    bool modified = false;

    // Make sure each tensor's dim shape is greater than 1. If not, push back strategy as 1 instead.
    for (size_t j = 0; j < ops[iter_ops]->inputs_tensor_info()[iter_op_inputs].shape().size(); j++) {
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

void GenerateEliminatedOperatorStrategyForward(const std::shared_ptr<Graph> &graph,
                                               const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                               const std::vector<std::vector<std::string>> &input_tensor_names,
                                               const std::shared_ptr<std::vector<size_t>> &index_list,
                                               const std::shared_ptr<std::vector<size_t>> &no_stra_op_list) {
  if (no_stra_op_list->size() == 0) {
    return;
  }
  std::vector<size_t> no_stra_op_list_bis;

  for (size_t iter_list = no_stra_op_list->size(); iter_list > 0; iter_list--) {
    size_t iter_ops = no_stra_op_list->at(iter_list - 1);
    Strategies stra;
    Dimensions s;
    size_t incoming_op_index = FindIndexOfOperatorIncoming(input_tensor_names, iter_ops);
    if (incoming_op_index != SIZE_MAX) {
      auto iter_graph = index_list->at(incoming_op_index);
      if (iter_graph != SIZE_MAX) {
        s = CopyIncomingOperatorOutputStrategy(graph, ops, iter_ops, iter_graph, incoming_op_index);
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

Dimensions ModifyStrategyIfSqueezeOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                           Dimensions s) {
  Dimensions s_Squeeze;
  auto axis_list = GetAxisList(ops, SizeToLong(iter_ops));
  size_t s_index = 0;
  size_t axis_list_index = 0;
  for (size_t i = 0; i < s.size() + axis_list.size(); i++) {
    if (i == LongToSize(axis_list[axis_list_index])) {
      s_Squeeze.push_back(1);
      axis_list_index++;
    } else {
      s_Squeeze.push_back(s[s_index]);
      s_index++;
    }
  }

  size_t cut = 1;
  for (size_t i = 0; i < s_Squeeze.size(); i++) {
    cut *= LongToSize(s_Squeeze[i]);
  }
  if (cut != size_t(g_device_manager->stage_device_num())) {
    s_Squeeze.clear();
  }

  return s_Squeeze;
}

Dimensions CopyOutgoingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                             const std::vector<std::vector<std::string>> &input_tensor_names,
                                             const size_t iter_ops) {
  Dimensions s;
  if (ops[iter_ops]->type() == REDUCE_MAX || ops[iter_ops]->type() == REDUCE_MIN ||
      ops[iter_ops]->type() == EXPAND_DIMS || ops[iter_ops]->type() == REDUCE_SUM ||
      ops[iter_ops]->type() == REDUCE_MEAN || ops[iter_ops]->type() == RESHAPE || ops[iter_ops]->type() == GATHERV2 ||
      ops[iter_ops]->type() == TRANSPOSE || ops[iter_ops]->type() == ARGMAXWITHVALUE ||
      ops[iter_ops]->type() == ARGMINWITHVALUE) {
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
    for (size_t k = 0; k < ops[iter_ops]->outputs_tensor_info()[0].shape().size(); ++k) {
      s.push_back(ops[outgoing_op_index]->selected_strategy()->GetInputDim()[iter_op_inputs][k]);
    }
  }
  return s;
}

void GenerateEliminatedOperatorStrategyBackward(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const std::vector<std::vector<std::string>> &input_tensor_names,
                                                const std::shared_ptr<std::vector<size_t>> &no_stra_op_list) {
  if (no_stra_op_list->size() == 0) {
    return;
  }
  std::vector<size_t> no_stra_op_list_bis;

  for (size_t iter_list = no_stra_op_list->size(); iter_list > 0; iter_list--) {
    auto iter_ops = no_stra_op_list->at(iter_list - 1);
    Strategies stra;
    Dimensions s = CopyOutgoingOperatorInputStrategy(ops, input_tensor_names, iter_ops);
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

void GenerateRemainingOperatorStrategy(const std::shared_ptr<Graph> &graph,
                                       const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                       const std::vector<std::vector<std::string>> &input_tensor_names,
                                       const std::shared_ptr<std::vector<size_t>> &index_list,
                                       const std::shared_ptr<std::vector<size_t>> &no_stra_op_list) {
  if (no_stra_op_list->size() == 0) {
    return;
  }

  size_t no_stra_op_list_size = no_stra_op_list->size();
  do {
    no_stra_op_list_size = no_stra_op_list->size();
    GenerateEliminatedOperatorStrategyForward(graph, ops, input_tensor_names, index_list, no_stra_op_list);
    GenerateEliminatedOperatorStrategyBackward(ops, input_tensor_names, no_stra_op_list);
  } while (no_stra_op_list_size > no_stra_op_list->size());

  for (size_t iter_list = 0; iter_list < no_stra_op_list->size(); iter_list++) {
    auto iter_ops = no_stra_op_list->at(iter_list);
    Strategies stra;
    Dimensions s;

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
