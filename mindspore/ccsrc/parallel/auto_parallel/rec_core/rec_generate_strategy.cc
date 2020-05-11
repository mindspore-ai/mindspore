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
void GenerateStrategy(std::shared_ptr<Graph> graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops) {
  MS_EXCEPTION_IF_NULL(graph);

  for (size_t iter_ops = 0; iter_ops < ops.size(); iter_ops++) {
    std::vector<std::vector<int32_t>> stra;
    for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
      stra.push_back(PrepareStrategy(graph, ops, iter_ops, iter_op_inputs));
    }
    // OneHot's scalar parameters were removed by entire_costgraph, we had to complete them.
    if (ops[iter_ops]->type() == ONEHOT) {
      std::vector<int32_t> s_Onehot = {};
      stra.push_back(s_Onehot);
      stra.push_back(s_Onehot);
    }
    StrategyPtr sp = std::make_shared<Strategy>(0, stra);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }
}

std::vector<int32_t> PrepareMatMul(const std::shared_ptr<Graph> &graph,
                                   const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_nodes,
                                   const size_t iter_op_inputs) {
  std::vector<int32_t> s;
  auto attrs = ops[iter_nodes]->attrs();
  bool transpose_a = attrs[TRANSPOSE_A]->cast<BoolImmPtr>()->value();
  bool transpose_b = attrs[TRANSPOSE_B]->cast<BoolImmPtr>()->value();
  if (transpose_a && (iter_op_inputs == 0)) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_w));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_h));
  } else if (transpose_b && (iter_op_inputs == 1)) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_w));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_h));
  } else {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_h));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_w));
  }
  return s;
}

std::vector<int32_t> MakeRecSearchStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                           const std::shared_ptr<Graph> &graph, const size_t iter_ops,
                                           const size_t iter_op_inputs) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  StrategyPtr origin_strategy = ops[iter_ops]->strategy();

  if (iter_op_inputs >= origin_strategy->GetInputDim().size()) {
    MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
  }

  // size_t output_size = ops[iter_ops]->outputs_tensor_info()[0].shape().size();
  size_t output_size = origin_strategy->GetInputDim()[iter_op_inputs].size();

  std::vector<int32_t> s = {};
  if (output_size == 4) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_ops].apply.arguments[iter_op_inputs].tensor_str.str_n));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_ops].apply.arguments[iter_op_inputs].tensor_str.str_c));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_ops].apply.arguments[iter_op_inputs].tensor_str.str_h));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_ops].apply.arguments[iter_op_inputs].tensor_str.str_w));
  } else if (output_size == 2) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_ops].apply.arguments[iter_op_inputs].tensor_str.str_h));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_ops].apply.arguments[iter_op_inputs].tensor_str.str_w));
  } else if (output_size == 1) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_ops].apply.arguments[iter_op_inputs].tensor_str.str_w));
  } else if (output_size == 0) {
    return s;
  } else {
    MS_LOG(ERROR) << "Tensor's output size is unexcepted.";
  }

  return s;
}

std::vector<int32_t> MakeDataParallelStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                              const size_t iter_ops, const size_t iter_op_inputs) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  StrategyPtr origin_strategy = ops[iter_ops]->strategy();

  if (iter_op_inputs >= origin_strategy->GetInputDim().size()) {
    MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
  }

  std::vector<int32_t> s;
  size_t input_size = origin_strategy->GetInputDim()[iter_op_inputs].size();
  for (size_t dim = 0; dim < input_size; dim++) {
    if (dim == 0 && input_size == 4) {
      size_t max_device_num = g_device_manager->DeviceNum();
      size_t target_tensor_batch = ops[iter_ops]->outputs_tensor_info()[0].shape()[0];
      s.push_back(std::min(max_device_num, target_tensor_batch));
    } else {
      s.push_back(1);
    }
  }

  return s;
}

std::vector<int32_t> PrepareStrategy(const std::shared_ptr<Graph> &graph,
                                     const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                     const size_t iter_op_inputs) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  auto type = ops[iter_ops]->type();
  auto idx = DictOpType.find(type);
  if (idx == DictOpType.end()) {
    return MakeDataParallelStrategy(ops, iter_ops, iter_op_inputs);
  }

  if (type == MATMUL) {
    return PrepareMatMul(graph, ops, iter_ops, iter_op_inputs);
  } else if (type == RESHAPE) {
    return MakeDataParallelStrategy(ops, iter_ops, iter_op_inputs);
  } else if (type == DIV || type == SUB || type == MUL) {
    return MakeDataParallelStrategy(ops, iter_ops, iter_op_inputs);
  } else {
    return MakeRecSearchStrategy(ops, graph, iter_ops, iter_op_inputs);
  }
}

}  // namespace parallel
}  // namespace mindspore
