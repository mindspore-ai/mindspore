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

#include "optimizer/parallel/auto_parallel/rec_core/rec_generate_strategy.h"

#include <vector>
#include <algorithm>
#include <memory>

#include "optimizer/parallel/ops_info/operator_info.h"
#include "optimizer/parallel/auto_parallel/rec_core/rec_partition.h"
#include "optimizer/parallel/strategy.h"
#include "ir/value.h"

namespace mindspore {
namespace parallel {
void GenerateStrategy(const std::shared_ptr<Graph> graph, std::vector<std::shared_ptr<OperatorInfo>> ops,
                      const std::shared_ptr<std::vector<size_t>> ops_nodes_list,
                      const std::shared_ptr<std::vector<size_t>> index_list,
                      const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list) {
  MaskNoSupportedOps(graph);
  for (size_t iter_ops = 0; iter_ops < ops.size(); iter_ops++) {
    auto type = ops[iter_ops]->type();
    size_t iter_nodes = index_list->at(ops_nodes_list->at(iter_ops));
    std::vector<std::vector<int32_t>> stra;
    iter_nodes = IterNodes(ops_nodes_list, index_list, eli_list, iter_ops, iter_nodes);
    for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_tensor_info().size(); iter_op_inputs++) {
      std::vector<int32_t> s = PrepareStrategy(graph, ops, type, iter_ops, iter_nodes, iter_op_inputs);
      stra.push_back(s);
    }
    StrategyPtr sp = std::make_shared<Strategy>(0, stra);
    ops[iter_ops]->SetSelectedStrategyAndCost(sp, ops[iter_ops]->selected_cost());
  }
}

size_t IterNodes(const std::shared_ptr<std::vector<size_t>> ops_nodes_list,
                 const std::shared_ptr<std::vector<size_t>> index_list,
                 const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list, const size_t iter_ops,
                 size_t iter_nodes) {
  if (iter_nodes > SIZE_MAX / 2) {
    for (size_t iter_eli = 0; iter_eli < eli_list->size(); iter_eli++) {
      if (eli_list->at(iter_eli)[0] == ops_nodes_list->at(iter_ops)) {
        iter_nodes = index_list->at(eli_list->at(iter_eli)[1]);
        break;
      }
    }
  }
  return iter_nodes;
}

void PrepareMatMul(const std::shared_ptr<Graph> graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                   const size_t iter_ops, const size_t iter_nodes, const size_t iter_op_inputs,
                   std::vector<int32_t> s) {
  auto attrs = ops[iter_ops]->attrs();
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
}

void PrepareConv2D(const std::shared_ptr<Graph> graph, const size_t iter_nodes, size_t iter_op_inputs,
                   std::vector<int32_t> s) {
  if (iter_op_inputs == 0) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[0].tensor_str.str_n));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[0].tensor_str.str_c));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[0].tensor_str.str_h));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[0].tensor_str.str_w));
  } else {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[1].tensor_str.str_n));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[1].tensor_str.str_c));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[1].tensor_str.str_h));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[1].tensor_str.str_w));
  }
}

void PrepareBiasAdd(const std::shared_ptr<Graph> graph, const size_t iter_nodes, const size_t iter_op_inputs,
                    std::vector<int32_t> s) {
  if (iter_op_inputs == 0) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_h));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_w));
  } else {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_w));
  }
}

void PrepareBN(const std::shared_ptr<Graph> graph, const size_t iter_nodes, const size_t iter_op_inputs,
               std::vector<int32_t> s) {
  if (iter_op_inputs == 0) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[0].tensor_str.str_n));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[0].tensor_str.str_c));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[0].tensor_str.str_h));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[0].tensor_str.str_w));
  } else {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[1].tensor_str.str_w));
  }
}

void PrepareSparse(const size_t iter_op_inputs, std::vector<int32_t> s) {
  if (iter_op_inputs == 0) {
    s.push_back(g_device_manager->DeviceNum());
    s.push_back(1);
  } else {
    s.push_back(g_device_manager->DeviceNum());
  }
}

void RefillOrigin(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                  const size_t iter_op_inputs, std::vector<int32_t> s) {
  StrategyPtr origin_strategy = ops[iter_ops]->strategy();
  if (iter_op_inputs == 0) {
    for (size_t j = 0; j < origin_strategy->GetInputDim()[0].size(); j++) {
      s.push_back(1);
    }
  } else {
    for (size_t k = 0; k < origin_strategy->GetInputDim()[iter_op_inputs].size(); k++) {
      s.push_back(1);
    }
  }
}

std::vector<int32_t> PrepareStrategy(const std::shared_ptr<Graph> graph,
                                     const std::vector<std::shared_ptr<OperatorInfo>> &ops, const std::string &type,
                                     const size_t iter_ops, const size_t iter_nodes, const size_t iter_op_inputs) {
  std::vector<int32_t> s;
  if (type == MATMUL) {
    PrepareMatMul(graph, ops, iter_ops, iter_nodes, iter_op_inputs, s);
  } else if ((type == MAXPOOL) || (type == SIMPLE_MEAN) || (type == TENSOR_ADD)) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_n));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_c));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_h));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].apply.arguments[iter_op_inputs].tensor_str.str_w));
  } else if (type == CONV2D) {
    PrepareConv2D(graph, iter_nodes, iter_op_inputs, s);
  } else if (type == BIAS_ADD) {
    PrepareBiasAdd(graph, iter_nodes, iter_op_inputs, s);
  } else if (type == RESHAPE) {
    s.push_back(1);
    s.push_back(1);
    s.push_back(1);
    s.push_back(1);
  } else if (type == RELU) {
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].tensor_parm.tensor_str.str_n));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].tensor_parm.tensor_str.str_c));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].tensor_parm.tensor_str.str_h));
    s.push_back(static_cast<int32_t>(1.0 / graph->nodes[iter_nodes].tensor_parm.tensor_str.str_w));
  } else if (type == BATCH_NORM || (type == FUSE_BATCH_NORM)) {
    PrepareBN(graph, iter_nodes, iter_op_inputs, s);
  } else if (type == SOFTMAX_CROSS_ENTROPY_WITH_LOGITS) {
    PrepareSparse(iter_op_inputs, s);
  } else {
    RefillOrigin(ops, iter_ops, iter_op_inputs, s);
  }
  return s;
}

void MaskNoSupportedOps(const std::shared_ptr<Graph> graph) {
  size_t iter_nodes = graph->nodes.size();
  for (size_t i = 0; i < iter_nodes; i++) {
    if (0 == graph->nodes[i].info) {
      Graph::NodeType &node = graph->nodes[i];

      if (node.apply.op_type == 1) {  // For Convolution
        // cover input tensor strategy
        node.apply.arguments[0].tensor_str.str_n = 1.0 / static_cast<float>(g_device_manager->DeviceNum());
        node.apply.arguments[0].tensor_str.str_c = 1;
        node.apply.arguments[0].tensor_str.str_h = 1;
        node.apply.arguments[0].tensor_str.str_w = 1;
        // cover filter tensor strategy
        node.apply.arguments[1].tensor_str.str_n = 1;
        node.apply.arguments[1].tensor_str.str_c = 1;
        node.apply.arguments[1].tensor_str.str_h = 1;
        node.apply.arguments[1].tensor_str.str_w = 1;
      } else if (node.apply.op_type == 8) {  // For BN
        node.apply.arguments[0].tensor_str.str_n = 1.0 / static_cast<float>(g_device_manager->DeviceNum());
        node.apply.arguments[0].tensor_str.str_c = 1;
        node.apply.arguments[0].tensor_str.str_h = 1;
        node.apply.arguments[0].tensor_str.str_w = 1;
        // cover 1-d argument blobs
        node.apply.arguments[1].tensor_str.str_w = 1;
        node.apply.arguments[2].tensor_str.str_w = 1;
        node.apply.arguments[3].tensor_str.str_w = 1;
        node.apply.arguments[4].tensor_str.str_w = 1;
      } else if (node.apply.op_type == 4 || node.apply.op_type == 9) {  // For SparseSoftmaxCrossEntropyWithLogits
        node.tensor_parm.tensor_str.str_h = 1.0 / static_cast<float>(g_device_manager->DeviceNum());
        node.tensor_parm.tensor_str.str_w = 1;
      }
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
