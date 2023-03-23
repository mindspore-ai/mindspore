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

#include "frontend/parallel/auto_parallel/rec_core/rec_cost.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "include/common/utils/parallel_context.h"
#include "ir/anf.h"

namespace mindspore {
namespace parallel {
// Compute redistributed cost
double CostRedis(const Graph::NodeType &node,
                 const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                 const std::vector<std::vector<float>> &mode, const Graph &graph) {
  // Store value of cost redist
  double cost_redis = 0;

  // Number of current strategies.
  size_t num_strategy = node_name_to_strategy.size();

  // Number of node-in and node-out
  size_t num_node_in = node.node_in.size();
  size_t num_node_out = node.node_out.size();

  // Set tensor edge value with original tensor shape and cutting times.
  double input_tensor = node.apply.arguments[0].tensor_shape.shape_n * node.apply.arguments[0].tensor_str.str_n *
                        node.apply.arguments[0].tensor_shape.shape_c * node.apply.arguments[0].tensor_str.str_c *
                        node.apply.arguments[0].tensor_shape.shape_h * node.apply.arguments[0].tensor_str.str_h *
                        node.apply.arguments[0].tensor_shape.shape_w * node.apply.arguments[0].tensor_str.str_w;

  double output_tensor = node.tensor_parm.tensor_shape.shape_n * node.tensor_parm.tensor_str.str_n *
                         node.tensor_parm.tensor_shape.shape_c * node.tensor_parm.tensor_str.str_c *
                         node.tensor_parm.tensor_shape.shape_h * node.tensor_parm.tensor_str.str_h *
                         node.tensor_parm.tensor_shape.shape_w * node.tensor_parm.tensor_str.str_w;

  // For each strategy candidate.
  for (size_t i_strategy = 0; i_strategy < num_strategy; i_strategy++) {
    // Find its forward nodes
    for (size_t i_node = 0; i_node < num_node_in; i_node++) {
      if (graph.nodes[node.node_in[i_node]].name == node_name_to_strategy[i_strategy].first) {
        bool is_search_forward = true;
        MS_LOG(INFO) << "Node_in, index =  " << i_node << node_name_to_strategy[i_strategy].first;
        cost_redis +=
          CostRedisWithAdjacentNode(node_name_to_strategy, mode, i_strategy, i_node, input_tensor, is_search_forward);
      }
    }

    // Find its backward nodes
    for (size_t i_node = 0; i_node < num_node_out; i_node++) {
      if (graph.nodes[node.node_out[i_node]].name == node_name_to_strategy[i_strategy].first) {
        bool is_search_forward = false;
        cost_redis +=
          CostRedisWithAdjacentNode(node_name_to_strategy, mode, i_strategy, i_node, output_tensor, is_search_forward);
      }
    }
  }

  return cost_redis;
}

double CostRedisWithAdjacentNode(const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                 const std::vector<std::vector<float>> &mode, size_t i_strategy, size_t i_node,
                                 double tensor_size, bool search_forward) {
  double new_redis_cost = 0;
  int64_t counter = 0;

  if (search_forward) {
    if (static_cast<int64_t>(1 / node_name_to_strategy[i_strategy].second.outputTensor.str_n) !=
        static_cast<int64_t>(1 / mode[i_node][0])) {
      counter += 1;
    }
    if (static_cast<int64_t>(1 / node_name_to_strategy[i_strategy].second.outputTensor.str_c) !=
        static_cast<int64_t>(1 / mode[i_node][1])) {
      counter += 1;
    }
    if (static_cast<int64_t>(1 / node_name_to_strategy[i_strategy].second.outputTensor.str_h) !=
        static_cast<int64_t>(1 / mode[i_node][2])) {
      counter += 1;
    }
    if (static_cast<int64_t>(1 / node_name_to_strategy[i_strategy].second.outputTensor.str_w) !=
        static_cast<int64_t>(1 / mode[i_node][3])) {
      counter += 1;
    }
  } else {
    if (static_cast<int64_t>(1 / node_name_to_strategy[i_strategy].second.inputTensor[0].str_n) !=
        static_cast<int64_t>(1 / mode[2][0])) {
      counter += 1;
    }
    if (static_cast<int64_t>(1 / node_name_to_strategy[i_strategy].second.inputTensor[0].str_c) !=
        static_cast<int64_t>(1 / mode[2][1])) {
      counter += 1;
    }
    if (static_cast<int64_t>(1 / node_name_to_strategy[i_strategy].second.inputTensor[0].str_h) !=
        static_cast<int64_t>(1 / mode[2][2])) {
      counter += 1;
    }
    if (static_cast<int64_t>(1 / node_name_to_strategy[i_strategy].second.inputTensor[0].str_w) !=
        static_cast<int64_t>(1 / mode[2][3])) {
      counter += 1;
    }
  }

  if (counter >= 1) {
    new_redis_cost = tensor_size * REDIS_COEF;
  } else if (counter == 0) {
    new_redis_cost = 0;
  } else {
    MS_LOG(EXCEPTION) << "Failure: CostRedis failed.";
  }

  MS_LOG(INFO) << "redis_cost = " << new_redis_cost;

  return new_redis_cost;
}

// Get optimal strategy for MatMul
StrategyRec CostMatMul::GetOptimalStr(const Graph::NodeType &node,
                                      const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                      const Graph &graph, const bool isTraining) {
  int64_t edge_i =
    static_cast<int64_t>(node.apply.arguments[0].tensor_shape.shape_h * node.apply.arguments[0].tensor_str.str_h);
  int64_t edge_j =
    static_cast<int64_t>(node.apply.arguments[1].tensor_shape.shape_w * node.apply.arguments[1].tensor_str.str_w);
  int64_t edge_k =
    static_cast<int64_t>(node.apply.arguments[0].tensor_shape.shape_w * node.apply.arguments[0].tensor_str.str_w);

  std::vector<double> cost_op;

  if (edge_i < 2 || edge_i % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 0.5, 1}, {1, 1, 1, 1}, {1, 1, 0.5, 1}};
    cost_op.push_back(StrConcatDimI(edge_j, edge_k) + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  // Do not partition the J-axis and K-axis for the same MatMul
  if (edge_j < 2 || edge_j % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 1, 1}, {1, 1, 1, 0.5}, {1, 1, 1, 0.5}};
    cost_op.push_back(StrConcatDimJ(edge_i, edge_k) + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  if (edge_k < 2 || edge_k % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 1, 0.5}, {1, 1, 0.5, 1}, {1, 1, 1, 1}};
    cost_op.push_back(StrReduceDimK(edge_i, edge_j) + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  // If optimizer parallel is enabled, then MatMul must be cut at least once on the DP dimension
  // node.apply.arguments[0].tensor_str.str_h == 1 means that the batch dimension is not partitioned.
  if (ParallelContext::GetInstance()->enable_parallel_optimizer() && node.apply.arguments[0].tensor_str.str_h == 1 &&
      isTraining) {
    cost_op[0] = DOUBLE_MIN;
  }

  return ChoseStr(cost_op, node.apply.str);
}

// Get weight for MatMul
double CostMatMul::GetMaxCostIn(const OperatorRec &op) {
  int64_t edge_i = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_h * op.arguments[0].tensor_str.str_h);
  int64_t edge_j = static_cast<int64_t>(op.arguments[1].tensor_shape.shape_w * op.arguments[1].tensor_str.str_w);
  int64_t edge_k = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_w * op.arguments[0].tensor_str.str_w);

  std::vector<double> cost_in;
  cost_in.push_back(StrConcatDimI(edge_j, edge_k));
  cost_in.push_back(StrConcatDimJ(edge_i, edge_k));
  cost_in.push_back(StrReduceDimK(edge_i, edge_j));

  return *max_element(cost_in.begin(), cost_in.end());
}

// Chose strategy for MatMul
StrategyRec CostMatMul::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) const {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] > (DOUBLE_MAX - 0.1)) {
    return str;
  }

  switch (min_position) {
    case 0:
      str.inputTensor[0].str_h /= 2.0;
      str.outputTensor.str_h /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_i_;
      break;

    case 1:
      str.inputTensor[1].str_w /= 2.0;
      str.outputTensor.str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_j_;
      break;

    case 2:
      str.inputTensor[0].str_w /= 2.0;
      str.inputTensor[1].str_h /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_k_;
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure:CostMatMul failed.";
  }

  return str;
}

// Get optimal strategy for Conv
StrategyRec CostConvolution::GetOptimalStr(
  const Graph::NodeType &node, const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
  const Graph &graph, bool channel_partition) {
  const OperatorRec &op = node.apply;

  int64_t input_tensor_h =
    static_cast<int64_t>(op.arguments[0].tensor_shape.shape_h * op.arguments[0].tensor_str.str_h);
  int64_t input_tensor_w =
    static_cast<int64_t>(op.arguments[0].tensor_shape.shape_w * op.arguments[0].tensor_str.str_w);
  int64_t input_tensor_n =
    static_cast<int64_t>(op.arguments[0].tensor_shape.shape_n * op.arguments[0].tensor_str.str_n);
  int64_t input_tensor_c =
    static_cast<int64_t>(op.arguments[0].tensor_shape.shape_c * op.arguments[0].tensor_str.str_c);

  int64_t tensor_in = input_tensor_h * input_tensor_w * input_tensor_n * input_tensor_c;

  int64_t tensor_filter_h =
    static_cast<int64_t>(op.arguments[1].tensor_shape.shape_h * op.arguments[1].tensor_str.str_h);
  int64_t tensor_filter_w =
    static_cast<int64_t>(op.arguments[1].tensor_shape.shape_w * op.arguments[1].tensor_str.str_w);
  int64_t tensor_filter_n =
    static_cast<int64_t>(op.arguments[1].tensor_shape.shape_n * op.arguments[1].tensor_str.str_n);
  int64_t tensor_filter_c =
    static_cast<int64_t>(op.arguments[1].tensor_shape.shape_c * op.arguments[1].tensor_str.str_c);

  int64_t tensor_filter = tensor_filter_h * tensor_filter_w * tensor_filter_n * tensor_filter_c;

  int64_t output_tensor_h =
    static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_h * node.tensor_parm.tensor_str.str_h);
  int64_t output_tensor_w =
    static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_w * node.tensor_parm.tensor_str.str_w);
  int64_t output_tensor_n =
    static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_n * node.tensor_parm.tensor_str.str_n);
  int64_t output_tensor_c =
    static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_c * node.tensor_parm.tensor_str.str_c);

  int64_t tensor_out = output_tensor_h * output_tensor_w * output_tensor_n * output_tensor_c;

  std::vector<double> cost_op;
  cost_op.reserve(7);

  if (input_tensor_n < 2 || input_tensor_n % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{0.5, 1, 1, 1}, {1, 1, 1, 1}, {0.5, 1, 1, 1}};
    cost_op.push_back(StrDimB(tensor_filter) + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  cost_op.push_back(DOUBLE_MAX);
  cost_op.push_back(DOUBLE_MAX);

  if (channel_partition == false || tensor_filter < 2 || tensor_filter % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 1, 1}, {0.5, 1, 1, 1}, {1, 0.5, 1, 1}};
    cost_op.push_back(StrDimK(tensor_in) + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  cost_op.push_back(DOUBLE_MAX);
  cost_op.push_back(DOUBLE_MAX);

  if (channel_partition == false || tensor_filter_c < 2 || tensor_filter_c % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 0.5, 1, 1}, {1, 0.5, 1, 1}, {1, 1, 1, 1}};
    cost_op.push_back(StrDimQ(tensor_out) + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  return ChoseStr(cost_op, node.apply.str);
}

// Get weight for Conv
double CostConvolution::GetMinCostIn(const Graph::NodeType &node) {
  const OperatorRec &op = node.apply;

  int64_t tensor_in = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_h * op.arguments[0].tensor_str.str_h) *
                      static_cast<int64_t>(op.arguments[0].tensor_shape.shape_n * op.arguments[0].tensor_str.str_n) *
                      static_cast<int64_t>(op.arguments[0].tensor_shape.shape_w * op.arguments[0].tensor_str.str_w) *
                      static_cast<int64_t>(op.arguments[0].tensor_shape.shape_c * op.arguments[0].tensor_str.str_c);
  int64_t tensor_filter =
    static_cast<int64_t>(op.arguments[1].tensor_shape.shape_h * op.arguments[1].tensor_str.str_h) *
    static_cast<int64_t>(op.arguments[1].tensor_shape.shape_n * op.arguments[1].tensor_str.str_n) *
    static_cast<int64_t>(op.arguments[1].tensor_shape.shape_w * op.arguments[1].tensor_str.str_w) *
    static_cast<int64_t>(op.arguments[1].tensor_shape.shape_c * op.arguments[1].tensor_str.str_c);
  int64_t tensor_out = static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_h * node.tensor_parm.tensor_str.str_h) *
                       static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_n * node.tensor_parm.tensor_str.str_n) *
                       static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_w * node.tensor_parm.tensor_str.str_w) *
                       static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_c * node.tensor_parm.tensor_str.str_c);

  std::vector<double> cost_in;
  cost_in.push_back(StrDimB(tensor_filter));
  cost_in.push_back(StrDimI(tensor_in, tensor_filter));
  cost_in.push_back(StrDimJ(tensor_in, tensor_filter));
  cost_in.push_back(StrDimK(tensor_in));
  cost_in.push_back(StrDimDI(tensor_in, tensor_out));
  cost_in.push_back(StrDimDJ(tensor_in, tensor_out));
  cost_in.push_back(StrDimQ(tensor_out));

  return *min_element(cost_in.begin(), cost_in.end());
}

// Chose strategy for Conv
StrategyRec CostConvolution::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) const {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] > (DOUBLE_MAX - 0.1)) {
    return str;
  }

  switch (min_position) {
    case 0:
      str.inputTensor[0].str_n /= 2.0;
      str.outputTensor.str_n /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_b_;
      break;

    case 1:
      str.inputTensor[0].str_h /= 2.0;
      str.outputTensor.str_h /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_i_;
      break;

    case 2:
      str.inputTensor[0].str_w /= 2.0;
      str.outputTensor.str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_j_;
      break;

    case 3:
      str.inputTensor[1].str_n /= 2.0;
      str.outputTensor.str_c /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_k_;
      break;

    case 4:
      str.inputTensor[1].str_h /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_di_;
      break;

    case 5:
      str.inputTensor[1].str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_dj_;
      break;

    case 6:
      str.inputTensor[0].str_c /= 2.0;
      str.inputTensor[1].str_c /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_q_;
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: CostConvolution failed.";
  }
  return str;
}

// Get optimal strategy for Pooling
StrategyRec CostPooling::GetOptimalStr(const Graph::NodeType &node,
                                       const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                       const Graph &graph) const {
  int64_t tensor_n = static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_n * node.tensor_parm.tensor_str.str_n);
  int64_t tensor_c = static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_c * node.tensor_parm.tensor_str.str_c);

  std::vector<double> cost_op;

  if (tensor_n < 2 || tensor_n % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{0.5, 1, 1, 1}, {0.5, 1, 1, 1}, {0.5, 1, 1, 1}};
    cost_op.push_back(cost_in_ + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  if (tensor_c < 2 || tensor_c % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 0.5, 1, 1}, {1, 0.5, 1, 1}, {1, 0.5, 1, 1}};
    cost_op.push_back(cost_in_ + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  cost_op.push_back(DOUBLE_MAX);
  cost_op.push_back(DOUBLE_MAX);

  return ChoseStr(cost_op, node.apply.str);
}

// Chose strategy for Pooling
StrategyRec CostPooling::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) const {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] > (DOUBLE_MAX - 0.1)) {
    return str;
  }

  switch (min_position) {
    case 0:
      str.inputTensor[0].str_n /= 2.0;
      str.outputTensor.str_n /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 1:
      str.inputTensor[0].str_c /= 2.0;
      str.outputTensor.str_c /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 2:
      str.inputTensor[0].str_h /= 2.0;
      str.outputTensor.str_h /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 3:
      str.inputTensor[0].str_w /= 2.0;
      str.outputTensor.str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: CostPooling failed.";
  }
  return str;
}

// Chose strategy for Add
StrategyRec CostTensorAdd::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] > (DOUBLE_MAX - 0.1)) {
    return str;
  }

  switch (min_position) {
    case 0:
      str.inputTensor[0].str_n /= 2.0;
      str.inputTensor[1].str_n /= 2.0;
      str.outputTensor.str_n /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 1:
      str.inputTensor[0].str_c /= 2.0;
      str.inputTensor[1].str_c /= 2.0;
      str.outputTensor.str_c /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 2:
      str.inputTensor[0].str_h /= 2.0;
      str.inputTensor[1].str_h /= 2.0;
      str.outputTensor.str_h /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 3:
      str.inputTensor[0].str_w /= 2.0;
      str.inputTensor[1].str_w /= 2.0;
      str.outputTensor.str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: CostAdd failed.";
  }
  return str;
}

// Get optimal strategy for Reshape
StrategyRec CostReshape::GetOptimalStr(const Graph::NodeType &node) const { return ChoseStr(node.apply.str); }

StrategyRec CostReshape::ChoseStr(StrategyRec str) const { return str; }

// Chose strategy for BiasAdd
StrategyRec CostBiasAdd::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] > (DOUBLE_MAX - 0.1)) {
    return str;
  }

  switch (min_position) {
    case 0:
      str.inputTensor[0].str_n /= 2.0;
      str.outputTensor.str_n /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 1:
      str.inputTensor[0].str_c /= 2.0;
      str.outputTensor.str_c /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 2:
      str.inputTensor[0].str_h /= 2.0;
      str.outputTensor.str_h /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 3:
      str.inputTensor[0].str_w /= 2.0;
      str.inputTensor[1].str_w /= 2.0;
      str.outputTensor.str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: CostBiasAdd failed.";
  }
  return str;
}

// Get optimal strategy for Common OPs
StrategyRec CostCommon::GetOptimalStr(const Graph::NodeType &node,
                                      const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                      const Graph &graph) {
  const OperatorRec &op = node.apply;
  int64_t tensor_n = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_n * op.arguments[0].tensor_str.str_n);
  int64_t tensor_c = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_c * op.arguments[0].tensor_str.str_c);
  int64_t tensor_h = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_h * op.arguments[0].tensor_str.str_h);
  int64_t tensor_w = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_w * op.arguments[0].tensor_str.str_w);

  std::vector<double> cost_op;

  if (tensor_n < 2 || tensor_n % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{0.5, 1, 1, 1}, {0.5, 1, 1, 1}, {0.5, 1, 1, 1}};
    cost_op.push_back(cost_in_ + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  if (tensor_c < 2 || tensor_c % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 0.5, 1, 1}, {1, 0.5, 1, 1}, {1, 0.5, 1, 1}};
    cost_op.push_back(cost_in_ + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  if (tensor_h < 2 || tensor_h % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 0.5, 1}, {1, 1, 0.5, 1}, {1, 1, 0.5, 1}};
    cost_op.push_back(cost_in_ + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  if (tensor_w < 2 || tensor_w % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 1, 0.5}, {1, 1, 1, 0.5}, {1, 1, 1, 0.5}};
    cost_op.push_back(cost_in_ + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  return ChoseStr(cost_op, node.apply.str);
}

// Chose strategy for Common op
StrategyRec CostCommon::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] > (DOUBLE_MAX - 0.1)) {
    return str;
  }

  switch (min_position) {
    case 0:
      str.inputTensor[0].str_n /= 2.0;
      str.outputTensor.str_n /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 1:
      str.inputTensor[0].str_c /= 2.0;
      str.outputTensor.str_c /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 2:
      str.inputTensor[0].str_h /= 2.0;
      str.outputTensor.str_h /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 3:
      str.inputTensor[0].str_w /= 2.0;
      str.outputTensor.str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: Common failed.";
  }
  return str;
}

// Get optimal strategy for BatchParallel OPs
StrategyRec CostBatchParallel::GetOptimalStr(const Graph::NodeType &node) {
  const OperatorRec &op = node.apply;
  int64_t tensor_n = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_n * op.arguments[0].tensor_str.str_n);
  int64_t tensor_c = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_c * op.arguments[0].tensor_str.str_c);
  int64_t tensor_h = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_h * op.arguments[0].tensor_str.str_h);
  int64_t tensor_w = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_w * op.arguments[0].tensor_str.str_w);

  std::vector<double> cost_op;

  if (tensor_n < 2 || tensor_n % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    cost_op.push_back(cost_in_);
  }

  if (tensor_c < 2 || tensor_c % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    cost_op.push_back(cost_in_);
  }

  if (tensor_h < 2 || tensor_h % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    cost_op.push_back(cost_in_);
  }

  if (tensor_w < 2 || tensor_w % 2 != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    cost_op.push_back(cost_in_);
  }

  return ChoseStr(cost_op, node.apply.str);
}

// Chose strategy for BatchParallel op
StrategyRec CostBatchParallel::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] > (DOUBLE_MAX - 0.1)) {
    return str;
  }

  switch (min_position) {
    case 0:
      str.inputTensor[0].str_n /= 2.0;
      str.outputTensor.str_n /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 1:
      str.inputTensor[0].str_c /= 2.0;
      str.outputTensor.str_c /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 2:
      str.inputTensor[0].str_h /= 2.0;
      str.outputTensor.str_h /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 3:
      str.inputTensor[0].str_w /= 2.0;
      str.outputTensor.str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: CostBatchParallel failed.";
  }
  return str;
}

// Chose strategy for CostSoftmaxCrossEntropyWithLogits
StrategyRec CostSoftmaxCrossEntropyWithLogits::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] > (DOUBLE_MAX - 0.1)) {
    return str;
  }

  switch (min_position) {
    case 0:
      str.inputTensor[0].str_n /= 2.0;
      str.inputTensor[1].str_n /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 1:
      str.inputTensor[0].str_c /= 2.0;
      str.inputTensor[1].str_c /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 2:
      str.inputTensor[0].str_h /= 2.0;
      str.inputTensor[1].str_h /= 2.0;
      str.outputTensor.str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    case 3:
      str.inputTensor[0].str_w /= 2.0;
      str.inputTensor[1].str_w /= 2.0;
      str.cut_counter += 1;
      str.cost = str.cost + cost_in_;
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: CostSoftmax failed.";
  }
  return str;
}
}  // namespace parallel
}  // namespace mindspore
