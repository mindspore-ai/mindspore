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
#include <set>
#include <map>

#include "include/utils/parallel_context.h"

namespace mindspore {
namespace parallel {
bool SameShape(const Shape4D &shape1, const Shape4D &shape2) {
  bool equal = (shape1 == shape2);

  return (equal || !ONLY_REDIST_WITH_SAME_SHAPE);
}

double costOfDistributing(const TensorParam &t) {
  return (static_cast<double>(t.tensor_shape.shape_n) * t.tensor_str.str_n *
          static_cast<double>(t.tensor_shape.shape_c) * t.tensor_str.str_c *
          static_cast<double>(t.tensor_shape.shape_h) * t.tensor_str.str_h *
          static_cast<double>(t.tensor_shape.shape_w) * t.tensor_str.str_w / FACTOR_TWO);
}

double minNodeSize(const Graph::NodeType &node) {
  double distributing0 = costOfDistributing(node.apply.arguments[0]);
  double distributing1 = costOfDistributing(node.apply.arguments[1]);
  double distributing2 = costOfDistributing(node.tensor_parm);
  double min_distribution = std::min(distributing0, distributing1);
  min_distribution = std::min(min_distribution, distributing2);
  min_distribution *= EXPERT_COEF;
  return min_distribution;
}

Shape4D ShapeConversion(const Shape4D &origin, const std::vector<int64_t> &operations) {
  std::vector<int64_t> shape_vec = origin.ShapeToVector();
  std::vector<int64_t> new_shape_vec(SIZE_FOUR, 0);
  for (size_t i = 0; i < operations.size(); ++i) {
    new_shape_vec[i] = shape_vec[operations[i]];
  }
  Shape4D new_shape = VectorToShape(new_shape_vec);
  return new_shape;
}

bool isCombine(const std::vector<std::vector<int64_t>> &reshape_operation) {
  for (size_t i = 0; i < reshape_operation.size(); ++i) {
    if (reshape_operation[i].size() > 1) {
      return true;
    }
  }
  return false;
}

std::vector<std::vector<int64_t>> RearrangeReshape(const std::vector<int64_t> &transpose_op,
                                                   const std::vector<std::vector<int64_t>> &reshape_op) {
  std::vector<std::vector<int64_t>> new_reshape;
  std::map<size_t, int64_t> swap_map;
  for (size_t i = 0; i < transpose_op.size(); ++i) {
    swap_map[i] = transpose_op[i];
  }

  if (isCombine(reshape_op)) {
    for (auto op : reshape_op) {
      std::vector<int64_t> tmp;
      for (size_t i = 0; i < op.size(); ++i) {
        auto it = swap_map.find(op[i]);
        if (it != swap_map.end()) {
          tmp.push_back(swap_map[op[i]]);
        }
      }
      new_reshape.push_back(tmp);
    }
  } else {
    for (size_t i = 0; i < transpose_op.size(); ++i) {
      if ((size_t)transpose_op[i] < reshape_op.size()) {
        new_reshape.push_back(reshape_op[transpose_op[i]]);
      }
    }
  }
  return new_reshape;
}

bool SameShapeAfterReshape(const Shape4D &origin, const std::vector<std::vector<int64_t>> &reshape_op,
                           const Shape4D &compared) {
  std::vector<int64_t> shape_vec = origin.ShapeToVector();
  std::vector<int64_t> compared_vec = compared.ShapeToVector();

  if (isCombine(reshape_op)) {
    std::vector<int64_t> new_shape_vec;
    for (size_t i = 0; i < reshape_op.size(); ++i) {
      int64_t tmp = 1;
      tmp = std::accumulate(reshape_op[i].begin(), reshape_op[i].end(), tmp,
                            [&](auto accum, auto pos) { return accum * shape_vec[pos]; });
      new_shape_vec.push_back(tmp);
    }
    Shape4D new_shape = VectorToShape(new_shape_vec);
    return SameShape(new_shape, compared);
  } else {
    for (size_t i = 0; i < reshape_op.size(); ++i) {
      if (compared_vec[i] != 0) {
        shape_vec[reshape_op[i][0]] /= compared_vec[i];
      } else {
        MS_LOG(EXCEPTION) << i << "th position of compared shape is zero";
      }
    }
    if (shape_vec[INDEX_ZERO] == 1 && shape_vec[INDEX_ONE] == 1 && shape_vec[INDEX_TWO] == 1 &&
        shape_vec[INDEX_THREE] == 1) {
      return true;
    }
    return false;
  }
}

bool isAdjacent(const Shape4D &incoming_shape, const Shape4D &current_shape, const std::vector<int64_t> &transpose_op,
                const std::vector<std::vector<int64_t>> &reshape_op) {
  if (!transpose_op.empty() && !reshape_op.empty()) {
    std::vector<std::vector<int64_t>> rearrange_reshape_op = RearrangeReshape(transpose_op, reshape_op);
    return SameShapeAfterReshape(incoming_shape, rearrange_reshape_op, current_shape);
  }

  if (!transpose_op.empty() && reshape_op.empty()) {
    return SameShape(ShapeConversion(incoming_shape, transpose_op), current_shape);
  }

  if (!reshape_op.empty() && transpose_op.empty()) {
    return SameShapeAfterReshape(incoming_shape, reshape_op, current_shape);
  }

  return SameShape(incoming_shape, current_shape);
}

double CalculateCostIfAdjacent(bool adjacent,
                               const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                               const std::vector<std::vector<float>> &mode, size_t i_strategy, size_t i_node,
                               double tensor_value, bool is_search_forward, int64_t loop) {
  if (adjacent) {
    return CostRedisWithAdjacentNode(node_name_to_strategy, mode, i_strategy, i_node, tensor_value, is_search_forward,
                                     loop);
  }

  return COST_MODEL_NO_COST;
}

// Compute redistributed cost
double CostRedis(const Graph::NodeType &node,
                 const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                 const std::vector<std::vector<float>> &mode, const Graph &graph, int64_t loop) {
  // Store value of cost redist
  double cost_redis = 0;

  // Set tensor edge value with original tensor shape and cutting times.
  double input_tensor = node.apply.arguments[0].tensor_shape.shape_n * node.apply.arguments[0].tensor_str.str_n *
                        node.apply.arguments[0].tensor_shape.shape_c * node.apply.arguments[0].tensor_str.str_c *
                        node.apply.arguments[0].tensor_shape.shape_h * node.apply.arguments[0].tensor_str.str_h *
                        node.apply.arguments[0].tensor_shape.shape_w * node.apply.arguments[0].tensor_str.str_w;

  double output_tensor = node.tensor_parm.tensor_shape.shape_n * node.tensor_parm.tensor_str.str_n *
                         node.tensor_parm.tensor_shape.shape_c * node.tensor_parm.tensor_str.str_c *
                         node.tensor_parm.tensor_shape.shape_h * node.tensor_parm.tensor_str.str_h *
                         node.tensor_parm.tensor_shape.shape_w * node.tensor_parm.tensor_str.str_w;

  // For each strategy candidate
  for (size_t i_strategy = 0; i_strategy < node_name_to_strategy.size(); ++i_strategy) {
    // Process forward nodes
    for (size_t i_node = 0; i_node < node.node_in.size(); ++i_node) {
      if (graph.nodes[node.node_in[i_node]].name == node_name_to_strategy[i_strategy].first) {
        bool adjacent = false;
        for (auto out_node : graph.nodes[node.node_in[i_node]].node_out) {
          if (graph.nodes[out_node.idx].name.compare(node.name) == 0) {
            std::vector<int64_t> transpose_combined =
              TransposeCombine(graph.nodes[node.node_in[i_node]].transpose_mapping, out_node.transpose_mapping);
            std::vector<std::vector<int64_t>> reshape_combined =
              ReshapeCombine(graph.nodes[node.node_in[i_node]].reshape_mapping, out_node.reshape_mapping);
            adjacent = isAdjacent(graph.nodes[node.node_in[i_node]].tensor_parm.tensor_shape,
                                  node.apply.arguments[i_node].tensor_shape, transpose_combined, reshape_combined);
            MS_LOG(INFO) << "Node " << node.name << " and Node " << graph.nodes[node.node_in[i_node]].name
                         << " adjacency: " << adjacent;

            bool is_search_forward = true;
            cost_redis += CalculateCostIfAdjacent(adjacent, node_name_to_strategy, mode, i_strategy, i_node,
                                                  input_tensor, is_search_forward, loop);
          }
        }
      }
    }

    // Process backward nodes
    for (size_t i_node = 0; i_node < node.node_out.size(); ++i_node) {
      if (graph.nodes[node.node_out[i_node].idx].name == node_name_to_strategy[i_strategy].first) {
        for (size_t i = 0; i < MAX_INPUT_NUM; ++i) {
          bool adjacent = false;
          std::vector<int64_t> transpose_combined =
            TransposeCombine(node.transpose_mapping, node.node_out[i_node].transpose_mapping);
          std::vector<std::vector<int64_t>> reshape_combined =
            ReshapeCombine(node.reshape_mapping, node.node_out[i_node].reshape_mapping);
          adjacent = isAdjacent(node.tensor_parm.tensor_shape,
                                graph.nodes[node.node_out[i_node].idx].apply.arguments[i].tensor_shape,
                                transpose_combined, reshape_combined);
          MS_LOG(INFO) << "Node " << node.name << " and Node " << graph.nodes[node.node_out[i_node].idx].name
                       << " adjacency: " << adjacent;

          bool is_search_forward = false;
          cost_redis += CalculateCostIfAdjacent(adjacent, node_name_to_strategy, mode, i_strategy, i_node,
                                                output_tensor, is_search_forward, loop);
          if (adjacent) {
            break;
          }
        }
      }
    }

    // Process auxiliary forward nodes
    for (size_t i_node = 0; i_node < node.node_in_aux.size(); ++i_node) {
      size_t index = node.node_in_aux_idx[i_node];
      if (graph.nodes[node.node_in_aux[i_node]].name == node_name_to_strategy[i_strategy].first) {
        for (auto out_node : graph.nodes[node.node_in_aux[i_node]].node_out) {
          if (graph.nodes[out_node.idx].name.compare(node.name) == 0) {
            bool adjacent = false;
            std::vector<int64_t> transpose_combined =
              TransposeCombine(graph.nodes[node.node_in_aux[i_node]].transpose_mapping, out_node.transpose_mapping);
            std::vector<std::vector<int64_t>> reshape_combined =
              ReshapeCombine(graph.nodes[node.node_in_aux[i_node]].reshape_mapping, out_node.reshape_mapping);
            adjacent = isAdjacent(graph.nodes[node.node_in_aux[i_node]].tensor_parm.tensor_shape,
                                  node.apply.arguments[index].tensor_shape, transpose_combined, reshape_combined);
            MS_LOG(INFO) << "Node " << node.name << " and Node " << graph.nodes[node.node_in_aux[i_node]].name
                         << " adjacency: " << adjacent;

            bool is_search_forward = true;
            cost_redis += CalculateCostIfAdjacent(adjacent, node_name_to_strategy, mode, i_strategy, index,
                                                  input_tensor, is_search_forward, loop);
          }
        }
      }
    }
  }

  return cost_redis;
}

double CostRedisWithAdjacentNode(const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                 const std::vector<std::vector<float>> &mode, size_t i_strategy, size_t i_node,
                                 double tensor_size, bool search_forward, int64_t loop) {
  double new_redis_cost = 0;
  bool diff = false;
  auto output_tensor = node_name_to_strategy[i_strategy].second.outputTensor;
  auto input_tensor = node_name_to_strategy[i_strategy].second.inputTensor[0];

  if (search_forward) {
    float output_dims[NDIMS] = {output_tensor.str_n, output_tensor.str_c, output_tensor.str_h, output_tensor.str_w};
    if (!output_tensor.decomposed_custom_strategy.empty() &&
        output_tensor.decomposed_custom_strategy.size() > static_cast<size_t>(loop)) {
      output_dims[INDEX_ZERO] = output_tensor.decomposed_custom_strategy[loop].str_n;
      output_dims[INDEX_ONE] = output_tensor.decomposed_custom_strategy[loop].str_c;
      output_dims[INDEX_TWO] = output_tensor.decomposed_custom_strategy[loop].str_h;
      output_dims[INDEX_THREE] = output_tensor.decomposed_custom_strategy[loop].str_w;
    }
    for (size_t i = 0; i < NDIMS; ++i) {
      if (output_dims[i] == 0 || mode[i_node][i] == 0) {
        MS_LOG(EXCEPTION) << "divisors cannot be 0!";
      }
      if (static_cast<int64_t>(1 / output_dims[i]) != static_cast<int64_t>(1 / mode[i_node][i])) {
        diff = true;
        break;
      }
    }
  } else {
    float input_dims[NDIMS] = {input_tensor.str_n, input_tensor.str_c, input_tensor.str_h, input_tensor.str_w};
    if (!input_tensor.decomposed_custom_strategy.empty() &&
        input_tensor.decomposed_custom_strategy.size() > static_cast<size_t>(loop)) {
      input_dims[INDEX_ZERO] = input_tensor.decomposed_custom_strategy[loop].str_n;
      input_dims[INDEX_ONE] = input_tensor.decomposed_custom_strategy[loop].str_c;
      input_dims[INDEX_TWO] = input_tensor.decomposed_custom_strategy[loop].str_h;
      input_dims[INDEX_THREE] = input_tensor.decomposed_custom_strategy[loop].str_w;
    }
    for (size_t i = 0; i < NDIMS; ++i) {
      if (input_dims[i] == 0 || mode[INDEX_TWO][i] == 0) {
        MS_LOG(EXCEPTION) << "divisors cannot be 0!";
      }
      if (static_cast<int64_t>(1 / input_dims[i]) != static_cast<int64_t>(1 / mode[INDEX_TWO][i])) {
        diff = true;
        break;
      }
    }
  }

  if (diff) {
    new_redis_cost = tensor_size * REDIS_COEF;
  }

  return new_redis_cost;
}

bool hasBeenSplitted(const Graph::NodeType &node, const bool dyn_shape_tmp_fix) {
  if (dyn_shape_tmp_fix) {
    if (node.apply.arguments[0].tensor_str.str_h < 1 || node.apply.arguments[0].tensor_str.str_w < 1 ||
        node.apply.arguments[1].tensor_str.str_w < 1 || node.apply.arguments[0].tensor_str.str_n < 1 ||
        node.apply.arguments[0].tensor_str.str_c < 1) {
      return true;
    }
  }
  return false;
}

// Get optimal strategy for MatMul
StrategyRec CostMatMul::GetOptimalStr(const Graph::NodeType &node,
                                      const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                      const Graph &graph, const bool isTraining, int64_t loop) {
  int64_t edge_i =
    static_cast<int64_t>(node.apply.arguments[0].tensor_shape.shape_h * node.apply.arguments[0].tensor_str.str_h);
  int64_t edge_j =
    static_cast<int64_t>(node.apply.arguments[1].tensor_shape.shape_w * node.apply.arguments[1].tensor_str.str_w);
  int64_t edge_k =
    static_cast<int64_t>(node.apply.arguments[0].tensor_shape.shape_w * node.apply.arguments[0].tensor_str.str_w);

  bool isMicroBatchSizeLargeEnough = true;
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    if (graph.micro_batch_size * node.apply.arguments[0].tensor_str.str_h <= 1) {
      isMicroBatchSizeLargeEnough = false;
    }
  }

  std::vector<double> cost_op;
  if (node.apply.arguments[0].tensor_str.str_h == 0) {
    MS_LOG(EXCEPTION) << "str_h cannot be 0!";
  }
  if (edge_i < INT64_TWO || edge_i % INT64_TWO != 0 || !isMicroBatchSizeLargeEnough) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 0.5, 1}, {1, 1, 1, 1}, {1, 1, 0.5, 1}};
    double cost_if_cut_i = StrConcatDimI(edge_j, edge_k);
    double redist_if_cut_i = CostRedis(node, node_name_to_strategy, mode, graph, loop);
    double total_cost_if_cut_i = cost_if_cut_i + redist_if_cut_i;
    MS_LOG(INFO) << "If the I-axis is cut, the op-cost is " << cost_if_cut_i << ", the redist-cost is "
                 << redist_if_cut_i << ", and the total cost is " << total_cost_if_cut_i;
    cost_op.push_back(total_cost_if_cut_i);
  }

  // Do not partition the J-axis and K-axis for the same MatMul
  if (edge_j < INT64_TWO || edge_j % INT64_TWO != 0 || node.apply.arguments[0].tensor_str.str_w < 1) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 1, 1}, {1, 1, 1, 0.5}, {1, 1, 1, 0.5}};
    double cost_if_cut_j = StrConcatDimJ(edge_i, edge_k);
    double redist_if_cut_j = CostRedis(node, node_name_to_strategy, mode, graph, loop);
    double total_cost_if_cut_j = cost_if_cut_j + redist_if_cut_j;
    MS_LOG(INFO) << "If the J-axis is cut, the op-cost is " << cost_if_cut_j << ", the redist-cost is "
                 << redist_if_cut_j << ", and the total cost is " << total_cost_if_cut_j;
    cost_op.push_back(total_cost_if_cut_j);
  }

  if (edge_k < INT64_TWO || edge_k % INT64_TWO != 0 || node.apply.arguments[1].tensor_str.str_w < 1) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 1, 0.5}, {1, 1, 0.5, 1}, {1, 1, 1, 1}};
    double cost_if_cut_k = StrReduceDimK(edge_i, edge_j);
    double redist_if_cut_k = CostRedis(node, node_name_to_strategy, mode, graph, loop);
    double total_cost_if_cut_k = cost_if_cut_k + redist_if_cut_k;
    MS_LOG(INFO) << "If the K-axis is cut, the op-cost is " << cost_if_cut_k << ", the redist-cost is "
                 << redist_if_cut_k << ", and the total cost is " << total_cost_if_cut_k;
    cost_op.push_back(total_cost_if_cut_k);
  }

  if (hasBeenSplitted(node, graph.dyn_shape_tmp_fix)) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};
    double cost_if_no_cut =
      StrRecom(StrConcatDimI(edge_j, edge_k), StrConcatDimJ(edge_i, edge_k), StrReduceDimK(edge_i, edge_j));
    double redist_if_no_cut = CostRedis(node, node_name_to_strategy, mode, graph, loop);
    double total_cost_if_no_cut = cost_if_no_cut + redist_if_no_cut;
    MS_LOG(INFO) << "If do NOT cut the axis, the op-cost is " << cost_if_no_cut << ", the redist-cost is "
                 << redist_if_no_cut << ", and the total cost is " << total_cost_if_no_cut;
    cost_op.push_back(total_cost_if_no_cut);
  }

  std::for_each(cost_op.begin(), cost_op.end(), [](double &cost) { cost = std::abs(cost); });

  return ChoseStr(cost_op, node.apply.str);
}

// Get weight for MatMul
double CostMatMul::GetMaxCostIn(const OperatorRec &op) {
  int64_t edge_i = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_h * op.arguments[0].tensor_str.str_h);
  int64_t edge_j = static_cast<int64_t>(op.arguments[1].tensor_shape.shape_w * op.arguments[1].tensor_str.str_w);
  int64_t edge_k = static_cast<int64_t>(op.arguments[0].tensor_shape.shape_w * op.arguments[0].tensor_str.str_w);

  double cost_if_cut_i = StrConcatDimI(edge_j, edge_k);
  double cost_if_cut_j = StrConcatDimJ(edge_i, edge_k);
  double cost_if_cut_k = StrReduceDimK(edge_i, edge_j);
  double cost_if_no_cut = StrRecom(cost_if_cut_i, cost_if_cut_j, cost_if_cut_k);

  std::vector<double> cost_in;
  cost_in.push_back(cost_if_cut_i);
  cost_in.push_back(cost_if_cut_j);
  cost_in.push_back(cost_if_cut_k);
  cost_in.push_back(cost_if_no_cut);

  return *max_element(cost_in.begin(), cost_in.end());
}

void updateTensors(const std::vector<float *> &input_values, float *output_value, StrategyRec *str,
                   const double cost_in) {
  MS_EXCEPTION_IF_NULL(str);
  for (auto input_value : input_values) {
    MS_EXCEPTION_IF_NULL(input_value);
    *input_value /= FACTOR_TWO;
  }
  if (output_value) {
    *output_value /= FACTOR_TWO;
  }
  str->cut_counter += 1;
  str->cost += cost_in;
}

// Chose strategy for MatMul
StrategyRec CostMatMul::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) const {
  MS_LOG(INFO) << "The costs of cutting the I-axis/J-axis/K-axis/no_cut are : " << cost_op;
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] == DOUBLE_MAX) {
    return str;
  }

  std::vector<float *> input_values;
  switch (min_position) {
    case 0:
      input_values = {&str.inputTensor[0].str_h};
      updateTensors(input_values, &str.outputTensor.str_h, &str, cost_in_i_);
      MS_LOG(INFO) << "The I-axis is chosen to cut";
      break;

    case 1:
      input_values = {&str.inputTensor[1].str_w};
      updateTensors(input_values, &str.outputTensor.str_w, &str, cost_in_j_);
      MS_LOG(INFO) << "The J-axis is chosen to cut";
      break;

    case INDEX_TWO:
      input_values = {&str.inputTensor[0].str_w, &str.inputTensor[1].str_h};
      updateTensors(input_values, nullptr, &str, cost_in_k_);
      MS_LOG(INFO) << "The K-axis is chosen to cut";
      break;

    case INDEX_THREE:
      MS_LOG(INFO) << "Choose NOT to cut";
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure:CostMatMul failed.";
  }

  return str;
}

size_t CostBatchMatMul::getBatchDimsSize(const OperatorRec &op) {
  return static_cast<double>(std::max(op.arguments[0].tensor_shape.shape_n, op.arguments[1].tensor_shape.shape_n)) *
         std::max(op.arguments[0].tensor_str.str_n, op.arguments[1].tensor_str.str_n) *
         static_cast<double>(std::max(op.arguments[0].tensor_shape.shape_c, op.arguments[1].tensor_shape.shape_c)) *
         std::max(op.arguments[0].tensor_str.str_c, op.arguments[1].tensor_str.str_c);
}

double CostBatchMatMul::cost(Axis a, const Graph::NodeType &node) {
  double mc_ratio;
  size_t batch_dims_size = getBatchDimsSize(node.apply);
  if (batch_dims_size == 1) {
    mc_ratio = static_cast<double>(NUMBER_ASCEND_CORES);
  } else {
    mc_ratio = std::max(NUMBER_ASCEND_CORES / static_cast<double>(batch_dims_size) - 1, COST_MODEL_NO_COST);
  }
  double min_size = minNodeSize(node);

  switch (a) {
    // Calculate the cost if the Batch-axis of BatchMatMul is cut
    case B:
      return (mc_ratio * min_size);

    // Calculate the cost if the Expert-axis of BatchMatMul is cut
    case X:
      return (mc_ratio * min_size) - 1;

    // Calculate the cost if the I-axis of BatchMatMul is cut
    case I:
      return costOfDistributing(node.apply.arguments[1]);

    // Calculate the cost if the J-axis of BatchMatMul is cut
    case J:
      return costOfDistributing(node.apply.arguments[0]);

    // Calculate the cost if the K-axis of BatchMatMul is cut
    case K:
      return costOfDistributing(node.tensor_parm);

    // Calculate the cost if BatchMatMul is not cut
    case R:
      return min_size * min_size / REPLICATE_BELOW;

    default:
      MS_LOG(EXCEPTION) << "Axis " << a << " is not taken into account";
  }

  return 1;
}

bool SplitOnlyOneDimension(const Graph &graph, float str) {
  if (graph.dyn_shape_tmp_fix && str < 1) {
    return true;
  }
  return false;
}

bool IsEdgeSplittable(const int64_t edge) {
  if (edge < INT64_TWO || edge % INT64_TWO != 0) {
    return false;
  }
  return true;
}

const char *CostBatchMatMul::AxisToString(CostBatchMatMul::Axis axis) {
  static const std::map<Axis, const char *> axis_to_string = {{B, "B"}, {X, "X"}, {I, "I"},
                                                              {J, "J"}, {K, "K"}, {R, "R"}};

  auto it = axis_to_string.find(axis);
  if (it != axis_to_string.end()) {
    return it->second;
  }
  return "Unknown";
}

void CostBatchMatMul::ComputeAndLogCost(double *cost_op, const std::vector<std::vector<float>> &mode,
                                        const Graph::NodeType &node,
                                        const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                        const Graph &graph, const CostBatchMatMul::Axis axis_name, const size_t coef) {
  double op_cost = cost(axis_name, node);
  double redist_cost = CostRedis(node, node_name_to_strategy, mode, graph);
  double total_cost = (op_cost + redist_cost) / coef;
  MS_LOG(INFO) << "If the " << AxisToString(axis_name) << "-axis is cut, the op-cost is " << op_cost
               << ", the redist-cost is " << redist_cost << ", and the total cost is " << total_cost;
  *cost_op = std::abs(total_cost);
}

// Get optimal strategy for BatchMatMul
StrategyRec CostBatchMatMul::GetOptimalStr(
  const Graph::NodeType &node, const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
  const Graph &graph, const bool isTraining) {
  int64_t edge_b =
    static_cast<int64_t>(node.apply.arguments[0].tensor_shape.shape_n * node.apply.arguments[0].tensor_str.str_n);
  int64_t edge_x =
    static_cast<int64_t>(node.apply.arguments[0].tensor_shape.shape_c * node.apply.arguments[0].tensor_str.str_c);
  int64_t edge_i =
    static_cast<int64_t>(node.apply.arguments[0].tensor_shape.shape_h * node.apply.arguments[0].tensor_str.str_h);
  int64_t edge_j =
    static_cast<int64_t>(node.apply.arguments[1].tensor_shape.shape_w * node.apply.arguments[1].tensor_str.str_w);
  int64_t edge_k =
    static_cast<int64_t>(node.apply.arguments[0].tensor_shape.shape_w * node.apply.arguments[0].tensor_str.str_w);

  bool isMicroBatchSizeLargeEnough = true;
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    if (graph.micro_batch_size * node.apply.arguments[0].tensor_str.str_n <= 1) {
      isMicroBatchSizeLargeEnough = false;
    }
  }

  if (node.apply.arguments[0].tensor_str.str_n == 0) {
    MS_LOG(EXCEPTION) << "str_n cannot be 0!";
  }

  std::vector<double> cost_op(INDEX_SIX, DOUBLE_MAX);
  if (IsEdgeSplittable(edge_b) && isMicroBatchSizeLargeEnough) {
    ComputeAndLogCost(&cost_op[INDEX_ZERO], {{0.5, 1, 1, 1}, {0.5, 1, 1, 1}, {0.5, 1, 1, 1}}, node,
                      node_name_to_strategy, graph, B);
  }
  if (IsEdgeSplittable(edge_x)) {
    ComputeAndLogCost(&cost_op[INDEX_ONE], {{1, 0.5, 1, 1}, {1, 0.5, 1, 1}, {1, 0.5, 1, 1}}, node,
                      node_name_to_strategy, graph, X);
  }
  if (IsEdgeSplittable(edge_i) && !SplitOnlyOneDimension(graph, node.apply.arguments[0].tensor_str.str_c)) {
    ComputeAndLogCost(&cost_op[INDEX_TWO], {{1, 1, 0.5, 1}, {1, 1, 1, 1}, {1, 1, 0.5, 1}}, node, node_name_to_strategy,
                      graph, I);
  }
  if (IsEdgeSplittable(edge_j) && node.apply.arguments[0].tensor_str.str_w >= 1 &&
      !SplitOnlyOneDimension(graph, node.apply.arguments[0].tensor_str.str_c)) {
    ComputeAndLogCost(&cost_op[INDEX_THREE], {{1, 1, 1, 1}, {1, 1, 1, 0.5}, {1, 1, 1, 0.5}}, node,
                      node_name_to_strategy, graph, J, BMM_COEF);
  }
  if (IsEdgeSplittable(edge_k) && node.apply.arguments[1].tensor_str.str_w >= 1 &&
      !SplitOnlyOneDimension(graph, node.apply.arguments[0].tensor_str.str_c)) {
    ComputeAndLogCost(&cost_op[INDEX_FOUR], {{1, 1, 1, 0.5}, {1, 1, 0.5, 1}, {1, 1, 1, 1}}, node, node_name_to_strategy,
                      graph, K, BMM_COEF);
  }
  if (!hasBeenSplitted(node, graph.dyn_shape_tmp_fix)) {
    ComputeAndLogCost(&cost_op[INDEX_FIVE], {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, node, node_name_to_strategy,
                      graph, R);
  }

  return ChoseStr(cost_op, node.apply.str);
}

// Get weight for BatchMatMul
double CostBatchMatMul::GetMaxCostIn(const Graph::NodeType &node) {
  std::vector<double> cost_in;
  cost_in.push_back(cost(B, node));
  cost_in.push_back(cost(X, node));
  cost_in.push_back(cost(I, node));
  cost_in.push_back(cost(J, node));
  cost_in.push_back(cost(K, node));
  cost_in.push_back(cost(R, node));

  return *max_element(cost_in.begin(), cost_in.end());
}

// Chose strategy for BatchMatMul
StrategyRec CostBatchMatMul::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) const {
  MS_LOG(INFO) << "The costs of cutting the Batch-axis/Expert-axis/I-axis/J-axis/K-axis/no_cut are : " << cost_op;
  int64_t min_position = min_element(cost_op.begin(), cost_op.end()) - cost_op.begin();
  if (cost_op[min_position] == DOUBLE_MAX) {
    return str;
  }

  str.cut_counter += 1;
  str.cost = str.cost + cost_op[min_position];

  std::vector<float *> input_values;
  switch (min_position) {
    case 0:
      input_values = {&str.inputTensor[0].str_n, &str.inputTensor[1].str_n};
      updateTensors(input_values, &str.outputTensor.str_n, &str, COST_MODEL_NO_COST);
      MS_LOG(INFO) << "The Batch-axis is chosen to cut";
      break;

    case 1:
      input_values = {&str.inputTensor[0].str_c, &str.inputTensor[1].str_c};
      updateTensors(input_values, &str.outputTensor.str_c, &str, COST_MODEL_NO_COST);
      MS_LOG(INFO) << "The Expert-axis is chosen to cut";
      break;

    case INDEX_TWO:
      input_values = {&str.inputTensor[0].str_h};
      updateTensors(input_values, &str.outputTensor.str_h, &str, COST_MODEL_NO_COST);
      MS_LOG(INFO) << "The I-axis is chosen to cut";
      break;

    case INDEX_THREE:
      input_values = {&str.inputTensor[1].str_w};
      updateTensors(input_values, &str.outputTensor.str_w, &str, COST_MODEL_NO_COST);
      MS_LOG(INFO) << "The J-axis is chosen to cut";
      break;

    case INDEX_FOUR:
      input_values = {&str.inputTensor[0].str_w, &str.inputTensor[1].str_h};
      updateTensors(input_values, nullptr, &str, COST_MODEL_NO_COST);
      MS_LOG(INFO) << "The K-axis is chosen to cut";
      break;

    case INDEX_FIVE:
      MS_LOG(INFO) << "Choose NOT to cut";
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure:CostBatchMatMul failed.";
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
  cost_op.reserve(INDEX_SEVEN);

  if (input_tensor_n < INT64_TWO || input_tensor_n % INT64_TWO != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{0.5, 1, 1, 1}, {1, 1, 1, 1}, {0.5, 1, 1, 1}};
    cost_op.push_back(StrDimB(tensor_filter) + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  cost_op.push_back(DOUBLE_MAX);
  cost_op.push_back(DOUBLE_MAX);

  if (channel_partition == false || tensor_filter < INT64_TWO || tensor_filter % INT64_TWO != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    std::vector<std::vector<float>> mode = {{1, 1, 1, 1}, {0.5, 1, 1, 1}, {1, 0.5, 1, 1}};
    cost_op.push_back(StrDimK(tensor_in) + CostRedis(node, node_name_to_strategy, mode, graph));
  }

  cost_op.push_back(DOUBLE_MAX);
  cost_op.push_back(DOUBLE_MAX);

  if (channel_partition == false || tensor_filter_c < INT64_TWO || tensor_filter_c % INT64_TWO != 0) {
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
  if (cost_op[min_position] == DOUBLE_MAX) {
    return str;
  }

  std::vector<float *> input_values;
  switch (min_position) {
    case 0:
      input_values = {&str.inputTensor[0].str_n};
      updateTensors(input_values, &str.outputTensor.str_n, &str, cost_in_b_);
      break;

    case 1:
      input_values = {&str.inputTensor[0].str_h};
      updateTensors(input_values, &str.outputTensor.str_h, &str, cost_in_i_);
      break;

    case INDEX_TWO:
      input_values = {&str.inputTensor[0].str_w};
      updateTensors(input_values, &str.outputTensor.str_w, &str, cost_in_j_);
      break;

    case INDEX_THREE:
      input_values = {&str.inputTensor[1].str_n};
      updateTensors(input_values, &str.outputTensor.str_c, &str, cost_in_k_);
      break;

    case INDEX_FOUR:
      input_values = {&str.inputTensor[1].str_h};
      updateTensors(input_values, nullptr, &str, cost_in_di_);
      break;

    case INDEX_FIVE:
      input_values = {&str.inputTensor[1].str_w};
      updateTensors(input_values, nullptr, &str, cost_in_dj_);
      break;

    case INDEX_SIX:
      input_values = {&str.inputTensor[0].str_c, &str.inputTensor[1].str_c};
      updateTensors(input_values, nullptr, &str, cost_in_q_);
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: CostConvolution failed.";
  }
  return str;
}

void EvaluateAndPushCost(int64_t tensor_value, const std::vector<std::vector<float>> &mode,
                         std::vector<double> *cost_op, const Graph::NodeType &node,
                         const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                         const Graph &graph, const double cost_in) {
  if (tensor_value < INT64_TWO || tensor_value % INT64_TWO != 0) {
    cost_op->push_back(DOUBLE_MAX);
  } else {
    cost_op->push_back(cost_in + CostRedis(node, node_name_to_strategy, mode, graph));
  }
}

// Get optimal strategy for Pooling
StrategyRec CostPooling::GetOptimalStr(const Graph::NodeType &node,
                                       const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                                       const Graph &graph) const {
  int64_t tensor_n = static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_n * node.tensor_parm.tensor_str.str_n);
  int64_t tensor_c = static_cast<int64_t>(node.tensor_parm.tensor_shape.shape_c * node.tensor_parm.tensor_str.str_c);

  std::vector<double> cost_op;
  EvaluateAndPushCost(tensor_n, {{0.5, 1, 1, 1}, {0.5, 1, 1, 1}, {0.5, 1, 1, 1}}, &cost_op, node, node_name_to_strategy,
                      graph, cost_in_);
  EvaluateAndPushCost(tensor_c, {{1, 0.5, 1, 1}, {1, 0.5, 1, 1}, {1, 0.5, 1, 1}}, &cost_op, node, node_name_to_strategy,
                      graph, cost_in_);
  cost_op.push_back(DOUBLE_MAX);
  cost_op.push_back(DOUBLE_MAX);
  return ChoseStr(cost_op, node.apply.str);
}

void processTensorUpdate(const uint64_t min_position, StrategyRec *str, const double cost_in,
                         const std::string &errorMessage) {
  std::vector<float *> input_values;

  switch (min_position) {
    case 0:
      input_values = {&str->inputTensor[0].str_n};
      updateTensors(input_values, &str->outputTensor.str_n, str, cost_in);
      break;

    case 1:
      input_values = {&str->inputTensor[0].str_c};
      updateTensors(input_values, &str->outputTensor.str_c, str, cost_in);
      break;

    case INDEX_TWO:
      input_values = {&str->inputTensor[0].str_h};
      updateTensors(input_values, &str->outputTensor.str_h, str, cost_in);
      break;

    case INDEX_THREE:
      input_values = {&str->inputTensor[0].str_w};
      updateTensors(input_values, &str->outputTensor.str_w, str, cost_in);
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: " << errorMessage;
      break;
  }
}

// Chose strategy for Pooling
StrategyRec CostPooling::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) const {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] == DOUBLE_MAX) {
    return str;
  }
  processTensorUpdate(min_position, &str, cost_in_, "CostPooling failed.");
  return str;
}

// Chose strategy for Add
StrategyRec CostTensorAdd::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] == DOUBLE_MAX) {
    return str;
  }

  std::vector<float *> input_values;
  switch (min_position) {
    case 0:
      input_values = {&str.inputTensor[0].str_n, &str.inputTensor[1].str_n};
      updateTensors(input_values, &str.outputTensor.str_n, &str, cost_in_);
      break;

    case 1:
      input_values = {&str.inputTensor[0].str_c, &str.inputTensor[1].str_c};
      updateTensors(input_values, &str.outputTensor.str_c, &str, cost_in_);
      break;

    case INDEX_TWO:
      input_values = {&str.inputTensor[0].str_h, &str.inputTensor[1].str_h};
      updateTensors(input_values, &str.outputTensor.str_h, &str, cost_in_);
      break;

    case INDEX_THREE:
      input_values = {&str.inputTensor[0].str_w, &str.inputTensor[1].str_w};
      updateTensors(input_values, &str.outputTensor.str_w, &str, cost_in_);
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
  if (cost_op[min_position] == DOUBLE_MAX) {
    return str;
  }

  std::vector<float *> input_values;
  switch (min_position) {
    case 0:
      input_values = {&str.inputTensor[0].str_n};
      updateTensors(input_values, &str.outputTensor.str_n, &str, cost_in_);
      break;

    case 1:
      input_values = {&str.inputTensor[0].str_c};
      updateTensors(input_values, &str.outputTensor.str_c, &str, cost_in_);
      break;

    case INDEX_TWO:
      input_values = {&str.inputTensor[0].str_h};
      updateTensors(input_values, &str.outputTensor.str_h, &str, cost_in_);
      break;

    case INDEX_THREE:
      input_values = {&str.inputTensor[0].str_w, &str.inputTensor[1].str_w};
      updateTensors(input_values, &str.outputTensor.str_w, &str, cost_in_);
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
  EvaluateAndPushCost(tensor_n, {{0.5, 1, 1, 1}, {0.5, 1, 1, 1}, {0.5, 1, 1, 1}}, &cost_op, node, node_name_to_strategy,
                      graph, cost_in_);
  EvaluateAndPushCost(tensor_c, {{1, 0.5, 1, 1}, {1, 0.5, 1, 1}, {1, 0.5, 1, 1}}, &cost_op, node, node_name_to_strategy,
                      graph, cost_in_);
  EvaluateAndPushCost(tensor_h, {{1, 1, 0.5, 1}, {1, 1, 0.5, 1}, {1, 1, 0.5, 1}}, &cost_op, node, node_name_to_strategy,
                      graph, cost_in_);
  EvaluateAndPushCost(tensor_w, {{1, 1, 1, 0.5}, {1, 1, 1, 0.5}, {1, 1, 1, 0.5}}, &cost_op, node, node_name_to_strategy,
                      graph, cost_in_);
  return ChoseStr(cost_op, node.apply.str);
}

// Chose strategy for Common op
StrategyRec CostCommon::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] == DOUBLE_MAX) {
    return str;
  }
  processTensorUpdate(min_position, &str, cost_in_, "Common failed.");
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

  if (tensor_n < INT64_TWO || tensor_n % INT64_TWO != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    cost_op.push_back(cost_in_);
  }

  if (tensor_c < INT64_TWO || tensor_c % INT64_TWO != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    cost_op.push_back(cost_in_);
  }

  if (tensor_h < INT64_TWO || tensor_h % INT64_TWO != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    cost_op.push_back(cost_in_);
  }

  if (tensor_w < INT64_TWO || tensor_w % INT64_TWO != 0) {
    cost_op.push_back(DOUBLE_MAX);
  } else {
    cost_op.push_back(cost_in_);
  }

  return ChoseStr(cost_op, node.apply.str);
}

// Chose strategy for BatchParallel op
StrategyRec CostBatchParallel::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] == DOUBLE_MAX) {
    return str;
  }
  processTensorUpdate(min_position, &str, cost_in_, "CostBatchParallel failed.");
  return str;
}

// Chose strategy for CostSoftmaxCrossEntropyWithLogits
StrategyRec CostSoftmaxCrossEntropyWithLogits::ChoseStr(const std::vector<double> &cost_op, StrategyRec str) {
  uint64_t min_position = LongToUlong(min_element(cost_op.begin(), cost_op.end()) - cost_op.begin());
  if (cost_op[min_position] == DOUBLE_MAX) {
    return str;
  }

  std::vector<float *> input_values;
  switch (min_position) {
    case 0:
      input_values = {&str.inputTensor[0].str_n, &str.inputTensor[1].str_n};
      updateTensors(input_values, nullptr, &str, cost_in_);
      break;

    case 1:
      input_values = {&str.inputTensor[0].str_c, &str.inputTensor[1].str_c};
      updateTensors(input_values, nullptr, &str, cost_in_);
      break;

    case INDEX_TWO:
      input_values = {&str.inputTensor[0].str_h, &str.inputTensor[1].str_h};
      updateTensors(input_values, &str.outputTensor.str_w, &str, cost_in_);
      break;

    case INDEX_THREE:
      input_values = {&str.inputTensor[0].str_w, &str.inputTensor[1].str_w};
      updateTensors(input_values, nullptr, &str, cost_in_);
      break;

    default:
      MS_LOG(EXCEPTION) << "Failure: CostSoftmax failed.";
  }
  return str;
}
}  // namespace parallel
}  // namespace mindspore
