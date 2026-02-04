/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/auto_parallel/rec_core/rec_partition.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <unordered_map>

#include "frontend/parallel/status.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/auto_parallel/stage_compute.h"
#include "include/utils/parallel_context.h"

namespace mindspore {
namespace parallel {

double GetWeights(const Graph::NodeType &node) {
  const OperatorRec &op = node.apply;

  auto func_matmul = [&op]() {
    auto cost_ptr = std::make_shared<CostMatMul>();
    return cost_ptr->GetMaxCostIn(op);
  };
  auto func_batch_matmul = [&node]() {
    auto cost_ptr = std::make_shared<CostBatchMatMul>();
    return cost_ptr->GetMaxCostIn(node);
  };
  auto func_convolution = [&node]() {
    auto cost_ptr = std::make_shared<CostConvolution>();
    return cost_ptr->GetMinCostIn(node);
  };
  auto func_pooling = []() {
    auto cost_ptr = std::make_shared<CostPooling>();
    return cost_ptr->GetMinCostIn();
  };
  auto func_tensor_add = []() {
    auto cost_ptr = std::make_shared<CostTensorAdd>();
    return cost_ptr->GetMinCostIn();
  };
  auto func_common = []() {
    auto cost_ptr = std::make_shared<CostCommon>();
    return cost_ptr->GetMinCostIn();
  };
  auto func_reshape = []() {
    auto cost_ptr = std::make_shared<CostReshape>();
    return cost_ptr->GetMinCostIn();
  };
  auto func_bias_add = []() {
    auto cost_ptr = std::make_shared<CostBiasAdd>();
    return cost_ptr->GetMinCostIn();
  };
  auto func_batch_parallel = []() {
    auto cost_ptr = std::make_shared<CostBatchParallel>();
    return cost_ptr->GetMaxCostIn();
  };
  auto func_zero = []() { return 0.0; };

  static const std::map<OperatorType, std::function<double()>> operator_map = {
    {OperatorType::kRecMatMul, func_matmul},
    {OperatorType::kRecBatchMatMul, func_batch_matmul},
    {OperatorType::kRecConvolution, func_convolution},
    {OperatorType::kRecPooling, func_pooling},
    {OperatorType::kRecElmWiseOp, func_tensor_add},
    {OperatorType::kRecReLU, func_common},
    {OperatorType::kRecLog, func_common},
    {OperatorType::kRecExp, func_common},
    {OperatorType::kRecAdd, func_common},
    {OperatorType::kRecSub, func_common},
    {OperatorType::kRecMul, func_common},
    {OperatorType::kRecDiv, func_common},
    {OperatorType::kRecSqueeze, func_common},
    {OperatorType::kRecCast, func_common},
    {OperatorType::kRecReshape, func_reshape},
    {OperatorType::kRecBiasAdd, func_bias_add},
    {OperatorType::kRecBatchNorm, func_batch_parallel},
    {OperatorType::kRecOneHot, func_batch_parallel},
    {OperatorType::kRecPReLU, func_batch_parallel},
    {OperatorType::kRecUnsortedSegmentOp, func_batch_parallel},
    {OperatorType::kRecSoftmax, func_batch_parallel},
    {OperatorType::kRecBatchParallel, func_batch_parallel},
    {OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits, func_batch_parallel},
    {OperatorType::kRecSoftmaxCrossEntropyWithLogits, func_batch_parallel},
    {OperatorType::kRecUnknownType, func_zero},
    {OperatorType::kRecVirtual, func_zero},
    {OperatorType::kRecStandAlone, func_zero}};

  auto it = operator_map.find(op.op_type);
  if (it != operator_map.end()) {
    return it->second();
  }
  MS_LOG(EXCEPTION) << "Failure: GetOperatorWeight failed.";
  return 0.0;
}

// Sort all the nodes by their weights
std::vector<size_t> SortByWeight(const std::shared_ptr<Graph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  std::vector<std::pair<double, size_t>> weight_to_node_index;
  std::vector<size_t> node_index_by_weights;

  // Get node's weight.
  for (size_t pos = 0; pos < graph->nodes.size(); pos++) {
    if (graph->nodes[pos].info == kApplication) {
      const Graph::NodeType &node_ptr = graph->nodes[pos];
      double weight;
      bool mem_first = false;
      if (g_device_manager->DeviceNum() > SIZE_THIRTY_TWO && graph->micro_batch_size < INT64_EIGHT) {
        mem_first = true;
      }
      if (PARTITION_ORDER == PartitionOrder::TopologyOrder && !mem_first) {
        weight = (node_ptr.apply.op_type == OperatorType::kRecUnknownType) ? DOUBLE_LOWEST : pos;
      } else {
        weight = GetWeights(node_ptr);
      }
      size_t index = pos;
      weight_to_node_index.push_back(std::make_pair(weight, index));
    }
  }

  // Ordering ops aka nodes of the graph
  std::sort(weight_to_node_index.begin(), weight_to_node_index.end());

  // Store the result in node_index_by_weights.
  uint64_t size = weight_to_node_index.size();
  for (uint64_t i = 1; i <= size; i++) {
    node_index_by_weights.push_back(weight_to_node_index[size - i].second);
  }

  return node_index_by_weights;
}

bool IsBatchMatMlSameInputs(Graph::NodeType *node) {
  bool same_inputs = false;
  for (size_t idx = 0; idx < node->node_in.size(); idx++) {
    if (idx == node->node_in.size() - 1) {
      break;
    }
    for (size_t idx_bis = idx + 1; idx_bis < node->node_in.size(); idx_bis++) {
      if (node->node_in[idx] == node->node_in[idx_bis]) {
        same_inputs = true;
        break;
      }
    }
    if (same_inputs) {
      break;
    }
  }
  return same_inputs;
}

bool HandleDynamicShapeFix(Graph::NodeType *node, const std::shared_ptr<Graph> &graph) {
  MS_EXCEPTION_IF_NULL(node);
  if (!graph->dyn_shape_tmp_fix) {
    return false;
  }

  const auto &param_name = node->param_name;

  if (node->apply.op_type == OperatorType::kRecMatMul) {
    if (param_name.find(".projection.weight") != std::string::npos) {
      node->apply.str.inputTensor[0].str_w /= SIZE_TWO;
      node->apply.str.inputTensor[1].str_h /= SIZE_TWO;
      return true;
    }
    if (param_name.find(".mapping.weight") != std::string::npos ||
        param_name.find(".attention.dense2.weight") != std::string::npos ||
        param_name.find(".attention_norm.weight") != std::string::npos) {
      node->apply.str.inputTensor[1].str_w /= SIZE_TWO;
      node->apply.str.outputTensor.str_w /= SIZE_TWO;
      return true;
    }
    if (param_name.find(".norm_out.weight") != std::string::npos) {
      return true;
    }
  } else if (node->apply.op_type == OperatorType::kRecBatchMatMul) {
    if (param_name.find(".projection.weight") != std::string::npos) {
      node->apply.str.inputTensor[0].str_w /= SIZE_TWO;
      node->apply.str.inputTensor[1].str_h /= SIZE_TWO;
      return true;
    }
    if (param_name.find(".mapping.weight") != std::string::npos) {
      node->apply.str.inputTensor[1].str_w /= SIZE_TWO;
      node->apply.str.outputTensor.str_w /= SIZE_TWO;
      return true;
    }

    bool same_inputs = IsBatchMatMlSameInputs(node);
    if (same_inputs) {
      return true;
    }

    bool projection_bias_bmm = false;
    bool mapping_bias_bmm = false;
    for (size_t idx = 0; idx < node->node_in.size(); idx++) {
      auto incoming_node_idx = node->node_in[idx];
      if (graph->nodes[incoming_node_idx].param_name.find(".projection.bias") != std::string::npos) {
        projection_bias_bmm = true;
        break;
      }
      if (graph->nodes[incoming_node_idx].param_name.find(".mapping.bias") != std::string::npos) {
        mapping_bias_bmm = true;
        break;
      }
    }
    if (projection_bias_bmm) {
      node->apply.str.inputTensor[0].str_w /= SIZE_TWO;
      node->apply.str.inputTensor[1].str_h /= SIZE_TWO;
      return true;
    }
    if (mapping_bias_bmm) {
      node->apply.str.inputTensor[1].str_w /= SIZE_TWO;
      node->apply.str.outputTensor.str_w /= SIZE_TWO;
      return true;
    }
  }

  return false;
}

std::function<StrategyRec(Graph::NodeType, const std::vector<std::pair<std::string, StrategyRec>> &,
                          const std::shared_ptr<Graph> &, bool, int64_t, bool &)>
  HandleCostCommon = [](Graph::NodeType node,
                        const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                        const std::shared_ptr<Graph> &graph, bool, int64_t, bool &) {
    auto cost_ptr = std::make_shared<CostCommon>();
    return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
  };

std::function<StrategyRec(Graph::NodeType, const std::vector<std::pair<std::string, StrategyRec>> &,
                          const std::shared_ptr<Graph> &, bool, int64_t, bool &)>
  HandleCostBatchParallel = [](Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &,
                               const std::shared_ptr<Graph> &, bool, int64_t, bool &) {
    auto cost_ptr = std::make_shared<CostBatchParallel>();
    return cost_ptr->GetOptimalStr(node);
  };

static const std::unordered_map<
  OperatorType, std::function<StrategyRec(Graph::NodeType, const std::vector<std::pair<std::string, StrategyRec>> &,
                                          const std::shared_ptr<Graph> &, bool, int64_t, bool &)>>
  operation_map = {
    {OperatorType::kRecMatMul,
     [](Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
        const std::shared_ptr<Graph> &graph, bool isTraining, int64_t loop, bool &) {
       auto cost_ptr = std::make_shared<CostMatMul>();
       return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph, isTraining, loop);
     }},
    {OperatorType::kRecBatchMatMul,
     [](Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
        const std::shared_ptr<Graph> &graph, bool isTraining, int64_t, bool &) {
       auto cost_ptr = std::make_shared<CostBatchMatMul>();
       return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph, isTraining);
     }},
    {OperatorType::kRecConvolution,
     [](Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
        const std::shared_ptr<Graph> &graph, bool, int64_t, bool &enable_conv_chw_partition) {
       auto cost_ptr = std::make_shared<CostConvolution>();
       return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph, enable_conv_chw_partition);
     }},
    {OperatorType::kRecPooling,
     [](Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
        const std::shared_ptr<Graph> &graph, bool, int64_t, bool &) {
       auto cost_ptr = std::make_shared<CostPooling>();
       return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
     }},
    {OperatorType::kRecElmWiseOp,
     [](Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
        const std::shared_ptr<Graph> &graph, bool, int64_t, bool &) {
       auto cost_ptr = std::make_shared<CostTensorAdd>();
       return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
     }},
    {OperatorType::kRecReshape,
     [](Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &, const std::shared_ptr<Graph> &,
        bool, int64_t, bool &) {
       auto cost_ptr = std::make_shared<CostReshape>();
       return cost_ptr->GetOptimalStr(node);
     }},
    {OperatorType::kRecBiasAdd,
     [](Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
        const std::shared_ptr<Graph> &graph, bool, int64_t, bool &) {
       auto cost_ptr = std::make_shared<CostBiasAdd>();
       return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
     }},
    {OperatorType::kRecSoftmaxCrossEntropyWithLogits,
     [](Graph::NodeType node, const std::vector<std::pair<std::string, StrategyRec>> &, const std::shared_ptr<Graph> &,
        bool, int64_t, bool &) {
       auto cost_ptr = std::make_shared<CostSoftmaxCrossEntropyWithLogits>();
       return cost_ptr->GetOptimalStr(node);
     }},
    {OperatorType::kRecLog, HandleCostCommon},
    {OperatorType::kRecExp, HandleCostCommon},
    {OperatorType::kRecAdd, HandleCostCommon},
    {OperatorType::kRecSub, HandleCostCommon},
    {OperatorType::kRecMul, HandleCostCommon},
    {OperatorType::kRecDiv, HandleCostCommon},
    {OperatorType::kRecReLU, HandleCostCommon},
    {OperatorType::kRecCast, HandleCostCommon},
    {OperatorType::kRecSqueeze, HandleCostCommon},
    {OperatorType::kRecPReLU, HandleCostBatchParallel},
    {OperatorType::kRecOneHot, HandleCostBatchParallel},
    {OperatorType::kRecSoftmax, HandleCostBatchParallel},
    {OperatorType::kRecVirtual, HandleCostBatchParallel},
    {OperatorType::kRecBatchNorm, HandleCostBatchParallel},
    {OperatorType::kRecBatchParallel, HandleCostBatchParallel},
    {OperatorType::kRecUnsortedSegmentOp, HandleCostBatchParallel},
    {OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits, HandleCostBatchParallel}};

// Get optimal strategy to partition the target node
StrategyRec PartitionNode(Graph::NodeType node,
                          const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                          const std::shared_ptr<Graph> &graph, const bool isTraining, const int64_t loop) {
  MS_EXCEPTION_IF_NULL(graph);
  if (HandleDynamicShapeFix(&node, graph)) {
    return node.apply.str;
  }

  auto it = operation_map.find(node.apply.op_type);
  if (it != operation_map.end()) {
    bool enable_conv_chw_partition = false;
    return it->second(node, node_name_to_strategy, graph, isTraining, loop, enable_conv_chw_partition);
  }

  if (node.apply.op_type == OperatorType::kRecUnknownType || node.apply.op_type == OperatorType::kRecStandAlone) {
    return StrategyRec();
  }

  MS_LOG(EXCEPTION) << "Failure: OperatorType not matched for " << node.name;
}

StrategyRec GetOneLoopStrategy(size_t op_inputs_num, const StrategyRec &old_str, StrategyRec new_str) {
  for (size_t i = 0; i < op_inputs_num; i++) {
    if (abs(old_str.inputTensor[i].str_n) > EPS && abs(old_str.inputTensor[i].str_c) > EPS &&
        abs(old_str.inputTensor[i].str_h) > EPS && abs(old_str.inputTensor[i].str_w) > EPS) {
      new_str.inputTensor[i].str_n = new_str.inputTensor[i].str_n / old_str.inputTensor[i].str_n;
      new_str.inputTensor[i].str_c = new_str.inputTensor[i].str_c / old_str.inputTensor[i].str_c;
      new_str.inputTensor[i].str_h = new_str.inputTensor[i].str_h / old_str.inputTensor[i].str_h;
      new_str.inputTensor[i].str_w = new_str.inputTensor[i].str_w / old_str.inputTensor[i].str_w;
    }
  }

  if (old_str.outputTensor.str_n > EPS && old_str.outputTensor.str_c > EPS && old_str.outputTensor.str_h > EPS &&
      old_str.outputTensor.str_w > EPS) {
    new_str.outputTensor.str_n = new_str.outputTensor.str_n / old_str.outputTensor.str_n;
    new_str.outputTensor.str_c = new_str.outputTensor.str_c / old_str.outputTensor.str_c;
    new_str.outputTensor.str_h = new_str.outputTensor.str_h / old_str.outputTensor.str_h;
    new_str.outputTensor.str_w = new_str.outputTensor.str_w / old_str.outputTensor.str_w;
  }

  return new_str;
}

Graph::NodeType ChangeStrategy(Graph::NodeType Node, size_t n_cut) {
  if (n_cut >= Node.apply.strs.size()) {
    MS_LOG(EXCEPTION) << "Strategy not available";
  }
  Node.apply.str = Node.apply.strs[n_cut];
  Node = ApplyStrToTensor(Node);

  return Node;
}

size_t GetStratNumber(const Graph::NodeType &Node) { return Node.apply.strs.size(); }

void PartitionPipelineStages(double device_memory, const std::shared_ptr<Graph> &graph) {
  if (!ENABLE_PIPE_ALGO) {
    size_t n_stage = LongToSize(parallel::ParallelContext::GetInstance()->pipeline_stage_split_num());
    size_t n_node = graph->nodes.size();
    size_t roll_back = FloatToSize(log2(n_stage));

    MS_LOG(INFO) << "ROLLING BACK ACCORDING TO STAGE NUMBER (" << n_stage << ") BY " << roll_back << " LEVELS"
                 << std::endl;
    for (size_t i_node = 0; i_node < n_node; ++i_node) {
      Graph::NodeType &node_ptr = graph->nodes[i_node];
      if (node_ptr.interfered_sapp) {
        MS_LOG(INFO) << "Skip partition pipeline stages. Found a user-defined strategy: " << node_ptr.name << std::endl;
        continue;
      }
      size_t n_cut = GetStratNumber(graph->nodes[i_node]) - roll_back - 1;
      graph->nodes[i_node] = ChangeStrategy(node_ptr, n_cut);
    }
  }
}

TensorStr4D DecomposeStrategy(const TensorStr4D &tensor_str) {
  TensorStr4D result;
  std::vector<float> strategy = {
    static_cast<float>(1.0f / tensor_str.str_n), static_cast<float>(1.0f / tensor_str.str_c),
    static_cast<float>(1.0f / tensor_str.str_h), static_cast<float>(1.0f / tensor_str.str_w)};

  for (size_t index = 0; index < strategy.size(); ++index) {
    while (strategy[index] > 1) {
      std::vector<float> tmp_vector(SIZE_FOUR, 1);
      tmp_vector[index] /= INT64_TWO;
      strategy[index] /= INT64_TWO;

      TensorStr4D tmp_strategy;
      tmp_strategy.str_n = tmp_vector[INDEX_ZERO];
      tmp_strategy.str_c = tmp_vector[INDEX_ONE];
      tmp_strategy.str_h = tmp_vector[INDEX_TWO];
      tmp_strategy.str_w = tmp_vector[INDEX_THREE];

      result.decomposed_custom_strategy.push_back(tmp_strategy);
    }
  }

  return result;
}

// Partition graph into all devices.
Status PartitionForAllDevices(size_t num_device, double device_memory, const std::shared_ptr<Graph> &graph,
                              bool isTraining, const FuncGraphPtr &root) {
  if (num_device < 1) {
    MS_LOG(EXCEPTION) << "ERROR: Number of devices can't be " << num_device << ".";
  }

  if (num_device > MAX_NUM_DEVICES) {
    MS_LOG(EXCEPTION) << "ERROR: Number of devices can't be larger than <<" << MAX_NUM_DEVICES << " .";
  }

  MS_EXCEPTION_IF_NULL(graph);

  // Comopute iter times
  int64_t iter_times = static_cast<int64_t>(log2(num_device));
  if (iter_times > INT64_TEN) {
    MS_LOG(EXCEPTION) << "ERROR: Number of iter_times can't be larger than 10.";
  }

  // N-cuts loop
  for (int64_t loop = 0; loop < iter_times; loop++) {
    // Sort by weights
    std::vector<size_t> reorder_node_list = SortByWeight(graph);

    // get total node number
    size_t iter_nodes = reorder_node_list.size();

    // temp vector to map nodename to its strategy.
    std::vector<std::pair<std::string, StrategyRec>> node_name_to_strategy;
    for (const auto &node : graph->nodes) {
      if (node.interfered_sapp) {
        StrategyRec str;
        // Splits the custom input strategy into strategy that SAPP understands
        for (size_t i = 0; i < node.node_in.size(); ++i) {
          str.inputTensor[i] = DecomposeStrategy(node.apply.arguments[i].tensor_str);
        }
        // Splits the custom output strategy into strategy that SAPP understands
        str.outputTensor = DecomposeStrategy(node.tensor_parm.tensor_str);
        node_name_to_strategy.emplace_back(node.name, str);
      }
    }

    // Loop for all the nodes
    for (size_t i_node = 0; i_node < iter_nodes; i_node++) {
      // get current node's index
      size_t index = reorder_node_list[i_node];

      Graph::NodeType &node_ptr = graph->nodes[index];
      if (graph->nodes[index].interfered_sapp) {
        MS_LOG(INFO) << "Skip partition for all devices. Found an operator with user-defined strategy: "
                     << node_ptr.name;
        continue;
      }

      // 2-parts partitioning StrategyRec of the last loop
      StrategyRec old_str = graph->nodes[index].apply.str;

      // Save first strategy too
      if (graph->nodes[index].apply.strs.size() == 0) {
        graph->nodes[index].apply.strs.push_back(old_str);
      }

      MS_LOG(INFO) << "------------Node_name: " << graph->nodes[index].name << " -------------";

      // Search optimal strategy to cut this operator. And store the result optimal strategy in graph.
      graph->nodes[index].apply.str = PartitionNode(node_ptr, node_name_to_strategy, graph, isTraining, loop);
      graph->nodes[index].apply.strs.push_back(graph->nodes[index].apply.str);

      // Get Current 2-parts partitioning strategy of this loop
      size_t op_inputs_num = graph->nodes[index].node_in.size();
      StrategyRec one_loop_strategyrec = GetOneLoopStrategy(op_inputs_num, old_str, graph->nodes[index].apply.str);

      // Apply OP Strategy to Tensor Strategy.
      graph->nodes[index] = ApplyStrToTensor(node_ptr);

      // Note down the node name and its strategy in this loop.
      auto node_name_to_str = std::pair<std::string, StrategyRec>(graph->nodes[index].name, one_loop_strategyrec);
      node_name_to_strategy.push_back(node_name_to_str);
    }
  }

  // Auto pipeline
  size_t new_stage_num = ParallelSuggestion(root, graph);
  if (parallel::ParallelContext::GetInstance()->auto_pipeline()) {
    ChangeStageNumber(root, new_stage_num);
  }

  // Partition stages
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    PartitionPipelineStages(device_memory, graph);
  }

  DevicesMemoryControl(num_device, device_memory, graph);
  return SUCCESS;
}

// Apply OP Strategy to Tensor Strategy
Graph::NodeType ApplyStrToTensor(Graph::NodeType Node) {
  // Set Node's tensor_parm
  Node.tensor_parm.tensor_str.str_n = Node.apply.str.outputTensor.str_n;
  Node.tensor_parm.tensor_str.str_c = Node.apply.str.outputTensor.str_c;
  Node.tensor_parm.tensor_str.str_h = Node.apply.str.outputTensor.str_h;
  Node.tensor_parm.tensor_str.str_w = Node.apply.str.outputTensor.str_w;

  // Set input tensors' tersor_parm
  for (int64_t i = 0; i < INT64_TWO; i++) {
    Node.apply.arguments[i].tensor_str.str_n = Node.apply.str.inputTensor[i].str_n;
    Node.apply.arguments[i].tensor_str.str_c = Node.apply.str.inputTensor[i].str_c;
    Node.apply.arguments[i].tensor_str.str_h = Node.apply.str.inputTensor[i].str_h;
    Node.apply.arguments[i].tensor_str.str_w = Node.apply.str.inputTensor[i].str_w;
  }
  return Node;
}

void DevicesMemoryControl(const size_t num_device, const double device_memory, const std::shared_ptr<Graph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (num_device == 0) {
    MS_LOG(EXCEPTION) << "Failure: device number is 0.";
  }

  uint64_t iter_nodes = graph->nodes.size();
  double used_memory = 0.0;

  for (uint64_t i_node = 0; i_node < iter_nodes; i_node++) {
    if (graph->nodes[i_node].info == InfoType::kApplication) {
      Graph::NodeType &Node = graph->nodes[i_node];
      if (Node.interfered_sapp) {
        MS_LOG(INFO) << "Skip device mem control. Found an operator with user-defined strategy: " << Node.name;
        continue;
      }
      for (int64_t index = 0; index < INT64_TWO; index++) {
        used_memory += Node.apply.arguments[index].tensor_str.str_n * Node.apply.arguments[index].tensor_shape.shape_n *
                       Node.apply.arguments[index].tensor_str.str_c * Node.apply.arguments[index].tensor_shape.shape_c *
                       Node.apply.arguments[index].tensor_str.str_h * Node.apply.arguments[index].tensor_shape.shape_h *
                       Node.apply.arguments[index].tensor_str.str_w * Node.apply.arguments[index].tensor_shape.shape_w *
                       GetDataTypeSize(Node.apply.arguments[index].tensor_type);
      }
    }
  }

  if (device_memory < (used_memory / num_device)) {
    MS_LOG(WARNING) << "It is estimated that the task may collapse due to out of memory!";
  }
}

size_t GetDataTypeSize(const TensorType &type) {
  switch (type) {
    case kInt8:
      return sizeof(int64_t);
    case kFloat16:
      return sizeof(float) / SIZE_TWO;
    case kFloat32:
      return sizeof(float);
    case kDouble64:
      return sizeof(double);
    default:
      MS_LOG(EXCEPTION) << "GetDataTypeSize Failed. Unexpected type";
  }
}
}  // namespace parallel
}  // namespace mindspore
