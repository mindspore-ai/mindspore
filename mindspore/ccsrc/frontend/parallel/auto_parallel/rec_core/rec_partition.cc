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

#include "frontend/parallel/auto_parallel/rec_core/rec_partition.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "ir/anf.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace parallel {
// Get the target node's weight for sorting.
double GetWeights(const Graph::NodeType &node) {
  const OperatorRec &op = node.apply;

  if (op.op_type == OperatorType::kRecMatMul) {
    // For MatMul
    auto cost_ptr = std::make_shared<CostMatMul>();

    return cost_ptr->GetMaxCostIn(op);
  } else if (op.op_type == OperatorType::kRecConvolution) {
    // For Convolution
    auto cost_ptr = std::make_shared<CostConvolution>();

    return cost_ptr->GetMinCostIn(node);
  } else if (op.op_type == OperatorType::kRecPooling) {
    // For Pooling
    auto cost_ptr = std::make_shared<CostPooling>();

    return cost_ptr->GetMinCostIn();
  } else if (op.op_type == OperatorType::kRecElmWiseOp) {
    // For TensorAdd
    auto cost_ptr = std::make_shared<CostTensorAdd>();

    return cost_ptr->GetMinCostIn();
  } else if (op.op_type == OperatorType::kRecReLU) {
    // For Activation
    auto cost_ptr = std::make_shared<CostCommon>();

    return cost_ptr->GetMinCostIn();
  } else if (op.op_type == OperatorType::kRecReshape) {
    // For Reshape
    auto cost_ptr = std::make_shared<CostReshape>();

    return cost_ptr->GetMinCostIn();
  } else if (op.op_type == OperatorType::kRecBiasAdd) {
    // For BiasAdd
    auto cost_ptr = std::make_shared<CostBiasAdd>();

    return cost_ptr->GetMinCostIn();
  } else if (op.op_type == OperatorType::kRecLog || op.op_type == OperatorType::kRecExp ||
             op.op_type == OperatorType::kRecAdd || op.op_type == OperatorType::kRecSub ||
             op.op_type == OperatorType::kRecMul || op.op_type == OperatorType::kRecDiv ||
             op.op_type == OperatorType::kRecSqueeze || op.op_type == OperatorType::kRecCast) {
    // For element-wise op
    auto cost_ptr = std::make_shared<CostCommon>();

    return cost_ptr->GetMinCostIn();
  } else if (op.op_type == OperatorType::kRecBatchNorm || op.op_type == OperatorType::kRecOneHot ||
             op.op_type == OperatorType::kRecPReLU || op.op_type == OperatorType::kRecUnsortedSegmentOp ||
             op.op_type == OperatorType::kRecSoftmax ||
             op.op_type == OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits ||
             op.op_type == OperatorType::kRecSoftmaxCrossEntropyWithLogits) {
    // For BatchParallel op
    auto cost_ptr = std::make_shared<CostBatchParallel>();

    return cost_ptr->GetMaxCostIn();
  } else if (op.op_type == OperatorType::kRecUnknownType) {
    // For Unknown type
    return 0.0;
  } else {
    MS_LOG(EXCEPTION) << "Failure: GetOperatorWeight failed.";
  }
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
      if (PARTITION_ORDER == PartitionOrder::TopologyOrder) {
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

// Get optimal strategy to partition the target node
StrategyRec PartitionNode(const Graph::NodeType &node,
                          const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                          const std::shared_ptr<Graph> &graph, const bool isTraining) {
  bool enable_conv_chw_partition = false;
  MS_EXCEPTION_IF_NULL(graph);

  if (node.apply.op_type == OperatorType::kRecMatMul) {
    // For MatMul
    auto cost_ptr = std::make_shared<CostMatMul>();

    return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph, isTraining);
  } else if (node.apply.op_type == OperatorType::kRecConvolution) {
    // For Convolution
    auto cost_ptr = std::make_shared<CostConvolution>();

    return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph, enable_conv_chw_partition);
  } else if (node.apply.op_type == OperatorType::kRecPooling) {
    // For Pooling
    auto cost_ptr = std::make_shared<CostPooling>();

    return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
  } else if (node.apply.op_type == OperatorType::kRecElmWiseOp) {
    // For TensorAdd
    auto cost_ptr = std::make_shared<CostTensorAdd>();

    return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
  } else if (node.apply.op_type == OperatorType::kRecReLU) {
    // For Activation
    auto cost_ptr = std::make_shared<CostCommon>();

    return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
  } else if (node.apply.op_type == OperatorType::kRecReshape) {
    // For Reshape
    auto cost_ptr = std::make_shared<CostReshape>();

    return cost_ptr->GetOptimalStr(node);
  } else if (node.apply.op_type == OperatorType::kRecBiasAdd) {
    // For BiasAdd
    auto cost_ptr = std::make_shared<CostBiasAdd>();

    return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
  } else if (node.apply.op_type == OperatorType::kRecLog || node.apply.op_type == OperatorType::kRecExp ||
             node.apply.op_type == OperatorType::kRecAdd || node.apply.op_type == OperatorType::kRecSub ||
             node.apply.op_type == OperatorType::kRecMul || node.apply.op_type == OperatorType::kRecDiv ||
             node.apply.op_type == OperatorType::kRecSqueeze || node.apply.op_type == OperatorType::kRecCast) {
    // For element-wise op
    auto cost_ptr = std::make_shared<CostCommon>();

    return cost_ptr->GetOptimalStr(node, node_name_to_strategy, *graph);
  } else if (node.apply.op_type == OperatorType::kRecBatchNorm || node.apply.op_type == OperatorType::kRecOneHot ||
             node.apply.op_type == OperatorType::kRecPReLU || node.apply.op_type == kRecSoftmax ||
             node.apply.op_type == OperatorType::kRecSparseSoftmaxCrossEntropyWithLogits ||
             node.apply.op_type == kRecUnsortedSegmentOp) {
    // For BatchParallel type
    auto cost_ptr = std::make_shared<CostBatchParallel>();
    return cost_ptr->GetOptimalStr(node);
  } else if (node.apply.op_type == OperatorType::kRecSoftmaxCrossEntropyWithLogits) {
    // For SoftmaxCrossEntropyWithLogits type
    auto cost_ptr = std::make_shared<CostSoftmaxCrossEntropyWithLogits>();
    return cost_ptr->GetOptimalStr(node);
  } else if (node.apply.op_type == OperatorType::kRecUnknownType) {
    // For Unknown type
    StrategyRec default_strategy;
    return default_strategy;
  } else {
    MS_LOG(EXCEPTION) << "Failure: Partition Operator failed.";
  }
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

void PartitionPipelineStages(size_t num_device, double device_memory, const std::shared_ptr<Graph> &graph) {
  if (!ENABLE_PIPE_ALGO) {
    size_t n_stage = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
    size_t n_node = graph->nodes.size();
    size_t roll_back = log2(n_stage);

    MS_LOG(INFO) << "ROLLING BACK ACCORDING TO STAGE NUMBER (" << n_stage << ") BY " << roll_back << " LEVELS"
                 << std::endl;
    for (size_t i_node = 0; i_node < n_node; ++i_node) {
      Graph::NodeType &node_ptr = graph->nodes[i_node];
      size_t n_cut = GetStratNumber(graph->nodes[i_node]) - roll_back - 1;
      graph->nodes[i_node] = ChangeStrategy(node_ptr, n_cut);
    }
  }
}

// Partition graph into all devices.
Status PartitionForAllDevices(const size_t num_device, const double device_memory, const std::shared_ptr<Graph> &graph,
                              const bool isTraining) {
  if (num_device < 1) {
    MS_LOG(EXCEPTION) << "ERROR: Number of devices can't be " << num_device << ".";
  }

  if (num_device > 1024) {
    MS_LOG(EXCEPTION) << "ERROR: Number of devices can't be larger than 1024.";
  }

  MS_EXCEPTION_IF_NULL(graph);

  // Comopute iter times
  int64_t iter_times = static_cast<int64_t>(log2(num_device));
  if (iter_times > 10) {
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

    // Loop for all the nodes
    for (size_t i_node = 0; i_node < iter_nodes; i_node++) {
      // get current node's index
      size_t index = reorder_node_list[i_node];

      Graph::NodeType &node_ptr = graph->nodes[index];

      // 2-parts partitioning StrategyRec of the last loop
      StrategyRec old_str = graph->nodes[index].apply.str;

      MS_LOG(INFO) << "------------Node_name: " << graph->nodes[index].name << " -------------";

      // Serch optimal strategy to cut this operator. And store the result optimal strategy in graph.
      graph->nodes[index].apply.str = PartitionNode(node_ptr, node_name_to_strategy, graph, isTraining);

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

  // Partition stages
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    PartitionPipelineStages(num_device, device_memory, graph);
  }

  if (DevicesMemoryControl(num_device, device_memory, graph) != SUCCESS) {
    return FAILED;
  }
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
  for (int64_t i = 0; i < 2; i++) {
    Node.apply.arguments[i].tensor_str.str_n = Node.apply.str.inputTensor[i].str_n;
    Node.apply.arguments[i].tensor_str.str_c = Node.apply.str.inputTensor[i].str_c;
    Node.apply.arguments[i].tensor_str.str_h = Node.apply.str.inputTensor[i].str_h;
    Node.apply.arguments[i].tensor_str.str_w = Node.apply.str.inputTensor[i].str_w;
  }
  return Node;
}

Status DevicesMemoryControl(const size_t num_device, const double device_memory, const std::shared_ptr<Graph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (num_device == 0) {
    MS_LOG(EXCEPTION) << "Failure: device number is 0.";
  }

  uint64_t iter_nodes = graph->nodes.size();
  double used_memory = 0.0;

  for (uint64_t i_node = 0; i_node < iter_nodes; i_node++) {
    if (graph->nodes[i_node].info == InfoType::kApplication) {
      Graph::NodeType &Node = graph->nodes[i_node];
      for (int64_t index = 0; index < 2; index++) {
        used_memory += Node.apply.arguments[index].tensor_str.str_n * Node.apply.arguments[index].tensor_shape.shape_n *
                       Node.apply.arguments[index].tensor_str.str_c * Node.apply.arguments[index].tensor_shape.shape_c *
                       Node.apply.arguments[index].tensor_str.str_h * Node.apply.arguments[index].tensor_shape.shape_h *
                       Node.apply.arguments[index].tensor_str.str_w * Node.apply.arguments[index].tensor_shape.shape_w *
                       GetDataTypeSize(Node.apply.arguments[index].tensor_type);
      }
    }
  }

  if (device_memory < (used_memory / num_device)) {
    MS_LOG(EXCEPTION) << "Failure: Out of memory!";
  } else {
    return SUCCESS;
  }
}

size_t GetDataTypeSize(const TensorType &type) {
  switch (type) {
    case kInt8:
      return sizeof(int64_t);
    case kFloat16:
      return sizeof(float) / 2;
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
