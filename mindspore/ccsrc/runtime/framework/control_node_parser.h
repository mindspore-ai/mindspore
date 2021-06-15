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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_PARSER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_PARSER_H_

#include <vector>
#include <string>
#include <memory>
#include <set>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include "runtime/hardware/device_context.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelGraph;
using mindspore::session::KernelWithIndex;

constexpr int kInvalidBranchID = -1;
constexpr int kMainBranchID = 0;
constexpr int kSubBranchStartID = 1;

using FrontToBackendNodeWithContext = std::unordered_map<AnfNodePtr, std::pair<AnfNodePtr, DeviceContext *>>;
using FuncGraphToParameter = std::unordered_map<FuncGraphPtr, std::vector<std::vector<AnfNodePtr>>>;
using HostParameterToWeight = std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>>;
using NodeWithDeviceContext = std::vector<std::pair<AnfNodePtr, DeviceContext *>>;

// Check whether node is a call node, there are two types of call nodes:
// 1. First input of node is a cnode.
// 2. First input of node is a funcgraph value node.
bool IsCallNode(const AnfNodePtr &node);

FuncGraphPtr FetchFuncGraphByNode(const AnfNodePtr &node);

// Get front node by backend node.
AnfNodePtr GetFrontNodeByBackendNode(const AnfNodePtr &backend_node);

// Get the funcgraph to which the node belongs.
FuncGraphPtr GetFuncgraphByBackendNode(const AnfNodePtr &backend_node);

// Find all funcgraphs that the call node will call.
std::vector<FuncGraphPtr> FetchFuncGraphbyCallNode(const CNodePtr &node);

// Fetch all backend input nodes by parameter for gather actor.
std::vector<AnfNodePtr> FetchInputNodeByParameter(const AnfNodePtr &parameter,
                                                  const std::vector<AnfNodePtr> &host_ds_parameters,
                                                  std::vector<AnfNodePtr> *invalid_inputs,
                                                  const FuncGraphToParameter &graph_to_real_parameters);

// ControlNodeParser is used to parse control nodes, and get the edges between nodes.
class ControlNodeParser {
 public:
  // Parse the control node and put the results of the parsing into member variables.
  void Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
             const std::vector<DeviceContext *> &device_contexts, const FuncGraphPtr &root_graph);

  std::vector<AnfNodePtr> GetControlNodeParameter() { return control_node_parameters_; }

  // Get the output of funcgraph, usually there is only one output node, In the control flow, there are
  // multiple branch outputs, there will be multiple output nodes.
  std::vector<AnfNodePtr> FetchAllBranchOutputs(const FuncGraphPtr &func_graph);

 private:
  friend class GraphScheduler;

  // Collect all front value nodes. In the control flow, when the input of the switch actor is the value node, these
  // value nodes will not enter the kernel graph, so these nodes need to be saved separately, and space is allocated for
  // them separately during initialization.
  // The interface is initialized by finding the backend node in the kernel graph that the front node finally sends to.
  void FetchFrontValueNode(const std::vector<KernelGraphPtr> &graphs,
                           const std::vector<DeviceContext *> &device_contexts);

  // Find all value nodes in the switch recursively.
  void FetchValueNodeInSwitchNode(const AnfNodePtr &switch_node, std::vector<AnfNodePtr> *value_nodes);

  // Fetch all the relationships between front parameters and backend parameters.The front parameters
  // include two parts:
  // 1. The parameter from kernel graph.
  // 2. The parameter from control nodes.
  void FetchFrontToBackendParameterMap(const std::vector<KernelGraphPtr> &graphs,
                                       const std::vector<DeviceContext *> &device_contexts,
                                       const std::vector<AnfNodePtr> &control_nodes);

  // Get inputs of control node which come from the host actor. These inputs generally come from the partial
  // nodes and call nodes of the root funcgraph.
  std::vector<AnfNodePtr> FetchControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes);

  // Get all the input parameters of funcgraph. The call of funcgraph is realized through the call node,
  // and the input of the call node is the input parameter of the corresponding funcgraph.
  void FetchFuncGraphToParameterMap(const std::vector<AnfNodePtr> &control_nodes);

  // Get all the front weight parameters related to the weight in the host parameter.
  void FetchHostParameterToWeightMap(const std::vector<AnfNodePtr> &control_nodes);

  // Get the pos input of call node to funcgraph.
  AnfNodePtr GetCallNodeInputByPos(const AnfNodePtr &call_node, const FuncGraphPtr &func_graph, const size_t pos);

  // Find the output of the funcgraph, if the output is a call node, return the output of the funcgraph
  // called by the call node.
  std::vector<AnfNodePtr> FetchFuncGraphOutput(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *call_nodes);

  // Find the corresponding backend parameter for the front_node. If the front_node does not have the corresponding
  // backend parameter, then recursively find the backend parameters of other front parameters corresponding to the
  // front_node.
  std::pair<AnfNodePtr, DeviceContext *> FetchBackendNodeByFrontNode(
    const AnfNodePtr &front_node,
    const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &front_to_front_parameter,
    const std::unordered_map<AnfNodePtr, std::pair<AnfNodePtr, DeviceContext *>> &front_to_backend_parameter,
    std::set<AnfNodePtr> *invalid_node);

  // The relationship between front parameters indicates that the parameter is directly used as the input of the
  // funcgraph. There are two situations:
  // 1. The parameter is used as the input of the call node,
  // 2. The parameter is used as the input of the partial and will be input to the funcgraph of the partial in the
  //    subsequent call node.
  void FetchFrontToFrontParameterMap(const std::vector<AnfNodePtr> &control_nodes,
                                     std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> *front_to_front_parameter);

  // The front to backend parameters is used to build and link the host data source actor in the control flow scenario.
  FrontToBackendNodeWithContext front_to_backend_parameters_;
  // The funcgraph to parameters map records the input parameters of funcgraph and is used to initialize
  // the input node of gather.
  FuncGraphToParameter func_graph_to_parameters_;
  // host parameter to weights records the weights in the subgraph corresponding to the node in the root funcgraph.
  // When initializing the weights, all related weights need to be recorded as the same device tensor.
  HostParameterToWeight host_parameter_to_weights_;
  // The front value node saves all value nodes that are not in the kernel graph. These nodes are generally the
  // input of the control node.
  NodeWithDeviceContext front_value_nodes_;
  // The front output_node is used to link the output actor in multi-branch output scenario.
  std::vector<AnfNodePtr> front_output_nodes_;
  // Parameters of control node which come from the host actor.
  std::vector<AnfNodePtr> control_node_parameters_;
};

using ControlNodeParserPtr = std::shared_ptr<ControlNodeParser>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_PARSER_H_
