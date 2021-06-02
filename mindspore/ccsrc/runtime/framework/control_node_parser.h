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
// ControlNodeParser is a series of tool functions used to parse control nodes.
class ControlNodeParser {
 public:
  // Check whether node is a call node, there are two types of call nodes:
  // 1. First input of node is a cnode.
  // 2. First input of node is a funcgraph value node.
  static bool IsCallNode(const AnfNodePtr &node);

  // Fetch all the relationships between front parameters and backend parameters.The front parameters
  // include two parts:
  // 1. The parameter from kernel graph.
  // 2. The parameter from control nodes.
  static void FetchFrontToBackendParameterMap(const std::vector<KernelGraphPtr> &graphs,
                                              const std::vector<DeviceContext *> &device_contexts,
                                              const std::vector<AnfNodePtr> &control_nodes,
                                              FrontToBackendNodeWithContext *front_to_backend_parameter);

  // Get inputs of control node which come from the host actor. These inputs generally come from the partial
  // nodes and call nodes of the root funcgraph.
  static std::vector<AnfNodePtr> FetchControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes);

  // Get the output of funcgraph, usually there is only one output node, In the control flow, there are
  // multiple branch outputs, there will be multiple output nodes.
  static std::vector<AnfNodePtr> FetchAllBranchOutputs(const FuncGraphPtr &func_graph);

  // Find all funcgraphs that the call node will call.
  static std::vector<FuncGraphPtr> FetchFuncGraphbyCallNode(const CNodePtr &node);

  // Get the funcgraph to which the node belongs.
  static FuncGraphPtr GetFuncgraphByBackendNode(const AnfNodePtr &backend_node);
  static FuncGraphPtr FetchFuncGraphByNode(const AnfNodePtr &node);

  // Get front node by backend node.
  static AnfNodePtr GetFrontNodeByBackendNode(const AnfNodePtr &backend_node);
  // Get the funcgraph in partial node.
  static FuncGraphPtr GetFuncGraphFromPartial(const AnfNodePtr &node);

  // Get all the input parameters of funcgraph. The call of funcgraph is realized through the call node,
  // and the input of the call node is the input parameter of the corresponding funcgraph.
  static void FetchFuncGraphToParameterMap(const std::vector<AnfNodePtr> &control_nodes,
                                           FuncGraphToParameter *graph_to_real_parameters);

  // Get all the front weight parameters related to the weight in the host parameter.
  static void FetchHostParameterToWeightMap(const std::vector<AnfNodePtr> &control_nodes,
                                            HostParameterToWeight *host_parameter_to_weights);

  // Fetch all backend input nodes by parameter for gather actor.
  static std::vector<AnfNodePtr> FetchInputNodeByParameter(const AnfNodePtr &parameter,
                                                           const std::vector<AnfNodePtr> &host_ds_parameters,
                                                           std::vector<AnfNodePtr> *invalid_inputs,
                                                           const FuncGraphToParameter &graph_to_real_parameters);

 private:
  // Get the pos input of call node to funcgraph.
  AnfNodePtr GetCallNodeInputByPos(const AnfNodePtr &call_node, const FuncGraphPtr &func_graph, const size_t pos);

  // Find the output of the funcgraph, if the output is a call node, return the output of the funcgraph
  // called by the call node.
  static std::vector<AnfNodePtr> FetchFuncGraphOutput(const FuncGraphPtr &func_graph,
                                                      std::vector<AnfNodePtr> *call_nodes);

  // Find the corresponding backend parameter for the front_node. If the front_node does not have the corresponding
  // backend parameter, then recursively find the backend parameters of other front parameters corresponding to the
  // front_node.
  static std::pair<AnfNodePtr, DeviceContext *> FetchBackendNodeByFrontNode(
    const AnfNodePtr &front_node,
    const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &front_to_front_parameter,
    const std::unordered_map<AnfNodePtr, std::pair<AnfNodePtr, DeviceContext *>> &front_to_backend_parameter,
    std::set<AnfNodePtr> *invalid_node);

  // The relationship between front parameters indicates that the parameter is directly used as the input of the
  // funcgraph. There are two situations:
  // 1. The parameter is used as the input of the call node,
  // 2. The parameter is used as the input of the partial and will be input to the funcgraph of the partial in the
  //    subsequent call node.
  static void FetchFrontToFrontParameterMap(
    const std::vector<AnfNodePtr> &control_nodes,
    std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> *front_to_front_parameter);
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_PARSER_H_
