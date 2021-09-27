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
#include <map>
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
using FrontToBackendKernelWithContext = std::map<KernelWithIndex, std::pair<KernelWithIndex, DeviceContext *>>;
using FuncGraphToParameter = std::unordered_map<FuncGraphPtr, std::vector<std::vector<AnfNodePtr>>>;
using HostParameterToWeight = std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>>;
using NodeWithDeviceContext = std::vector<std::pair<AnfNodePtr, DeviceContext *>>;
using RealToFormalNode = std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>>;

// Check whether node is a call node, there are two types of call nodes:
// 1. First input of node is a cnode.
// 2. First input of node is a funcgraph value node.
bool IsCallNode(const AnfNodePtr &node);

// Check if the call node is the input of another call node.
bool IsSubCallNode(const AnfNodePtr &node);

// Recursive interface, find the real output of funcgraph called by call node.
AnfNodePtr FetchRealOutputByCallNode(const AnfNodePtr &node, std::set<AnfNodePtr> *call_nodes);

// Check whether the parameter is a weight. In the control flow, weight is passed to the subgraph, and in the subgraph,
// it is determined whether it is a weight.
bool HasAbstractRef(const AnfNodePtr &node);

// Recursive interface, get the funcgraph which the node belongs, if the node has a front node, return the funcgraph
// which the front node belongs, if not, find the funcgraph which the input of the node belongs.
FuncGraphPtr FetchFuncGraphByNode(const AnfNodePtr &node);

// Recursive interface, get the number of output nodes of funcgraph called by call node.
size_t FetchOutputSizebyCallNode(const AnfNodePtr &node, std::vector<AnfNodePtr> *call_nodes);

// Get front node by backend node.
AnfNodePtr GetFrontNodeByBackendNode(const AnfNodePtr &backend_node);

// Get the front node corresponding to the backend node, if the front node is not a parameter node, return the
// corresponding cnode.
KernelWithIndex GetFrontNodeByKernelGraph(const AnfNodePtr &backend_node, const KernelGraphPtr &graph);

// Get the funcgraph to which the node belongs.
FuncGraphPtr GetFuncgraphByBackendNode(const AnfNodePtr &backend_node);

// Find all funcgraphs that the call node will call.
std::vector<FuncGraphPtr> FetchFuncGraphbyCallNode(const AnfNodePtr &node);

// Get parameters in kernel graph.
std::vector<KernelWithIndex> FetchParameterbyKernelGraph(const KernelGraphPtr &graph);

// ControlNodeParser is used to parse control nodes, and get the edges between nodes.
class ControlNodeParser {
 public:
  // Parse the control node and put the results of the parsing into member variables.
  void Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
             const std::vector<DeviceContext *> &device_contexts, const FuncGraphPtr &root_graph);

  const std::vector<AnfNodePtr> &control_node_parameters() const { return control_node_parameters_; }
  const FrontToBackendNodeWithContext &front_to_backend_parameters() const { return front_to_backend_parameters_; }
  const HostParameterToWeight &host_parameter_to_weights() const { return host_parameter_to_weights_; }
  const NodeWithDeviceContext &front_value_nodes() const { return front_value_nodes_; }

  // Get the output of funcgraph, usually there is only one output node, In the control flow, there are
  // multiple branch outputs, there will be multiple output nodes.
  std::vector<AnfNodePtr> FetchAllBranchOutputs(const FuncGraphPtr &func_graph);

  // Get all possible input nodes of the output node. When the switch actor is the output, it need to send the node
  // which device address belongs, so switch actor need to get all the possible nodes.
  std::set<KernelWithIndex> FetchBackendInputNodeByFrontNode(const AnfNodePtr &front_output);

  // Get the device context corresponding to the value node.
  DeviceContext *GetFrontValueNodeDeviceContext(const AnfNodePtr &value_node);

  // Get the branch id corresponding to funcgraph.
  int GetBranchIDByFuncGraph(const FuncGraphPtr &func_graph);

  // Get the number of calls to funcgraph
  size_t GetCallNumByFuncGraph(const FuncGraphPtr &func_graph);

  // Get all possible input nodes of the output node. When the gather actor is the output, it need to send the node
  // which device address belongs, so gather actor need to get all the possible nodes.
  std::vector<KernelWithIndex> GetBackendInputByParameter(const AnfNodePtr &parameter);

  // Check whether there is a call node in the front input nodes of the kernel graph.
  bool IsCallInputKernelGraph(const KernelGraphPtr &graph);

  // Check whether the kernel actor belongs to the root graph.
  // In general, all no output nodes belong to the root funcgraph, and the corresponding switch actor for output should
  // be empty. In control flow, the control arrow of the no output node in the sub funcgraph should be sent to the
  // output switch actor.
  bool IsKernelInRootFuncGraph(const AnfNodePtr &kernel);

  // Get the backend node corresponding to the weight node in the subgraph.
  AnfNodePtr FetchBackendNodebyWeightNode(const AnfNodePtr &node);

  KernelWithIndex GetBackendKernelByFrontKernel(const KernelWithIndex &front_node_with_index) {
    return front_to_backend_kernels_[front_node_with_index].first;
  }

  AnfNodePtr FetchRootGraphFrontNodeBySubFrontNode(const AnfNodePtr &sub_front_node);

 private:
  friend class GraphScheduler;

  // Collect all front value nodes. In the control flow, when the input of the switch actor is the value node, these
  // value nodes will not enter the kernel graph, so these nodes need to be saved separately, and space is allocated for
  // them separately during initialization.
  // The interface is initialized by finding the backend node in the kernel graph that the front node finally sends to.
  void FetchFrontValueNode(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
                           const std::vector<DeviceContext *> &device_contexts);
  // Create branch id for all subgraphs in the control flow.
  void CreateBranchIDForFuncGraph(const std::vector<AnfNodePtr> &control_nodes);
  // Find all value nodes in the switch recursively.
  void FetchValueNodeBySwitchNode(const AnfNodePtr &switch_node, std::vector<AnfNodePtr> *value_nodes);
  // Fetch all the relationships between front parameters and backend parameters.The front parameters
  // include two parts:
  // 1. The parameter from kernel graph.
  // 2. The parameter from control nodes.
  void FetchFrontToBackendParameter(const std::vector<KernelGraphPtr> &graphs,
                                    const std::vector<DeviceContext *> &device_contexts,
                                    const RealToFormalNode &real_to_formal_front_parameters,
                                    const RealToFormalNode &formal_to_real_front_parameters);
  // Get the relationship between the front and backend of the executable kernel in all kernel graphs.
  void FetchFrontToBackendKernel(const std::vector<KernelGraphPtr> &graphs,
                                 const std::vector<DeviceContext *> &device_contexts);
  // Get inputs of control node which come from the host actor. These inputs generally come from the partial
  // nodes and call nodes of the root funcgraph.
  std::vector<AnfNodePtr> FetchControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes,
                                                    DeviceContext *device_context);
  // Get all the input parameters of funcgraph. The call of funcgraph is realized through the call node,
  // and the input of the call node is the input parameter of the corresponding funcgraph.
  void FetchFuncGraphToParameter(const std::vector<AnfNodePtr> &control_nodes);
  // Get all the front weight parameters related to the weight in the host parameter.
  void FetchHostParameterToWeight(const RealToFormalNode &real_to_formal_front_parameters);
  // The relationship between front parameters indicates that the parameter is directly used as the input of the
  // funcgraph. There are two situations:
  // 1. The parameter is used as the input of the call node,
  // 2. The parameter is used as the input of the partial and will be input to the funcgraph of the partial in the
  //    subsequent call node.
  void FetchFrontToFrontParameter(const std::vector<AnfNodePtr> &control_nodes,
                                  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> *front_to_front_parameter);
  // Get the number of calls to all subgraphs in the whole funcgraph.
  void FetchFuncGraphCallNum(const std::vector<AnfNodePtr> &control_nodes);
  // Get all the kernel graphs where the input node has a call node.
  void FetchCallInputKernelGraph(const std::vector<KernelGraphPtr> &graphs,
                                 const std::vector<DeviceContext *> &device_contexts);
  // Get the relationship of all real and formal nodes in the whole funcgraph.
  void FetchBackendInputNode(const std::vector<KernelGraphPtr> &graphs,
                             const std::vector<DeviceContext *> &device_contexts,
                             const RealToFormalNode &real_to_formal_front_parameters,
                             const RealToFormalNode &formal_to_real_front_parameters);
  // Get the relationship of all real and formal parameters in the whole funcgraph.
  void FetchBackendParameterNode(const std::vector<KernelGraphPtr> &graphs,
                                 const std::vector<DeviceContext *> &device_contexts,
                                 const RealToFormalNode &real_to_formal_front_parameters,
                                 const RealToFormalNode &formal_to_real_front_parameters,
                                 FrontToBackendNodeWithContext *front_to_backend_parameters);
  // Get all possible input node of real parameter.
  void FetchBackendInputNodebyFrontNode(const AnfNodePtr &real_parameter, const AnfNodePtr &formal_parameter,
                                        const FrontToBackendNodeWithContext &front_to_backend_parameters);
  // Recursive interface, get all Backend node by front_output.
  void FetchBackendOutputByFrontOutput(const AnfNodePtr &front_output, std::set<AnfNodePtr> *call_nodes,
                                       std::set<AnfNodePtr> *switch_nodes, std::set<KernelWithIndex> *results);

  // Get the dependency between kernel and call node in auto monad.
  void FetchAutoMonadNode(const std::vector<AnfNodePtr> &control_nodes);
  // The front to backend parameters is used to build and link the host data source actor in the control flow scenario.
  FrontToBackendNodeWithContext front_to_backend_parameters_;

  // The relationship between all real parameters and formal parameters in the entire func_graph.
  // In control flow, the control actor will be the output actor. Since the actor needs to send the node to the output
  // actor, it is necessary to save all the real parameters corresponding to the formal parameters in the control actor.
  // When the control actor receives the device address, it can find the corresponding input node.
  std::unordered_map<AnfNodePtr, std::vector<KernelWithIndex>> formal_to_real_parameters_;

  // Relationship between the front and backend of the executable kernel in all kernel graphs.
  FrontToBackendKernelWithContext front_to_backend_kernels_;

  // The funcgraph to parameters map records the input parameters of funcgraph and is used to initialize
  // the input node of gather.
  FuncGraphToParameter func_graph_to_parameters_;

  // The relationship between the valuenode inputs of the call node and the backend parameter
  std::map<KernelWithIndex, std::pair<AnfNodePtr, DeviceContext *>> call_node_to_backend_parameters_;

  // Branch id of funcgraph.
  // In control flow, funcgraph will be called in multiple places, and the output of funcgraph needs to return to
  // different places. Therefore, a branch id is created for each funcgraph. When funcgraph is called, the branch
  // id needs to be sent to the gather actor corresponding to the funcgraph, and the gather will send the branch id
  // to its output switch actor.
  std::unordered_map<FuncGraphPtr, int> func_graph_to_branch_id_;

  // host parameter to weights records the weights in the subgraph corresponding to the node in the root funcgraph.
  // When initializing the weights, all related weights need to be recorded as the same device tensor.
  HostParameterToWeight host_parameter_to_weights_;
  std::unordered_map<AnfNodePtr, AnfNodePtr> sub_front_node_to_root_front_node_;

  // The front value node saves all value nodes that are not in the kernel graph. These nodes are generally the
  // input of the control node.
  NodeWithDeviceContext front_value_nodes_;
  // The front value node saves all parameters that are not in the kernel graph. These nodes are generally the
  // output of subgraph, or the switch condition node.
  NodeWithDeviceContext front_parameters_;

  // Parameters of control node which come from the host actor.
  std::vector<AnfNodePtr> control_node_parameters_;
  // The number of calls to func_graph.
  std::unordered_map<FuncGraphPtr, size_t> func_graph_to_call_num_;
  // The kernel graph of call exists in the front input node.
  // In the scene of funcgrarph recursive call, general input and call input are passed recursively, so a gather actor
  // is created for kernel graph which has a call input.
  std::unordered_map<KernelGraphPtr, DeviceContext *> call_input_kernel_graphs_;
  // Root funcgraph and its parameters.
  FuncGraphPtr root_func_graph_;
  std::vector<AnfNodePtr> root_graph_parameters_;

  // The dependency between kernel and call node in auto monad.
  std::unordered_map<AnfNodePtr, AnfNodePtr> kernel_to_call_nodes_;
};

using ControlNodeParserPtr = std::shared_ptr<ControlNodeParser>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_PARSER_H_
