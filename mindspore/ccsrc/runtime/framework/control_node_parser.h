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
#include <queue>
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
constexpr size_t kSwitchInputNum = 4;
constexpr size_t kSwitchCondPos = 1;
constexpr size_t kSwitchPartialNum = 2;
constexpr size_t kSwitchLayerCondPos = 1;
constexpr size_t kSwitchLayerBranchPos = 2;
constexpr size_t kSwitchLayerInputNum = 3;
constexpr size_t kSwitchTrueBranchPos = 2;
constexpr size_t kSwitchFalseBranchPos = 3;
constexpr size_t kPartialFuncGraphPos = 1;
constexpr size_t kPartialInputStartPos = 2;
constexpr size_t kCallInputStartPos = 1;
constexpr size_t kMakeTupleInputStartPos = 1;
constexpr size_t kCNodeInputStartPos = 1;
constexpr size_t kReturnInputPos = 1;
constexpr size_t kSingleControlNode = 1;

const char kEntranceActorNameSuffix[] = "_EntranceActor";
const char kStackActorNameSuffix[] = "_StackActor";

using FrontToBackendNodeWithContext = std::unordered_map<AnfNodePtr, std::set<std::pair<AnfNodePtr, DeviceContext *>>>;
using FrontToBackendKernelWithContext = std::map<KernelWithIndex, std::pair<KernelWithIndex, DeviceContext *>>;
using FuncGraphToKernelGraph = std::unordered_map<FuncGraphPtr, std::vector<KernelGraphPtr>>;
using HostParameterToWeight = std::unordered_map<AnfNodePtr, std::set<AnfNodePtr>>;
using NodeWithDeviceContext = std::set<std::pair<AnfNodePtr, DeviceContext *>>;
using RealToFormalNode = std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>>;
using FormalToRealParameter = std::unordered_map<AnfNodePtr, std::set<KernelWithIndex>>;
using RealToFormalParameter = std::unordered_map<AnfNodePtr, std::set<AnfNodePtr>>;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
using FrontNodeToKernelGraph = std::unordered_map<AnfNodePtr, KernelGraphPtr>;
using FuncGraphCallRelation = std::unordered_map<FuncGraphPtr, std::vector<std::set<FuncGraphPtr>>>;

// Check whether the parameter is a weight. In the control flow, weight is passed to the subgraph, and in the subgraph,
// it is determined whether it is a weight.
bool HasAbstractRef(const AnfNodePtr &node);
// Get the front node corresponding to the backend node, if the front node is not a parameter node, return the
// corresponding cnode.
KernelWithIndex GetFrontNodeByKernelGraph(const AnfNodePtr &backend_node, const KernelGraphPtr &graph);

// ControlNodeParser is used to parse control nodes, and get the edges between nodes.
class ControlNodeParser {
 public:
  // Parse the control node and put the results of the parsing into member variables.
  void Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
             const std::vector<DeviceContext *> &device_contexts, const FuncGraphPtr &root_graph,
             const FuncGraphToKernelGraph &func_graph_to_kernel_graphs);

  bool IsInited() { return is_inited_; }
  // Check whether there is a call node in the front input nodes of the kernel graph.
  bool IsCallInputKernelGraph(const KernelGraphPtr &graph);
  // Check whether the data arrow of the kernel actor needs to be connected to the control actor.
  // There are two situations:
  // 1. In control flow, the parameter input needs to be connected to the entrance actor of the funcgraph.
  // 2. In the kernel graph with call node input, the data arrow needs to be connected to the stack actor.
  bool IsControlFlowDataArrow(const KernelGraphPtr &graph, const AnfNodePtr &node);

  const std::vector<AnfNodePtr> &control_node_parameters() const { return control_node_parameters_; }
  const FrontToBackendNodeWithContext &front_to_backend_parameters() const { return front_to_backend_parameters_; }
  const HostParameterToWeight &host_parameter_to_weights() const { return host_parameter_to_weights_; }
  const NodeWithDeviceContext &front_value_nodes() const { return front_value_nodes_; }

  // Fetch all funcgraphs that the call node may call.
  const std::set<FuncGraphPtr> &FetchFuncGraphbyCallNode(const AnfNodePtr &control_node);
  // Fetch the branch id corresponding to funcgraph.
  int FetchBranchIDByCallNode(const AnfNodePtr &call_node);
  // Fetch the funcgraph which the kernel belongs.
  FuncGraphPtr FetchKernelGraphByFrontNode(const AnfNodePtr &kernel);
  // Fetch the backend kernel of front node.
  KernelWithIndex FetchBackendNodeByFrontNode(const KernelWithIndex &node_with_index);

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  // Collect all front value nodes. In the control flow, when the input of the switch actor is the value node, these
  // value nodes will not enter the kernel graph, so these nodes need to be saved separately, and space is allocated for
  // them separately during initialization.
  // The interface is initialized by finding the backend node in the kernel graph that the front node finally sends to.
  void FetchFrontValueNode();
  // Create branch id for all call node in the control flow.
  void CreateBranchIDForCallNode(const std::vector<AnfNodePtr> &control_nodes);

  // Parse all the relationships between front parameters and backend parameters.The front parameters
  // include two parts:
  // 1. The parameter from kernel graph.
  // 2. The parameter from control nodes.
  void ParseFrontToBackendParameter(const std::vector<KernelGraphPtr> &graphs,
                                    const std::vector<DeviceContext *> &device_contexts);
  // The relationship between front parameters indicates that the parameter is directly used as the input of the
  // funcgraph. There are two situations:
  // 1. The parameter is used as the input of the call node,
  // 2. The parameter is used as the input of the partial and will be input to the funcgraph of the partial in the
  //    subsequent call node.
  void ParseFormalToRealParameter(const std::vector<AnfNodePtr> &control_nodes);
  // Recursively get all the real parameters corresponding to the formal parameters.
  void ParseAllRealParameterByFormalParameter(const AnfNodePtr &formal_parameter,
                                              const FormalToRealParameter &formal_to_real_parameters,
                                              std::set<KernelWithIndex> *total_real_parameters,
                                              std::set<AnfNodePtr> *invalid_real_parameter);

  // Parse the device context of the control node. In a heterogeneous scenario, different device contexts need to be
  // copied between different device memories. The analysis steps:
  // 1. Get the device context of the funcgraph parameter according to the device type of the kernel in the funcgraph.
  // 2. Determine the type of device context output by funcgraph according to the call relationship of funcgrpah.
  void ParseDeviceContext(const std::vector<AnfNodePtr> &control_nodes,
                          const std::vector<KernelGraphPtr> &kernel_graphs,
                          const std::vector<DeviceContext *> &device_contexts,
                          const FuncGraphToKernelGraph &func_graph_to_kernel_graphs);
  void ParseDeviceContextForFuncGraph(const std::vector<AnfNodePtr> &control_nodes,
                                      const std::vector<KernelGraphPtr> &kernel_graphs,
                                      const std::vector<DeviceContext *> &device_contexts,
                                      const FuncGraphToKernelGraph &func_graph_to_kernel_graphs);
  void ParseDeviceContextForControlNode(const DeviceContext *default_context);

  // In the actor model, when the funcgraph comes to an end temporarily, the exit of the funcgraph needs to notify
  // the entrance actor so that it can process next parameters. This is used to obtain the nodes corresponding to all
  // actors in the funcgraph that need to send control messages to the entrance.
  // These node are control nodes without control node input in the topological sort of the funcgraph.
  void ParseFirstControlNodeForFuncGraph(const std::vector<AnfNodePtr> &control_nodes);
  // Parse all funcgraphs that call nodes may call.
  void ParseCallNodeToFuncGraph(const std::vector<AnfNodePtr> &control_nodes);

  // Get the relationship between the front and backend of the executable kernel in all kernel graphs.
  void FetchFrontToBackendKernel(const std::vector<KernelGraphPtr> &graphs,
                                 const std::vector<DeviceContext *> &device_contexts);
  void FetchFrontNodeToKernelGraph(const std::vector<KernelGraphPtr> &graphs);
  // nodes and call nodes of the root funcgraph.
  void FetchControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes);
  // Get all the front weight parameters related to the weight in the host parameter.
  void FetchHostParameterToWeight();
  // Get all the kernel graphs where the input node has a call node.
  void FetchCallInputKernelGraph(const std::vector<KernelGraphPtr> &graphs,
                                 const std::vector<DeviceContext *> &device_contexts);
  // Get the dependency between kernel and call node in auto monad.
  void FetchAutoMonadNode(const std::vector<AnfNodePtr> &control_nodes);
  // Fetch the formal parameter in root graph by parameters in subgraph.
  AnfNodePtr FetchRootGraphFrontNodeBySubFrontNode(const AnfNodePtr &sub_front_node);

  // In control flow, funcgraph will be cut into multiple kernel graphs for execution, and this relationship is recorded
  // in this map.
  FuncGraphToKernelGraph func_graph_to_kernel_graphs_;
  // The kernel graph to which the front node belongs after the funcgraph is cut.
  FrontNodeToKernelGraph front_node_to_kernel_graph_;

  // The front to backend parameters is used to build and link the host data source actor in the control flow scenario.
  FrontToBackendNodeWithContext front_to_backend_parameters_;
  // Relationship between the front and backend of the executable kernel in all kernel graphs.
  FrontToBackendKernelWithContext front_to_backend_kernels_;

  // Relationship between formal parameters and real parameters.
  FormalToRealParameter formal_to_real_parameters_;
  RealToFormalParameter real_to_formal_parameters_;

  // Branch id of funcgraph.
  // In control flow, funcgraph will be called in multiple places, and the output of funcgraph needs to return to
  // different places. Therefore, a branch id is created for each funcgraph. When funcgraph is called, the branch
  // id needs to be sent to the gather actor corresponding to the funcgraph, and the gather will send the branch id
  // to its output switch actor.
  std::unordered_map<AnfNodePtr, int> call_node_to_branch_id_;
  std::unordered_map<AnfNodePtr, std::set<FuncGraphPtr>> call_node_to_func_graphs_;
  // host parameter to weights records the weights in the subgraph corresponding to the node in the root funcgraph.
  // When initializing the weights, all related weights need to be recorded as the same device tensor.
  HostParameterToWeight host_parameter_to_weights_;
  std::unordered_map<AnfNodePtr, AnfNodePtr> sub_front_node_to_root_front_node_;
  // The front value node saves all value nodes that are not in the kernel graph. These nodes are generally the
  // input of the control node.
  NodeWithDeviceContext front_value_nodes_;

  // Parameters of control node which come from the host actor.
  std::vector<AnfNodePtr> control_node_parameters_;
  // The kernel graph of call exists in the front input node.
  // In the scene of funcgrarph recursive call, general input and call input are passed recursively, so a gather actor
  // is created for kernel graph which has a call input.
  std::unordered_map<KernelGraphPtr, DeviceContext *> call_input_kernel_graphs_;
  // The dependency between kernel and call node in auto monad.
  std::unordered_map<AnfNodePtr, AnfNodePtr> kernel_to_call_nodes_;
  // Control nodes without a control node input in the topological sorting of funcgraph.
  std::unordered_map<FuncGraphPtr, std::set<AnfNodePtr>> func_graph_to_first_control_nodes_;

  // In heterogeneous scenario, each parameter has its own device context type, so the device context corresponding
  // to the type needs to be parsed in advance so that it can add some copy operation in the scheduler.
  // 1. The device context type of the formal parameters of funcgraph.
  std::unordered_map<FuncGraphPtr, std::vector<const DeviceContext *>> func_graph_to_device_contexts_;
  // 2. The device context type of the control node inputs.
  std::unordered_map<AnfNodePtr, std::vector<const DeviceContext *>> control_node_to_device_contexts_;

  // Is control flow enable.
  bool is_inited_{false};

  // Root funcgraph and its parameters.
  FuncGraphPtr root_func_graph_;
  std::vector<AnfNodePtr> root_graph_parameters_;
};

using ControlNodeParserPtr = std::shared_ptr<ControlNodeParser>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_PARSER_H_
