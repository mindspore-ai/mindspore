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
#include <stack>
#include <utility>
#include <algorithm>
#include "utils/hash_map.h"
#include "runtime/hardware/device_context.h"
#include "include/backend/kernel_graph.h"

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

constexpr size_t kCsrTensorIndPtrIndex = 0;
constexpr size_t kCsrTensorIndicesIndex = 1;
constexpr size_t kCsrTensorValuesIndex = 2;
constexpr size_t kCsrTensorDenseShapeIndex = 3;
constexpr size_t kCsrParamOutputSize = 3;
constexpr size_t kCooTensorIndicesIndex = 0;
constexpr size_t kCooTensorValuesIndex = 1;
constexpr size_t kCooTensorDenseShapeIndex = 2;
constexpr size_t kMakeCSRTensorInputStartPos = 1;
constexpr size_t kMakeTensorInputStartPos = 1;
constexpr size_t kMakeCSRTensorInputNum = 4;
constexpr size_t kMakeCOOTensorInputNum = 3;

using NodeWithIndexToContext = std::pair<KernelWithIndex, DeviceContext *>;
struct NodeWithContextCmp {
  bool operator()(const NodeWithIndexToContext &node1, const NodeWithIndexToContext &node2) const {
    return node1.second->GetDeviceType() < node2.second->GetDeviceType();
  }
};

using FrontToBackendNodeWithContext = std::map<KernelWithIndex, std::set<NodeWithIndexToContext, NodeWithContextCmp>>;
using FrontToBackendKernelWithContext = std::map<KernelWithIndex, std::pair<KernelWithIndex, DeviceContext *>>;
using FuncGraphToKernelGraphGroup = mindspore::HashMap<FuncGraphPtr, std::vector<std::vector<KernelGraphPtr>>>;
using HostParameterToWeight = std::map<AnfNodePtr, std::set<AnfNodePtr>>;
using NodeWithDeviceContext = std::set<std::pair<KernelWithIndex, const DeviceContext *>>;
using RealToFormalNode = mindspore::HashMap<AnfNodePtr, std::vector<AnfNodePtr>>;
using FormalToRealParameter = std::map<KernelWithIndex, std::set<KernelWithIndex>>;
using RealToFormalParameter = std::map<KernelWithIndex, std::set<KernelWithIndex>>;
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
using FrontNodeToKernelGraph = mindspore::HashMap<AnfNodePtr, KernelGraphPtr>;
using FuncGraphCallRelation = mindspore::HashMap<FuncGraphPtr, std::vector<std::set<FuncGraphPtr>>>;
using FuncGraphToCallNode = mindspore::HashMap<FuncGraphPtr, std::set<AnfNodePtr>>;
using CallNodeToFuncGraph = mindspore::HashMap<AnfNodePtr, std::set<FuncGraphPtr>>;
using KernelGraphToDeviceContext = mindspore::HashMap<KernelGraphPtr, DeviceContext *>;
using GroupNameToCommuNodes =
  mindspore::HashMap<std::string, std::pair<std::vector<CNodePtr>, std::vector<KernelGraphPtr>>>;
// In the control flow, heterogeneous kernel graphs need to be reconnected in the same group, and the kernel graph
// group info is used to store the inputs and outputs of the group.
// Need stack indicates whether a stack actor needs to be created for the group.
// Level indicates the level of the output of the graph in the group.
struct KernelGraphGroupInfo {
  bool need_stack_{0};
  size_t level_;
  std::string group_name_;
  std::set<KernelGraphPtr> graphs_;
  std::set<AnfNodePtr> monad_inputs_;
  std::set<KernelWithIndex> monad_outputs_;
  std::map<KernelWithIndex, const DeviceContext *> front_input_nodes_;
  FrontToBackendKernelWithContext front_output_nodes_;
};
using KernelGraphGroupInfoPtr = std::shared_ptr<KernelGraphGroupInfo>;

// Check whether the node is a csr node.
bool IsCsrNode(const AnfNodePtr &node);
bool IsCooNode(const AnfNodePtr &node);
// Get the front node corresponding to the backend node, if the front node is not a parameter node, return the
// corresponding cnode.
KernelWithIndex GetFrontNodeByKernelGraph(const AnfNodePtr &backend_node, const KernelGraph *const graph);
// Get all the real input of the frontend node, skip the virtual node like maketuple, tuplegetitem.
std::vector<KernelWithIndex> FetchInputNodeByCNode(const AnfNodePtr &node);
// Fetch the sub abstract from the top abstract by the index.
abstract::AbstractBasePtr FetchAbstractByIndex(const AbstractBasePtr &abstract, size_t index);
// Fetch the real input of tuple get item node.
KernelWithIndex FetchRealNodeByGetItem(const KernelWithIndex &node_with_index);
// Check if the partial node is valid.
// Invalid partial nodes are those partial cnodes whose funcgraph is deadnode.
bool IsInvalidPartial(const AnfNodePtr &node);
// Check whether the switch node abstract is functional.
bool IsPartialInput(const AnfNodePtr &node);
// Fetch the depend nodes according to the monad node.
void FetchRealDependNodeByAutoMonad(const AnfNodePtr &node, std::set<AnfNodePtr> *const depend_nodes);
// Get all the depend nodes of node in side effect.
std::vector<AnfNodePtr> FetchAllMonadNodeByNode(const AnfNodePtr &node);
// ControlNodeParser is used to parse control nodes, and get the edges between nodes.
class ControlNodeParser {
 public:
  ControlNodeParser() : is_inited_(false), root_func_graph_(nullptr) {}

  // Parse the control node and put the results of the parsing into member variables.
  void Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
             const std::vector<DeviceContext *> &device_contexts, const FuncGraphPtr &root_graph,
             const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs);

  bool IsInited() const { return is_inited_; }
  // Check whether there is a call node in the front input nodes of the kernel graph.
  bool IsCallInputKernelGraph(KernelGraph *const graph);
  // Check whether there is a call node in the front input nodes of the kernel graph group.
  bool IsCallInputKernelGraphGroup(const std::string &group_name);
  // Check whether the data arrow of the kernel actor needs to be connected to the control actor.
  // There are two situations:
  // 1. In control flow, the parameter input needs to be connected to the entrance actor of the funcgraph.
  // 2. In the kernel graph with call node input, the data arrow needs to be connected to the stack actor.
  bool IsControlFlowDataArrow(const KernelGraphPtr &graph, const AnfNodePtr &backend_node);
  // Only the parameters of root graph are persistent and fetched from the store, the parameters of sub graphs are not
  // persistent and real parameters passed.
  bool IsRootGraphPersistentDeviceTensor(const AnfNodePtr &node);
  bool IsRecursionCallNode(const AnfNodePtr &node);
  bool IsNeedStackControlNode(const AnfNodePtr &node);
  // If there is a recursive call node in the input of the kernel graph, the graph is recursive.
  bool IsRecursionKernelGraph(const KernelGraphPtr &graph);
  bool IsSameKernelGraphGroup(const AnfNodePtr &node, const KernelGraphPtr &graph);
  bool IsInputInSameLevel(const AnfNodePtr &node);
  // If the two input call nodes will call the same recursion graph in same time.
  bool IsParallelCallRecursionGraph(const AnfNodePtr &call_node1, const AnfNodePtr &call_node2,
                                    const FuncGraphToCallNode &func_graph_to_call_node);
  const std::vector<KernelWithIndex> &control_node_parameters() const { return control_node_parameters_; }
  const FrontToBackendNodeWithContext &front_to_backend_parameters() const { return front_to_backend_parameters_; }
  const NodeWithDeviceContext &front_value_nodes() const { return front_value_nodes_; }

  // Fetch all funcgraphs that the call node may call.
  const std::set<FuncGraphPtr> &FetchFuncGraphbyCallNode(const AnfNodePtr &control_node);
  // Fetch the branch id corresponding to funcgraph.
  int FetchBranchIDByCallNode(const AnfNodePtr &call_node);
  // Fetch the kernel graph which the kernel belongs.
  KernelGraphPtr FetchKernelGraphByFrontNode(const AnfNodePtr &kernel);
  // Fetch the backend kernel of front node.
  KernelWithIndex FetchBackendNodeByFrontNode(const KernelWithIndex &node_with_index);
  FuncGraphPtr FetchFuncGraphByKernelGraph(const KernelGraph *const graph);
  std::string FetchGroupNameByKernelGraph(const KernelGraphPtr &graph);
  NodeWithIndexToContext FetchBackendParameterWithContextByFrontParameter(
    const KernelWithIndex &front_parameter_with_index);
  // Create tensor for value like scalar or monad U.
  tensor::TensorPtr CreateTensorForValue(const ValuePtr &value);

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  // Collect all front value nodes. In the control flow, when the input of the switch actor is the value node, these
  // value nodes will not enter the kernel graph, so these nodes need to be saved separately, and space is allocated for
  // them separately during initialization.
  // The interface is initialized by finding the backend node in the kernel graph that the front node finally sends to.
  void FetchFrontValueNode(const std::vector<AnfNodePtr> &control_nodes, const DeviceContext *const default_context);
  void CreateDeviceTensors(const std::vector<AnfNodePtr> &control_nodes, const DeviceContext *const default_context);
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
  void ParseAllRealParameterByFormalParameter(const KernelWithIndex &formal_parameter,
                                              const FormalToRealParameter &formal_to_real_parameters,
                                              std::set<KernelWithIndex> *const total_real_parameters,
                                              std::set<KernelWithIndex> *invalid_real_parameter);
  // Get all the call nodes without a recursion call relation.
  void ParseUnRecursionCallNode();

  // Parse the device context of the control node. In a heterogeneous scenario, different device contexts need to be
  // copied between different device memories. The analysis steps:
  // 1. Get the device context of the funcgraph parameter according to the device type of the kernel in the funcgraph.
  // 2. Determine the type of device context output by funcgraph according to the call relationship of funcgrpah.
  // 3. Determine the type of device context output for the real parameters on the partial nodes and call nodes.
  void ParseDeviceContext(const std::vector<AnfNodePtr> &control_nodes,
                          const std::vector<KernelGraphPtr> &kernel_graphs,
                          const std::vector<DeviceContext *> &device_contexts, DeviceContext *default_context,
                          const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs);
  void ParseDeviceContextForFuncGraph(const std::vector<KernelGraphPtr> &kernel_graphs,
                                      const std::vector<DeviceContext *> &device_contexts,
                                      DeviceContext *default_context,
                                      const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs);
  void ParseDeviceContextForReturnNode(const DeviceContext *default_context);
  void ParseDeviceContextForCallNode(const std::vector<AnfNodePtr> &control_nodes);
  void ParseDeviceContextForPartialNode(const std::vector<AnfNodePtr> &control_nodes);

  // In the actor model, when the funcgraph comes to an end temporarily, the exit of the funcgraph needs to notify
  // the entrance actor so that it can process next parameters. This is used to obtain the nodes corresponding to all
  // actors in the funcgraph that need to send control messages to the entrance.
  // These node are control nodes without control node input in the topological sort of the funcgraph.
  void ParseFirstControlNodeAndKernelGraphForFuncGraph(const std::vector<AnfNodePtr> &control_nodes);
  // Parse all funcgraphs that call nodes may call.
  void ParseCallNodeToFuncGraph(const std::vector<AnfNodePtr> &control_nodes);

  // Get the relationship between the front and backend of the executable kernel in all kernel graphs.
  void ParseFrontToBackendKernel(const std::vector<KernelGraphPtr> &graphs,
                                 const std::vector<DeviceContext *> &device_contexts);
  void ParseFrontNodeToKernelGraph(const std::vector<KernelGraphPtr> &graphs);
  // nodes and call nodes of the root funcgraph.
  void ParseControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes);
  // Get the control nodes and kernel graphs which need to add a stack actor for them.
  // When a control node or kernel graph has input that is a call node, you need to add a stack actor for it.
  void ParseNeedStackControlNode(const std::vector<AnfNodePtr> &control_nodes);
  bool IsCallNodeNeedStack(const AnfNodePtr &node);
  void ParseKernelGraphGroup(const KernelGraphToDeviceContext &kernel_graph_to_device_contexts);
  // Parse the level of inputs and outputs of graphs and all control nodes.
  void ParseNodeLevel(const std::vector<AnfNodePtr> &control_nodes);
  // Get the level of the control node, recursively traverse all the inputs of the node, and find the largest level
  // among them.
  size_t ParseControlNodeLevel(const AnfNodePtr &node, std::set<AnfNodePtr> *checked_nodes);
  // When there is the possibility of calling the same funcgraph at multiple places in the graph, the graph cannot
  // be executed in parallel, and all call nodes need to be executed serially.
  void InsertDependForParallelCall(const std::vector<AnfNodePtr> &control_nodes);
  // When the parameter is directly used as the condition of the switch, there will be no back-end node, and a device
  // tensor needs to be created for it.
  void CreateDeviceTensorForRootGraphParameter(DeviceContext *const default_context);
  // In control flow, funcgraph will be cut into multiple kernel graphs for execution, and this relationship is recorded
  // in this map.
  FuncGraphToKernelGraphGroup func_graph_to_kernel_graph_groups_;
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
  mindspore::HashMap<AnfNodePtr, int> call_node_to_branch_id_;
  // Level indicates that the input of the node depends on the number of the recursive call node in the funcgraph.
  // During graph scheduler, the input needs to be graded according to the input's dependence on the recursive call
  // node, and according to this level, the lower-level inputs are pushed in the stack actor. When arranging, first
  // sort the call nodes in the funcgraph according to their topological relationships, and then confirm the
  // dependencies of other nodes on these call nodes in turn.
  // For example, the dependencies are a -> b, b -> d, c -> d, where b is a call node, then the level of a and c is 0,
  // and the level of bd is 1, then since d has inputs with different levels of b and c, it is necessary to add a
  // stack to d.
  mindspore::HashMap<AnfNodePtr, size_t> node_to_level_;
  CallNodeToFuncGraph call_node_to_func_graphs_;
  // The front value node saves all value nodes that are not in the kernel graph. These nodes are generally the
  // input of the control node.
  NodeWithDeviceContext front_value_nodes_;

  // Parameters of control node which come from the host actor.
  std::vector<KernelWithIndex> control_node_parameters_;
  // The kernel graph of call exists in the front input node.
  // In the scene of funcgrarph recursive call, general input and call input are passed recursively, so a stack actor
  // is created for kernel graph which has a call input.
  std::set<KernelGraph *> call_input_kernel_graphs_;
  std::set<KernelGraphGroupInfoPtr> kernel_graph_group_infos_;
  // Control nodes without a control node input in the topological sorting of funcgraph.
  mindspore::HashMap<FuncGraphPtr, std::set<AnfNodePtr>> func_graph_to_first_control_nodes_;
  // Kernel graphs need to link a control arrow to its entrance actor.
  // In the recursive scene, some kernel graph needs to be completed before the next set of data is sent by the
  // entrance actor. At this time, it is necessary to connect a control arrow from the exit actor of the graph
  // to the entrance actor.
  mindspore::HashMap<FuncGraphPtr, std::set<KernelGraphGroupInfoPtr>> func_graph_to_first_kernel_graphs_;
  // Call nodes without recursive call. The funcgraphs of the call will not call the funcgraph where the call node
  // belong.
  std::set<AnfNodePtr> unrecursion_call_nodes_;
  // Those control nodes that need to create the corresponding stack actor, when there is a call node in the inputs
  // of the control node, the stack actor is needed to collect these inputs.
  std::set<AnfNodePtr> need_stack_control_nodes_;

  // In heterogeneous scenario, each parameter has its own device context type, so the device context corresponding
  // to the type needs to be parsed in advance so that it can add some copy operation in the scheduler.
  // 1. The device context type of the formal parameters of funcgraph.
  mindspore::HashMap<FuncGraphPtr, std::vector<const DeviceContext *>> func_graph_to_device_contexts_;
  // 2. The device context type of the control node inputs.
  mindspore::HashMap<AnfNodePtr, std::vector<const DeviceContext *>> control_node_to_device_contexts_;

  // Kernel graph to the group info it belongs.
  mindspore::HashMap<KernelGraphPtr, KernelGraphGroupInfoPtr> kernel_graphs_to_group_info_;
  // Scalar value will be convert to tensor in control flow, these tensors are placed in the vector.
  std::vector<tensor::TensorPtr> control_node_tensors_;
  // Is control flow enable.
  bool is_inited_;

  // Root funcgraph and its parameters.
  FuncGraphPtr root_func_graph_;
  std::vector<AnfNodePtr> root_graph_parameters_;
};

using ControlNodeParserPtr = std::shared_ptr<ControlNodeParser>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_PARSER_H_
