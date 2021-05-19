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
#include <tuple>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include "runtime/hardware/device_context.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;

// The meaning of switch node output tuple: 1. switch node 2. output branch id 3. output index.
using SwitchNodeOutput = std::tuple<AnfNodePtr, size_t, size_t>;
// The output arrow info: 1. from node index 2.to node 3. to node index.
using NodeOutputInfo = std::tuple<size_t, AnfNodePtr, size_t>;
// External input of kernel graph, the key means the front node of input
// and value vector is pairs of from node and to node.
using KernelGraphExternInput = std::unordered_map<AnfNodePtr, std::vector<std::pair<KernelWithIndex, KernelWithIndex>>>;

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

// Get all possible outputs of funcgraph. Search recursively by the input of the return node of the funcgraph.
// If the input is a call node, enter all the funcgraphs it called until the input of the non-call node is found
// and return all of the output node.
std::vector<AnfNodePtr> GetAllBranchOutputs(const FuncGraphPtr &func_graph);

// ControlNodeParser is used to parse control nodes, and get the edges between nodes. Call node is used to
// implement the call relationship between funcgraphs, the actual parameters are connected to the call node,
// and the call node then calls the corresponding funcgraph, sends the actual parameters to next nodes
// according to the relationship between the actual parameters and formal parameters.
// From the function of the call node, the structure of the edge can be split into two parts:
// the relationship between the output nodes and the formal parameters, and relationship between formal parameters
//  and input nodes. And then they are connected to become the final edge.
// Therefore, the analysis is mainly divided into 2 steps:
// 1. Get all input and output relationship with formal parameters;
// 2. Connect all input and output to edges.
class ControlNodeParser {
 public:
  ControlNodeParser() = default;
  ~ControlNodeParser() = default;
  ControlNodeParser(const ControlNodeParser &) = delete;
  ControlNodeParser &operator=(const ControlNodeParser &) = delete;

  // Analyze the relationship between switch and kernel nodes.
  // Parameter kernel_graph_input_ indicates that in a multi-graph case, parameters of the subgraph
  // should be the passed in when called by main graph, rather than directly sent by the input, so it
  // needs to be connected when parsing the control node.
  // The result of parse is the edge between nodes, which is stored in member variables.
  void Parse(const std::vector<AnfNodePtr> &control_nodes, const KernelGraphExternInput &kernel_graph_input_);

 private:
  friend class GraphScheduler;

  void ParseCall(const AnfNodePtr &node);
  void ParseSwitch(const AnfNodePtr &node, const std::vector<AnfNodePtr> &inputs_on_call);
  void ParseSwitchLayer(const AnfNodePtr &node, const std::vector<AnfNodePtr> &inputs_on_call);
  void ParsePartial(const AnfNodePtr &node, const std::vector<AnfNodePtr> &switch_inputs, const size_t branch_id,
                    const std::vector<AnfNodePtr> &inputs_on_call);
  void ParseInput(const AnfNodePtr &from_node, const AnfNodePtr &to_node, size_t to_index);
  // Parse input which is a call node, This means that we need to find the output of the funcgraph called by
  // the call node as the input of to_node.
  void ParseCallInput(const CNodePtr &from_node, const AnfNodePtr &to_node, size_t to_index);

  // Get all inputs of switch nodes, inputs_on_call is the inputs which was inputs of call node which the switch
  // node connected.
  std::vector<AnfNodePtr> GetSwitchInput(const AnfNodePtr &node, const std::vector<AnfNodePtr> &inputs_on_call);

  // Connect the input and output of the call node to get the final edge.
  void LinkInputAndOutput();
  // Link the formal parameter to its final actual parameter in member variables parameter_to_arguments_.
  // For example, if we have a map like {{a, b}, {b, c}, {c, d}}, final we will get {{a, d}, {b, d}, {c, d}}.
  void LinkParameterAndArgument();
  // Recursively find all inputs corresponding to node.
  void GetOutputNode(const AnfNodePtr &node, std::vector<KernelWithIndex> *inputs);

  // Relationship between formal parameter and actual parameter
  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> actual_to_formal_parameters_;
  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> formal_to_actual_parameters_;

  // In control nodes, edge is a structure like kernel output --> formal parameter --> kernel input.
  // In the parsing process, the edge is divided into two parts, input and output:
  // input represents the relationship between formal parameters and kernel input,
  // output represents the relationship between kernel output and formal parameters.
  // In order to merge input and output into edge, both of them are stored in map and use parameter as the key.
  // All inputs.
  std::unordered_map<AnfNodePtr, std::vector<KernelWithIndex>> parameter_to_input_;
  // Three kinds of output.
  // output of switch node.
  std::unordered_map<AnfNodePtr, std::vector<SwitchNodeOutput>> parameter_to_switch_out_;
  // output of kernel node.
  std::unordered_map<AnfNodePtr, std::vector<KernelWithIndex>> parameter_to_kernel_out_;
  // parameters in root funcgraph.
  std::vector<AnfNodePtr> parameters_;

  // Final edges.
  std::unordered_map<AnfNodePtr, std::vector<NodeOutputInfo>> kernel_outputs_;
  std::unordered_map<std::pair<AnfNodePtr, size_t>, std::vector<NodeOutputInfo>, PairHash> switch_outputs_;
  std::unordered_map<AnfNodePtr, std::vector<KernelWithIndex>> parameter_out_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_PARSER_H_
