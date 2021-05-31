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

#include "runtime/framework/control_node_parser.h"
#include "runtime/framework/actor/switch_actor.h"

namespace mindspore {
namespace runtime {

bool ControlNodeParser::IsCallNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &cnode = node->cast<CNodePtr>();
  const auto &inputs = cnode->inputs();
  return inputs[0]->isa<CNode>() || (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0]));
}

FuncGraphPtr ControlNodeParser::GetFuncGraphFromPartial(const AnfNodePtr &node) {
  const auto &partial_inputs = node->cast<CNodePtr>()->inputs();
  return GetValueNode<FuncGraphPtr>(partial_inputs[1]);
}

std::vector<FuncGraphPtr> ControlNodeParser::FetchFuncGraphbyCallNode(const CNodePtr &node) {
  std::vector<FuncGraphPtr> func_graphs;
  const auto &call_inputs = node->inputs();

  if (call_inputs[0]->isa<CNode>()) {
    const auto &cnode = call_inputs[0]->cast<CNodePtr>();
    const auto &cnode_inputs = cnode->inputs();
    if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
      for (size_t i = kSwitchTrueBranchPos; i < cnode_inputs.size(); ++i) {
        if (IsPrimitiveCNode(cnode_inputs[i], prim::kPrimPartial)) {
          func_graphs.emplace_back(GetFuncGraphFromPartial(cnode_inputs[i]));
        }
      }
    } else if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer) &&
               AnfAlgo::CheckPrimitiveType(cnode_inputs[kSwitchLayerBranchPos], prim::kPrimMakeTuple)) {
      const auto &tuple_inputs = cnode_inputs[kSwitchLayerBranchPos]->cast<CNodePtr>()->inputs();

      for (size_t i = 1; i < tuple_inputs.size(); ++i) {
        if (AnfAlgo::CheckPrimitiveType(tuple_inputs[i], prim::kPrimPartial)) {
          func_graphs.emplace_back(GetFuncGraphFromPartial(cnode_inputs[i]));
        } else if (IsValueNode<FuncGraph>(tuple_inputs[i])) {
          func_graphs.emplace_back(GetValueNode<FuncGraphPtr>(tuple_inputs[i]));
        }
      }
    } else {
      MS_LOG(EXCEPTION) << "Unable to identify call node" << node->DebugString();
    }
  } else if (call_inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(call_inputs[0])) {
    func_graphs.emplace_back(GetValueNode<FuncGraphPtr>(call_inputs[0]));
  } else {
    MS_LOG(EXCEPTION) << "Unable to identify call node" << node->DebugString();
  }
  return func_graphs;
}

std::vector<AnfNodePtr> ControlNodeParser::FetchFuncGraphOutput(const FuncGraphPtr &func_graph,
                                                                std::vector<AnfNodePtr> *call_nodes) {
  std::vector<AnfNodePtr> outputs;
  const auto &output = func_graph->output();
  const auto &real_output = AnfAlgo::VisitKernelWithReturnType(output, 0, false, {prim::kPrimTupleGetItem});
  if (find((*call_nodes).begin(), (*call_nodes).end(), real_output.first) != (*call_nodes).end()) {
    return outputs;
  }
  if (!IsCallNode(real_output.first)) {
    outputs.push_back(real_output.first);
    return outputs;
  }

  (*call_nodes).push_back(real_output.first);
  const auto &call_cnode = real_output.first->cast<CNodePtr>();
  std::vector<FuncGraphPtr> func_graphs = FetchFuncGraphbyCallNode(call_cnode);
  for (const auto &graph : func_graphs) {
    auto single_outputs = FetchFuncGraphOutput(graph, call_nodes);
    outputs.insert(outputs.end(), single_outputs.begin(), single_outputs.end());
  }
  return outputs;
}

std::pair<AnfNodePtr, DeviceContext *> ControlNodeParser::FetchBackendNodeByFrontNode(
  const AnfNodePtr &front_node, const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &front_to_front_parameter,
  const std::unordered_map<AnfNodePtr, std::pair<AnfNodePtr, DeviceContext *>> &front_to_backend_parameter,
  std::set<AnfNodePtr> *invalid_node) {
  // Check whether the front_node has been looked for.
  if ((*invalid_node).find(front_node) != (*invalid_node).end()) {
    return std::pair<AnfNodePtr, DeviceContext *>();
  }
  (*invalid_node).insert(front_node);

  const auto front_to_backend_iter = front_to_backend_parameter.find(front_node);
  if (front_to_backend_iter != front_to_backend_parameter.end()) {
    return front_to_backend_iter->second;
  }

  const auto &front_to_front_iter = front_to_front_parameter.find(front_node);
  if (front_to_front_iter == front_to_front_parameter.end()) {
    return std::pair<AnfNodePtr, DeviceContext *>();
  }
  for (const auto &next_node : front_to_front_iter->second) {
    auto banckend_node =
      FetchBackendNodeByFrontNode(next_node, front_to_front_parameter, front_to_backend_parameter, invalid_node);
    if (banckend_node.first != nullptr) {
      return banckend_node;
    }
  }
  return std::pair<AnfNodePtr, DeviceContext *>();
}

void ControlNodeParser::FetchFrontToFrontParameterMap(
  const std::vector<AnfNodePtr> &control_nodes,
  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> *front_to_front_parameter) {
  // Function used to collect the input of call node.
  const auto &call_input_parse = [front_to_front_parameter](const std::vector<AnfNodePtr> &parameters,
                                                            const std::vector<AnfNodePtr> &call_inputs,
                                                            const size_t call_input_start_pos) {
    for (size_t i = 0; i < call_inputs.size(); ++i) {
      if (call_inputs[i]->isa<Parameter>()) {
        (*front_to_front_parameter)[call_inputs[i]].push_back(parameters[i + call_input_start_pos]);
      }
    }
  };

  // Function used to collect the input of partial node.
  const auto &partial_input_parse = [call_input_parse, front_to_front_parameter](
                                      const AnfNodePtr &partial_node, const std::vector<AnfNodePtr> &call_inputs) {
    const auto &cnode = partial_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    const auto &func_graph = GetValueNode<FuncGraphPtr>(inputs[kPartialFuncGraphPos]);
    const auto &parameters = func_graph->parameters();
    for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
      if (inputs[i]->isa<Parameter>()) {
        (*front_to_front_parameter)[inputs[i]].push_back(parameters[i - kPartialInputStartPos]);
      }
    }
    call_input_parse(parameters, call_inputs, inputs.size() - kPartialInputStartPos);
  };

  // Function used to collect the input of switch node.
  const auto &switch_input_parse = [&](const AnfNodePtr &switch_node, const std::vector<AnfNodePtr> &call_inputs) {
    CNodePtr cnode = switch_node->cast<CNodePtr>();
    const auto &switch_inputs = cnode->inputs();
    if (AnfAlgo::CheckPrimitiveType(switch_node, prim::kPrimSwitch)) {
      // Parse the switch node. The switch node has two partial node inputs.
      if (AnfAlgo::CheckPrimitiveType(switch_inputs[kSwitchTrueBranchPos], prim::kPrimPartial)) {
        partial_input_parse(switch_inputs[kSwitchTrueBranchPos], call_inputs);
        partial_input_parse(switch_inputs[kSwitchFalseBranchPos], call_inputs);
      }
    } else {
      // Parse the switchlayer node. The switchlayer node has a maketuple node input, which is a tuple of funcgraphs.
      // call_inputs will be the input of these funcgraphs.
      const auto &tuple_node = switch_inputs[kSwitchLayerBranchPos]->cast<CNodePtr>();
      const auto &tuple_inputs = tuple_node->inputs();
      for (const auto &input : tuple_inputs) {
        if (AnfAlgo::CheckPrimitiveType(input, prim::kPrimPartial)) {
          partial_input_parse(input, call_inputs);
        } else {
          auto func_graph = GetValueNode<FuncGraphPtr>(input);
          call_input_parse(func_graph->parameters(), call_inputs, 0);
        }
      }
    }
  };

  for (const auto &node : control_nodes) {
    CNodePtr cnode = node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      // Call node which the first input node is a valuenode of funcgraph.
      const auto &func_graph = GetValueNode<FuncGraphPtr>(inputs[0]);
      const auto &parameters = func_graph->parameters();
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          (*front_to_front_parameter)[inputs[i]].push_back(parameters[i - kCallInputStartPos]);
        }
      }
    } else if (inputs[0]->isa<CNode>()) {
      // Call node which the first input node is a switch or switchlayer node.
      if ((!AnfAlgo::CheckPrimitiveType(inputs[0], prim::kPrimSwitch)) &&
          (!AnfAlgo::CheckPrimitiveType(inputs[0], prim::kPrimSwitchLayer))) {
        MS_LOG(EXCEPTION) << "First input node of call node is not switch, node:"
                          << AnfAlgo::GetNodeDebugString(inputs[0]);
      }
      std::vector<AnfNodePtr> call_inputs;
      call_inputs.assign(inputs.begin() + kCallInputStartPos, inputs.end());
      switch_input_parse(inputs[0], call_inputs);
    }
  }
}

std::vector<AnfNodePtr> ControlNodeParser::FetchControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes) {
  std::vector<AnfNodePtr> parameters;

  for (const auto &control_node : control_nodes) {
    CNodePtr cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      break;
    } else if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial)) {
      for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          parameters.emplace_back(inputs[i]);
        }
      }
    } else if (cnode->input(0)->isa<CNode>() || IsValueNode<FuncGraph>(cnode->input(0))) {
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          parameters.emplace_back(inputs[i]);
        }
      }
    }
  }

  return parameters;
}

std::vector<AnfNodePtr> ControlNodeParser::FetchAllBranchOutputs(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> call_nodes;
  return FetchFuncGraphOutput(func_graph, &call_nodes);
}

void ControlNodeParser::FetchFrontToBackendParameterMap(const std::vector<KernelGraphPtr> &graphs,
                                                        const std::vector<DeviceContext *> &device_contexts,
                                                        const std::vector<AnfNodePtr> &control_nodes,
                                                        FrontToBackendNodeWithContext *front_to_backend_parameter) {
  if (graphs.size() != device_contexts.size()) {
    MS_LOG(EXCEPTION) << "Graph num is not equal to device context num.";
  }

  // Fetch the mapping relationship between front parameters and backend parameters in the kernel graphs.
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    auto device_context = device_contexts[i];
    for (const auto &parameter : graph->parameters()) {
      auto front_node = graph->GetFrontAnfByBackendAnf(parameter);
      if (front_node != nullptr && front_node->isa<Parameter>() &&
          (*front_to_backend_parameter).find(front_node) == (*front_to_backend_parameter).end()) {
        (*front_to_backend_parameter)[front_node] = {parameter, device_context};
      }
    }
  }

  // Fetch the mapping relationship between front parameters and backend parameters in the control nodes. First
  // fetch the mapping relationship of the frontparameter. When the input of the call node or the partial node
  // is a parameter node, it means that the parameter is directly transmitted. If a parameter does not have a
  // corresponding backend node, then recursively find whether the front parameter corresponding to the parameter
  // has one.
  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> front_to_front_parameter;
  FetchFrontToFrontParameterMap(control_nodes, &front_to_front_parameter);

  for (const auto &front_pair : front_to_front_parameter) {
    std::set<AnfNodePtr> invalid_node;
    const auto &backend_node = FetchBackendNodeByFrontNode(front_pair.first, front_to_front_parameter,
                                                           *front_to_backend_parameter, &invalid_node);
    if (backend_node.first != nullptr) {
      (*front_to_backend_parameter)[front_pair.first] = backend_node;
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
