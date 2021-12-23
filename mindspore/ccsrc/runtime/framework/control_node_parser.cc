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
#include "runtime/framework/actor/actor_common.h"
#include "utils/func_graph_analyzer.h"
#include "utils/convert_utils.h"
#include "abstract/utils.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
namespace {
// Check whether node has a partial structure, a node is a partial structure whicih:
// 1. a partial cnode.
// 2. a funcgraph value node.
bool IsPartial(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return (node->isa<ValueNode>() && IsValueNode<FuncGraph>(node)) ||
         AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial);
}

// Check if node is a value node need to create a device tensor.
bool IsFrontValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->isa<ValueNode>() && (!IsValueNode<FuncGraph>(node)) && (!IsValueNode<Primitive>(node));
}

// Get funcgraph in partial structure.
// Depth represents the number of layers of the call. When the first input of the call node is a call node,
// the funcgraph in the return value of the inner call needs to be returned.
FuncGraphPtr GetFuncGraphFromPartial(const AnfNodePtr &node, std::stack<size_t> *output_indexs) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(output_indexs);

  if (output_indexs->empty()) {
    if (node->isa<ValueNode>() && IsValueNode<FuncGraph>(node)) {
      // Value node of funcgraph.
      return GetValueNode<FuncGraphPtr>(node);
    } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      // Partial cnode.
      const auto &partial_inputs = node->cast<CNodePtr>()->inputs();
      return GetValueNode<FuncGraphPtr>(partial_inputs[kPartialFuncGraphPos]);
    } else {
      MS_LOG(EXCEPTION) << "Invalid partial construct node:" << node->DebugString();
    }
    return nullptr;
  }

  // Get funcgraph in the output of inner call.
  FuncGraphPtr func_graph;
  if (node->isa<ValueNode>() && IsValueNode<FuncGraph>(node)) {
    func_graph = GetValueNode<FuncGraphPtr>(node);
  } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    const auto &partial_inputs = node->cast<CNodePtr>()->inputs();
    const auto &func_graph_node = partial_inputs[kPartialFuncGraphPos];
    if (!func_graph_node->isa<ValueNode>() || !IsValueNode<FuncGraph>(func_graph_node)) {
      MS_LOG(EXCEPTION) << "Invalid partial node:" << node->DebugString();
    }
    func_graph = GetValueNode<FuncGraphPtr>(func_graph_node);
  } else {
    MS_LOG(EXCEPTION) << "Invalid partial node:" << node->DebugString();
  }
  MS_EXCEPTION_IF_NULL(func_graph);

  size_t index = output_indexs->top();
  output_indexs->pop();
  const auto &output = func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  KernelWithIndex real_output_with_index = AnfAlgo::VisitKernelWithReturnType(output, 0);
  auto real_output = real_output_with_index.first;
  MS_EXCEPTION_IF_NULL(real_output);
  if (AnfAlgo::CheckPrimitiveType(real_output, prim::kPrimMakeTuple)) {
    const auto &cnode = real_output->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs.size() <= index + kMakeTupleInputStartPos) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << index << " for node:" << real_output->DebugString();
    }
    real_output = inputs[index + kMakeTupleInputStartPos];
  }
  MS_EXCEPTION_IF_NULL(real_output);
  return GetFuncGraphFromPartial(real_output, output_indexs);
}

// Find all funcgraphs that the call node will call.
// The output index represents the index of the funcgraph in the graph output. When the funcgraph is passed through
// the function return value, the index in the return value needs to be placed on the stack so that the index can be
// obtained from the stack when the graph output is found.
std::set<FuncGraphPtr> GetFuncGraphbyCallNode(const AnfNodePtr &node, std::stack<size_t> *output_indexs,
                                              AnfNodePtr make_tuple = nullptr) {
  MS_EXCEPTION_IF_NULL(node);
  std::set<FuncGraphPtr> func_graphs;
  if (!node->isa<CNode>()) {
    return func_graphs;
  }

  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &call_input0 = cnode->input(0);
  MS_EXCEPTION_IF_NULL(call_input0);
  const auto &call_input_with_index = AnfAlgo::VisitKernelWithReturnType(call_input0, 0);
  const auto &real_input = call_input_with_index.first;
  if (AnfAlgo::IsCallNode(real_input)) {
    output_indexs->push(call_input_with_index.second);
    return GetFuncGraphbyCallNode(real_input, output_indexs);
  }

  if (AnfAlgo::CheckPrimitiveType(real_input, prim::kPrimSwitch)) {
    // First input node of call is switch node.
    const auto &input_cnode = real_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    const auto &switch_inputs = input_cnode->inputs();
    for (size_t i = kSwitchTrueBranchPos; i < switch_inputs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(switch_inputs[i]);
      std::stack<size_t> tmp_output_indexs = *output_indexs;
      (void)func_graphs.emplace(GetFuncGraphFromPartial(switch_inputs[i], &tmp_output_indexs));
    }
  } else if (AnfAlgo::CheckPrimitiveType(real_input, prim::kPrimSwitchLayer)) {
    // First input node of call is switch layer node.
    const auto &input_cnode = real_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    const auto &tuple_node = input_cnode->input(kSwitchLayerBranchPos);
    if (AnfAlgo::CheckPrimitiveType(tuple_node, prim::kPrimMakeTuple)) {
      const auto &tuple_inputs = tuple_node->cast<CNodePtr>()->inputs();
      for (size_t i = kMakeTupleInputStartPos; i < tuple_inputs.size(); ++i) {
        MS_EXCEPTION_IF_NULL(tuple_inputs[i]);
        std::stack<size_t> tmp_output_indexs = *output_indexs;
        func_graphs.emplace(GetFuncGraphFromPartial(tuple_inputs[i], &tmp_output_indexs));
      }
    } else if (tuple_node->isa<Parameter>()) {
      const auto &abstract = tuple_node->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      if (!abstract->isa<abstract::AbstractTuple>()) {
        MS_LOG(EXCEPTION) << "Invalid abstract:" << abstract->ToString() << " in node:" << tuple_node->DebugString()
                          << " for switch layer node:" << real_input->DebugString();
      }
      MS_EXCEPTION_IF_NULL(make_tuple);
      auto partials = make_tuple->cast<CNodePtr>()->inputs();
      for (const auto partial : partials) {
        if (IsPartial(partial)) {
          std::stack<size_t> tmp_output_indexs;
          (void)func_graphs.emplace(GetFuncGraphFromPartial(partial, &tmp_output_indexs));
        }
      }
    } else {
      MS_LOG(EXCEPTION) << "Invalid input tuple node:" << tuple_node->DebugString()
                        << " for switch layer node:" << cnode->DebugString();
    }
  } else if (IsPartial(real_input)) {
    // First input node of call is partial node or value node of funcgraph.
    (void)func_graphs.emplace(GetFuncGraphFromPartial(real_input, output_indexs));
  } else {
    MS_LOG(EXCEPTION) << "Unable to identify call node" << real_input->DebugString();
  }
  return func_graphs;
}

// Fetch real input node in maketuple.
KernelWithIndex FetchRealInputNode(const KernelWithIndex &node_with_index) {
  const auto &node = node_with_index.first;
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    return node_with_index;
  }

  const auto &abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  size_t output_num = AnfAlgo::GetOutputNumByAbstract(abstract);
  if (output_num <= node_with_index.second) {
    MS_LOG(EXCEPTION) << "Invalid index:" << node_with_index.second << "for tuple node:" << node->DebugString();
  }

  const auto &cnode = node->cast<CNodePtr>();
  const auto &inputs = cnode->inputs();
  size_t real_index = node_with_index.second;
  for (size_t i = kMakeTupleInputStartPos; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    const auto &sub_abstract = inputs[i]->abstract();
    MS_EXCEPTION_IF_NULL(sub_abstract);
    size_t tmp_index = AnfAlgo::GetOutputNumByAbstract(sub_abstract);
    // If it is not the output of node, need to subtract the number of inputs of it.
    if (real_index >= tmp_index) {
      real_index -= tmp_index;
      continue;
    }
    return {inputs[i], real_index};
  }
  MS_LOG(EXCEPTION) << "Failed to get real output from node:" << node->DebugString()
                    << " index:" << node_with_index.second;
  return {};
}

// Fetch all the output index in the sub-abstract of abstract.
std::set<size_t> FetchRealIndexByAbstract(const AbstractBasePtr &abstract, size_t index) {
  MS_EXCEPTION_IF_NULL(abstract);
  AbstractBasePtr dst_abstract = abstract;
  size_t pre_abstract_num = 0;

  // Fetch the dest abstract by index, and the abstracts num before the dest abstract.
  if (abstract->isa<abstract::AbstractCSRTensor>()) {
    auto csr_abs = abstract->cast<abstract::AbstractCSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_abs);
    switch (index) {
      case kCsrTensorIndPtrIndex:
        dst_abstract = csr_abs->indptr();
        pre_abstract_num = kCsrTensorIndPtrIndex;
        break;
      case kCsrTensorIndicesIndex:
        dst_abstract = csr_abs->indices();
        pre_abstract_num = kCsrTensorIndicesIndex;
        break;
      case kCsrTensorValuesIndex:
        dst_abstract = csr_abs->values();
        pre_abstract_num = kCsrTensorValuesIndex;
        break;
      case kCsrTensorDenseShapeIndex:
        dst_abstract = csr_abs->dense_shape();
        pre_abstract_num = kCsrTensorDenseShapeIndex;
        break;
      default:
        MS_LOG(EXCEPTION) << "Invalid index:" << index << " for abstract:" << abstract->ToString();
        break;
    }
  } else if (abstract->isa<abstract::AbstractTuple>()) {
    auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_abstract);
    const auto &sub_abstracts = tuple_abstract->elements();
    if (sub_abstracts.size() <= index) {
      MS_LOG(EXCEPTION) << "Invalid index:" << index << " for abstract:" << abstract->ToString();
    }
    for (size_t i = 0; i < index; ++i) {
      pre_abstract_num += AnfAlgo::GetOutputNumByAbstract(sub_abstracts[i]);
    }
    dst_abstract = sub_abstracts[index];
  } else {
    if (index != 0) {
      MS_LOG(EXCEPTION) << "Invalid abstract index:" << index << " for abstract:" << abstract->ToString();
    }
  }
  MS_EXCEPTION_IF_NULL(dst_abstract);

  // Fetch real output index.
  size_t ouput_num = AnfAlgo::GetOutputNumByAbstract(dst_abstract);
  std::set<size_t> real_indexs;
  for (size_t i = pre_abstract_num; i < ouput_num + pre_abstract_num; ++i) {
    real_indexs.emplace(i);
  }
  return real_indexs;
}

// Get all the real parameters corresponding to node.
void FetchRealParameterByNode(const KernelWithIndex &node, std::set<KernelWithIndex> *real_parameters,
                              std::set<KernelWithIndex> *invalid_call_nodes,
                              const mindspore::HashMap<AnfNodePtr, std::set<FuncGraphPtr>> &call_node_to_func_graphs) {
  auto node_with_index = node;
  if (!node.first->isa<ValueNode>()) {
    node_with_index = AnfAlgo::VisitKernelWithReturnType(node.first, node.second);
  }
  if (node_with_index.first->isa<ValueNode>() || node_with_index.first->isa<Parameter>()) {
    // If node is a valuenode or parameter, the real parameter is itself.
    real_parameters->emplace(node_with_index);
  } else if (AnfAlgo::IsCallNode(node_with_index.first)) {
    // If node is a call node, the real parameters are the outputs of funcgraph the node called.
    if (invalid_call_nodes->find(node_with_index) != invalid_call_nodes->end()) {
      return;
    }
    invalid_call_nodes->emplace(node_with_index);
    const auto &iter = call_node_to_func_graphs.find(node_with_index.first);
    if (iter == call_node_to_func_graphs.end()) {
      MS_LOG(EXCEPTION) << "Invalid call node:" << node_with_index.first->DebugString();
    }
    const auto &func_graphs = iter->second;
    for (const auto &func_graph : func_graphs) {
      MS_EXCEPTION_IF_NULL(func_graph);
      FetchRealParameterByNode({func_graph->output(), node_with_index.second}, real_parameters, invalid_call_nodes,
                               call_node_to_func_graphs);
    }
  } else if (AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimMakeTuple)) {
    // If node is a maketuple node, the real parameters are its total inputs.
    const auto &real_input = FetchRealInputNode(node_with_index);
    MS_LOG(DEBUG) << "Real input node:" << real_input.first->DebugString() << " index:" << real_input.second
                  << " for tuple node:" << node_with_index.first->DebugString() << " index:" << node_with_index.second;
    FetchRealParameterByNode(real_input, real_parameters, invalid_call_nodes, call_node_to_func_graphs);
  } else if (AnfAlgo::CheckPrimitiveType(node.first, prim::kPrimSwitch)) {
    // If node is a switch node, the real parameters are its both true and false branches.
    const auto cnode = node_with_index.first->cast<CNodePtr>();
    const auto inputs = cnode->inputs();
    for (size_t i = kSwitchTrueBranchPos; i < inputs.size(); ++i) {
      FetchRealParameterByNode({inputs[i], 0}, real_parameters, invalid_call_nodes, call_node_to_func_graphs);
    }
  } else if (AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimSwitchLayer)) {
    // If node is a switchlyaer node, the real parameters are its total branches.
    const auto &switch_layer_cnode = node_with_index.first->cast<CNodePtr>();
    const auto &switch_layer_inputs = switch_layer_cnode->inputs();
    if (switch_layer_inputs.size() != kSwitchLayerInputNum ||
        (!AnfAlgo::CheckPrimitiveType(switch_layer_inputs[kSwitchLayerBranchPos], prim::kPrimMakeTuple))) {
      MS_LOG(EXCEPTION) << "Invalid switch layer node:" << switch_layer_cnode->DebugString();
    }
    const auto &make_tuple_cnode = switch_layer_inputs[kSwitchLayerBranchPos]->cast<CNodePtr>();
    const auto &make_tuple_inputs = make_tuple_cnode->inputs();
    for (size_t i = kSwitchTrueBranchPos; i < make_tuple_inputs.size(); ++i) {
      FetchRealParameterByNode({make_tuple_inputs[i], 0}, real_parameters, invalid_call_nodes,
                               call_node_to_func_graphs);
    }
  } else {
    // If node is a kernel, the real parameter is itself.
    real_parameters->emplace(node_with_index);
  }
}

// Fetch all the weight parameters related to node. It runs like this:
// if we have a map like {{a, {b, c}}, {b, {d, e}}}, final we will get {{a, {b, c, d, e}}, {b, {c, d}}}.
void FetchWeightbyHostParameter(const AnfNodePtr &node, std::set<AnfNodePtr> *dest_nodes,
                                const RealToFormalParameter &front_to_front_weight) {
  if (dest_nodes->find(node) != dest_nodes->end()) {
    return;
  }
  dest_nodes->emplace(node);
  auto iter = front_to_front_weight.find({node, 0});
  if (iter == front_to_front_weight.end()) {
    return;
  }

  const auto weight_nodes = iter->second;
  for (const auto weight_node : weight_nodes) {
    FetchWeightbyHostParameter(weight_node.first, dest_nodes, front_to_front_weight);
  }
}

// Recursive interface, get the real kernel that UpdateState node depends on.
AnfNodePtr FetchSourceNodeByAutoMonad(const AnfNodePtr &node) {
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimUpdateState)) {
    const auto &cnode = node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs.size() <= kUpdateStateRealInput) {
      MS_LOG(EXCEPTION) << "Invalid updatestate node:" << AnfAlgo::GetNodeDebugString(node);
    }

    return FetchSourceNodeByAutoMonad(inputs[kUpdateStateRealInput]);
  }
  return node;
}

// Topologically sort all funcgraphs according to the function call relationship.
std::vector<FuncGraphPtr> TopoSortForFuncGraph(const FuncGraphPtr &root, FuncGraphCallRelation *edges) {
  MS_EXCEPTION_IF_NULL(root->manager());
  std::set<FuncGraphPtr> nodes;
  nodes.emplace(root);

  FuncGraphSet subs = root->manager()->func_graphs();
  for (auto sub : subs) {
    if (sub != root && root != nullptr) {
      nodes.emplace(sub);
    }
  }

  std::queue<FuncGraphPtr> que;
  for (const auto &node : nodes) {
    if (edges->find(node) == edges->end()) {
      que.push(node);
    }
  }

  std::vector<FuncGraphPtr> result;
  while (!que.empty()) {
    const auto node = que.front();
    que.pop();
    result.emplace_back(node);
    for (auto iter = edges->begin(); iter != edges->end();) {
      auto &sub_edges = iter->second;
      for (auto sub_iter = sub_edges.begin(); sub_iter != sub_edges.end();) {
        if (sub_iter->find(node) != sub_iter->end()) {
          sub_iter = sub_edges.erase(sub_iter);
        } else {
          ++sub_iter;
        }
      }
      if (sub_edges.empty()) {
        que.push(iter->first);
        iter = edges->erase(iter);
      } else {
        ++iter;
      }
    }
  }

  return result;
}

// Fetch all output of node, and this function will not parse the call node.
std::vector<KernelWithIndex> FetchAllOutputWithIndex(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<KernelWithIndex> result;

  if (node->isa<ValueNode>() && IsValueNode<ValueTuple>(node)) {
    const auto &value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    const auto &value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    const auto &value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    const auto tuple_value = value_tuple->value();
    for (size_t i = 0; i < tuple_value.size(); ++i) {
      result.emplace_back(node, i);
    }
    return result;
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; ++i) {
    const auto node_with_index = AnfAlgo::VisitKernelWithReturnType(node, i);
    if (AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimMakeTuple)) {
      const auto &cnode = node_with_index.first->cast<CNodePtr>();
      const auto &inputs = cnode->inputs();
      const auto &tmp_list = FetchAllOutputWithIndex(inputs[i + kMakeTupleInputStartPos]);
      result.insert(result.end(), tmp_list.begin(), tmp_list.end());
    } else if (AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimSwitch) ||
               AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimSwitchLayer)) {
    } else if (AnfAlgo::IsCallNode(node_with_index.first)) {
      result.emplace_back(node_with_index.first, i);
    } else {
      result.emplace_back(node_with_index);
    }
  }

  return result;
}

// Create a device tensor for the front node.
// Get the output format and select kernel build info from the backend node corresponding to the front node to
// create the device address.
void CreateDeviceTensorForValueNode(const KernelWithIndex &front_node_with_index, const AnfNodePtr &backend_node,
                                    const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &front_node = front_node_with_index.first;
  MS_EXCEPTION_IF_NULL(front_node);

  const auto &node_value = front_node->cast<ValueNodePtr>()->value();
  if (node_value->isa<FuncGraph>() || node_value->isa<Primitive>()) {
    return;
  }

  size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(backend_node, 0);
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(backend_node, 0);
  if (output_type_id == kTypeUnknown) {
    output_type_id = AnfAlgo::GetOutputInferDataType(backend_node, 0);
  }

  if (front_node->kernel_info() == nullptr) {
    front_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  }

  // Get the select kernel build info.
  auto kernel_info = static_cast<device::KernelInfo *>(backend_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(build_info);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, front_node.get());

  // Create device tensor.
  std::string output_format = AnfAlgo::GetOutputFormat(backend_node, 0);
  device::DeviceAddressPtr address =
    device_context->CreateDeviceAddress(nullptr, tensor_size, output_format, output_type_id);
  MS_EXCEPTION_IF_NULL(address);
  MS_LOG(DEBUG) << "Create addr for node:" << AnfAlgo::GetNodeDebugString(front_node) << " addr:" << address;
  AnfAlgo::SetOutputAddr(address, front_node_with_index.second, front_node.get());
  UpdateRefCount(address.get(), true);
}

// Create a device tensor for front node.
// When the condition input of the switch and switchlayer or the output of a subgraph is a parameter or value node,
// there is no corresponding backend node for this parameter, so a device tensor needs to be created for it.
void CreateDeviceTensorForFrontNode(const KernelWithIndex &front_node_with_index, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &node = front_node_with_index.first;
  MS_EXCEPTION_IF_NULL(device_context);

  TypeId type_id = AnfAlgo::GetOutputInferDataType(node, 0);
  if (node->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
    builder->SetOutputsFormat({kOpFormat_DEFAULT});
    builder->SetOutputsDeviceType({type_id});
    kernel_info->set_select_kernel_build_info(builder->Build());
    node->set_kernel_info(kernel_info);
  }
  size_t size = AnfAlgo::GetOutputTensorMemSize(node, 0);

  // Create device tensor.
  device::DeviceAddressPtr address = device_context->CreateDeviceAddress(nullptr, size, kOpFormat_DEFAULT, type_id);
  MS_EXCEPTION_IF_NULL(address);
  MS_LOG(DEBUG) << "Create addr for node:" << AnfAlgo::GetNodeDebugString(node) << " addr:" << address;
  AnfAlgo::SetOutputAddr(address, front_node_with_index.second, node.get());
  UpdateRefCount(address.get(), true);
}

// Fetch all funcgraph by a seed graph, if a calls b, b calls c, and c calls a, return a set of a, b, c.
void FetchAllExecutionFunction(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *checked_funcgraphs,
                               const std::unordered_map<FuncGraphPtr, std::set<FuncGraphPtr>> &call_relation) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (checked_funcgraphs->find(func_graph) != checked_funcgraphs->end()) {
    return;
  }
  checked_funcgraphs->emplace(func_graph);
  auto iter = call_relation.find(func_graph);
  if (iter == call_relation.end()) {
    return;
  }

  for (const auto &called_func_graph : iter->second) {
    MS_EXCEPTION_IF_NULL(called_func_graph);
    FetchAllExecutionFunction(called_func_graph, checked_funcgraphs, call_relation);
  }
}

// Fetch all inputs of node.
std::vector<KernelWithIndex> FetchInputNodeByNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (HasAbstractMonad(node)) {
    return {};
  }

  // The node is divided into the following types:
  // 1. depend and load.
  const auto &node_with_index =
    AnfAlgo::VisitKernelWithReturnType(node, 0, false, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple});
  auto real_node = node_with_index.first;
  size_t real_index = node_with_index.second;
  MS_EXCEPTION_IF_NULL(real_node);
  std::vector<KernelWithIndex> results;
  // 2. MakeTuple.
  if (AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimMakeTuple)) {
    const auto &make_tuple_cnode = real_node->cast<CNodePtr>();
    const auto &make_tuple_inputs = make_tuple_cnode->inputs();
    for (size_t i = kMakeTupleInputStartPos; i < make_tuple_inputs.size(); ++i) {
      const auto &sub_results = FetchInputNodeByNode(make_tuple_inputs[i]);
      results.insert(results.end(), sub_results.begin(), sub_results.end());
    }
    return results;
  }

  // 3. kPrimMakeCSRTensor.
  if (IsCsrNode(real_node)) {
    const auto &cnode = real_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs.size() <= kMakeCSRTensorInputStartPos) {
      MS_LOG(EXCEPTION) << "Invalid make csr tensor node:" << cnode->DebugString();
    }

    // Fetch output put index.
    const auto &prim_node = inputs[0]->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_node);
    const auto &prim_value = prim_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(prim_value);
    const auto &src_node = inputs[kMakeCSRTensorInputStartPos];
    MS_EXCEPTION_IF_NULL(src_node);

    const auto iter = sparse_attr_map.find(prim_value->name());
    // Csr node from the make csr tensor node.
    if (AnfAlgo::CheckPrimitiveType(src_node, prim::kPrimMakeCSRTensor)) {
      const auto &make_csr_tensor_cnode = src_node->cast<CNodePtr>();
      const auto &csr_tensor_inputs = make_csr_tensor_cnode->inputs();
      if (csr_tensor_inputs.size() <= kMakeCSRTensorInputNum) {
        MS_LOG(EXCEPTION) << "Invalid make csr tensor node:" << cnode->DebugString();
      }
      const auto &sub_results = FetchInputNodeByNode(csr_tensor_inputs[iter->second + kMakeCSRTensorInputStartPos]);
      results.insert(results.end(), sub_results.begin(), sub_results.end());
    } else {
      // Csr node from parameter or call node.
      auto abstract = src_node->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      auto real_indexs = FetchRealIndexByAbstract(abstract, iter->second);
      (void)std::transform(real_indexs.begin(), real_indexs.end(), std::back_inserter(results),
                           [&src_node](const auto &index) { return KernelWithIndex(src_node, index); });
    }
    return results;
  }

  // 4. One output node.
  const auto &abstract = real_node->abstract();
  if (abstract == nullptr) {
    MS_LOG(WARNING) << "Empty abstract for node:" << real_node->DebugString();
    results.emplace_back(AnfAlgo::VisitKernelWithReturnType(real_node, real_index));
    return results;
  }

  // 5 Other.
  size_t output_num = AnfAlgo::GetOutputNumByAbstract(abstract);
  if (AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimTupleGetItem)) {
    const auto &get_item_cnode = real_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(get_item_cnode);
    const auto &get_item_src_node = AnfAlgo::GetTupleGetItemRealInput(get_item_cnode);
    size_t get_item_src_index = AnfAlgo::GetTupleGetItemOutIndex(get_item_cnode);

    // Input node of getitm is a make tuple.
    if (AnfAlgo::CheckPrimitiveType(get_item_src_node, prim::kPrimMakeTuple)) {
      const auto &make_tuple_cnode = get_item_src_node->cast<CNodePtr>();
      const auto &makt_tuple_inputs = make_tuple_cnode->inputs();
      if (makt_tuple_inputs.size() <= get_item_src_index) {
        MS_LOG(EXCEPTION) << "Invalid index:" << get_item_src_index
                          << " for make tuple node : " << get_item_src_node->DebugString();
      }
      const auto &sub_results = FetchInputNodeByNode(makt_tuple_inputs[get_item_src_index + kMakeTupleInputStartPos]);
      results.insert(results.end(), sub_results.begin(), sub_results.end());
    } else {
      // Input node of getitm is a parameter or make tuple.
      auto get_item_src_abstract = get_item_src_node->abstract();
      MS_EXCEPTION_IF_NULL(get_item_src_abstract);
      auto real_indexs = FetchRealIndexByAbstract(get_item_src_abstract, get_item_src_index);
      (void)std::transform(
        real_indexs.begin(), real_indexs.end(), std::back_inserter(results),
        [&get_item_src_node](const auto &index) { return KernelWithIndex(get_item_src_node, index); });
    }
  } else {
    for (size_t i = 0; i < output_num; ++i) {
      results.emplace_back(real_node, i);
    }
  }
  return results;
}

// Add formal parameter and real parameter into realationship map.
void AddFormalToRealParameter(const AnfNodePtr &formal_parameter, const AnfNodePtr &real_parameter,
                              const CallNodeToFuncGraph &call_node_to_func_graphs,
                              FormalToRealParameter *formal_to_real_parameters) {
  MS_EXCEPTION_IF_NULL(formal_parameter);
  auto abstract = formal_parameter->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  size_t output_num = AnfAlgo::GetOutputNumByAbstract(abstract);

  for (size_t i = 0; i < output_num; ++i) {
    std::set<KernelWithIndex> real_parameters;
    std::set<KernelWithIndex> invalid_call_nodes;
    FetchRealParameterByNode({real_parameter, i}, &real_parameters, &invalid_call_nodes, call_node_to_func_graphs);
    if (real_parameters.empty()) {
      MS_LOG(EXCEPTION) << "Failed to find real parameter for formal parameter:" << real_parameter->DebugString();
    }

    (*formal_to_real_parameters)[{formal_parameter, i}].insert(real_parameters.begin(), real_parameters.end());
  }
}
}  // namespace

bool HasAbstractRef(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto &abs = node->abstract();
  return (abs != nullptr) && abs->isa<abstract::AbstractRef>();
}

bool IsCsrNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return AnfAlgo::CheckPrimitiveType(node, prim::kPrimCSRTensorGetIndptr) ||
         AnfAlgo::CheckPrimitiveType(node, prim::kPrimCSRTensorGetIndices) ||
         AnfAlgo::CheckPrimitiveType(node, prim::kPrimCSRTensorGetValues) ||
         AnfAlgo::CheckPrimitiveType(node, prim::kPrimCSRTensorGetDenseShape);
}

KernelWithIndex GetFrontNodeByKernelGraph(const AnfNodePtr &backend_node, KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &front_node = graph->GetFrontAnfByBackendAnf(backend_node);
  if (front_node != nullptr) {
    return {front_node, 0};
  }
  const auto &front_node_with_index = graph->GetFrontNodeByInternalParameter(backend_node);
  if (front_node_with_index.first != nullptr) {
    return front_node_with_index;
  }
  const auto &front_tuple_node_with_index = graph->GetElementInTupleBackendFrontIndexMap(backend_node);
  if (front_tuple_node_with_index.first == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot find front node for backend node:" << backend_node->DebugString()
                      << " in graph:" << graph->ToString();
  }
  return front_tuple_node_with_index;
}

std::vector<KernelWithIndex> FetchInputNodeByCNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return {};
  }

  std::vector<KernelWithIndex> results;
  // The first input of normal cnode is the primitive of node, and the real input starts from the second input,
  // but in control flow, the call node has no primitive, and the 0th input is funcgraph or partial.
  size_t input_start_pos = kCNodeInputStartPos;
  if (AnfAlgo::IsCallNode(node)) {
    input_start_pos = 0;
  }
  const auto &cnode = node->cast<CNodePtr>();
  const auto inputs = cnode->inputs();

  // The first branch of the input of the switch node is the true branch, and the second is the false branch.
  // But in switch actor, since the false value is 0, it corresponds to the first branch. Therefore, the input
  // of the switch node needs to exchange the positions of the two branches. So deal separately.
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch)) {
    if (inputs.size() != kSwitchInputNum) {
      MS_LOG(EXCEPTION) << "Invalid switch node:" << node->DebugString();
    }
    results.emplace_back(AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchCondPos], 0));
    results.emplace_back(AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchFalseBranchPos], 0));
    results.emplace_back(AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchTrueBranchPos], 0));
    return results;
  }

  for (size_t i = input_start_pos; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    // skip monad node.
    if (HasAbstractMonad(inputs[i])) {
      continue;
    }
    const auto &sub_results = FetchInputNodeByNode(inputs[i]);
    results.insert(results.end(), sub_results.begin(), sub_results.end());
  }
  return results;
}

abstract::AbstractBasePtr FetchAbstractByIndex(const AbstractBasePtr &abstract, size_t index) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractCSRTensor>()) {
    auto csr_abs = abstract->cast<abstract::AbstractCSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_abs);
    if (index == kCsrTensorIndPtrIndex) {
      return csr_abs->indptr();
    } else if (index == kCsrTensorIndicesIndex) {
      return csr_abs->indices();
    } else if (index == kCsrTensorValuesIndex) {
      return csr_abs->values();
    } else if (index >= kCsrTensorDenseShapeIndex) {
      return FetchAbstractByIndex(csr_abs->dense_shape(), index - kCsrTensorDenseShapeIndex);
    } else {
      MS_LOG(EXCEPTION) << "Invalid index:" << index << " for abstract:" << abstract->ToString();
    }
  }

  if (!abstract->isa<abstract::AbstractTuple>()) {
    if (index != 0) {
      MS_LOG(EXCEPTION) << "Invalid abstract index:" << index << " for abstract:" << abstract->ToString();
    }
    return abstract;
  }

  auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  const auto &sub_abstracts = tuple_abstract->elements();
  size_t real_index = index;
  for (const auto &sub_abstract : sub_abstracts) {
    size_t tmp_index = AnfAlgo::GetOutputNumByAbstract(sub_abstract);
    if (real_index >= tmp_index) {
      real_index -= tmp_index;
      continue;
    }
    return FetchAbstractByIndex(sub_abstract, real_index);
  }
  MS_LOG(EXCEPTION) << "Invalid abstract index:" << index << " for abstract:" << abstract->ToString();
  return nullptr;
}

void ControlNodeParser::Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
                              const std::vector<DeviceContext *> &device_contexts, const FuncGraphPtr &root_graph,
                              const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs) {
  if (graphs.size() != device_contexts.size()) {
    MS_LOG(EXCEPTION) << "Graph num is not equal to device context, graph:" << graphs.size()
                      << " device context num:" << device_contexts.size();
  }

  if (control_nodes.size() <= 1 || device_contexts.empty()) {
    return;
  }
  KernelGraphToDeviceContext kernel_graph_to_device_contexts;
  for (size_t i = 0; i < graphs.size(); ++i) {
    kernel_graph_to_device_contexts[graphs[i]] = device_contexts[i];
  }

  is_inited_ = true;

  root_func_graph_ = root_graph;

  root_graph_parameters_ = root_graph->parameters();

  func_graph_to_kernel_graph_groups_ = func_graph_to_kernel_graphs;
  for (const auto &func_graph_to_kernel_graph_groups : func_graph_to_kernel_graph_groups_) {
    for (const auto &kernel_graph_group : func_graph_to_kernel_graph_groups.second) {
      for (const auto &kernel_graph : kernel_graph_group) {
        MS_LOG(DEBUG) << "Funcgraph to kernel graph, func:" << func_graph_to_kernel_graph_groups.first->ToString()
                      << " kernel_graph:" << kernel_graph->ToString();
      }
    }
  }

  CreateBranchIDForCallNode(control_nodes);

  ParseFrontNodeToKernelGraph(graphs);

  ParseCallNodeToFuncGraph(control_nodes);

  ParseUnRecursionCallNode();

  ParseNeedStackKernelGraph(kernel_graph_to_device_contexts);

  ParseNeedStackControlNode(control_nodes);

  ParseFormalToRealParameter(control_nodes);

  ParseFrontToBackendParameter(graphs, device_contexts);

  CreateDeviceTensorForRootGraphParameter(device_contexts[0]);

  FetchFrontToBackendKernel(graphs, device_contexts);

  FetchHostParameterToWeight();

  ParseDeviceContext(control_nodes, graphs, device_contexts, func_graph_to_kernel_graphs);

  FetchFrontValueNode(control_nodes, device_contexts[0]);

  ParseControlNodeParameter(control_nodes);

  FetchAutoMonadNode(control_nodes);

  ParseFirstControlNodeForFuncGraph(control_nodes);
}

bool ControlNodeParser::IsControlFlowDataArrow(const KernelGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  // Has no control flow node.
  if (!IsInited()) {
    return false;
  }

  if (graph->is_executing_sink()) {
    return true;
  }

  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<Parameter>()) {
    return false;
  }
  auto parameter_node = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter_node);

  // Parameter input should be linked to its entrance actor.
  auto front_node = graph->GetFrontAnfByBackendAnf(node);
  auto internal_node_with_index = graph->GetFrontNodeByInternalParameter(node);
  front_node = (front_node != nullptr ? front_node : internal_node_with_index.first);
  if (front_node == nullptr) {
    auto front_node_with_index = graph->GetElementInTupleBackendFrontIndexMap(node);
    front_node = front_node_with_index.first;
  }

  // If parameter is a weight node, it should be set to kernel actor directly.
  if (AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()) ||
      (front_node != nullptr && front_node->isa<Parameter>() &&
       AnfAlgo::IsParameterWeight(front_node->cast<ParameterPtr>()))) {
    return false;
  }

  // If the graph has a call input, all of its inputs in the graph should be linked to its stack actor.
  if (IsCallInputKernelGraph(graph.get())) {
    // If the input come from a kernel graph belong the same group, it should be linked by internal parameter.
    if (front_node != nullptr && IsSameKernelGraphGroup(front_node, graph)) {
      return false;
    }
    return true;
  }

  return (front_node != nullptr && front_node->isa<Parameter>());
}

bool ControlNodeParser::IsRootGraphParameter(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return find(root_graph_parameters_.begin(), root_graph_parameters_.end(), node) != root_graph_parameters_.end();
}

bool ControlNodeParser::IsRecursionCallNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsCallNode(node)) {
    return false;
  }
  return (find(unrecursion_call_nodes_.begin(), unrecursion_call_nodes_.end(), node) ==
          unrecursion_call_nodes_.end()) ||
         (need_stack_control_nodes_.find(node) != need_stack_control_nodes_.end());
}

bool ControlNodeParser::IsRecursionKernelGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto group_info_iter = kernel_graphs_to_group_info_.find(graph);
  if (group_info_iter == kernel_graphs_to_group_info_.end()) {
    MS_LOG(EXCEPTION) << "Invalid kernel graph:" << graph->ToString();
  }
  MS_EXCEPTION_IF_NULL(group_info_iter->second);
  if (!group_info_iter->second->is_call_input_) {
    return false;
  }
  for (const auto &front_input_node : group_info_iter->second->front_input_nodes_) {
    const auto &node = front_input_node.first.first;
    MS_EXCEPTION_IF_NULL(node);
    if (IsRecursionCallNode(node)) {
      return true;
    }
  }
  return false;
}

bool ControlNodeParser::IsSameKernelGraphGroup(const AnfNodePtr &node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  if (!node->isa<CNode>()) {
    MS_LOG(DEBUG) << "Not a cnode:" << node->DebugString();
    return false;
  }

  const auto node_graph = FetchKernelGraphByFrontNode(node);
  if (node_graph == nullptr) {
    MS_LOG(DEBUG) << "Fail to get kernel graph for cnode:" << node->DebugString();
    return false;
  }
  MS_LOG(DEBUG) << "Get kernel graph:" << node_graph->ToString() << " for cnode:" << node->DebugString()
                << " compare to graph:" << graph->ToString();
  const auto iter1 = kernel_graphs_to_group_info_.find(node_graph);
  const auto iter2 = kernel_graphs_to_group_info_.find(graph);

  return iter1 != kernel_graphs_to_group_info_.end() && iter2 != kernel_graphs_to_group_info_.end() &&
         iter1->second == iter2->second;
}

void ControlNodeParser::ParseDeviceContext(const std::vector<AnfNodePtr> &control_nodes,
                                           const std::vector<KernelGraphPtr> &kernel_graphs,
                                           const std::vector<DeviceContext *> &device_contexts,
                                           const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs) {
  if (device_contexts.empty()) {
    MS_LOG(EXCEPTION) << "Invalid device contexts.";
  }

  ParseDeviceContextForFuncGraph(control_nodes, kernel_graphs, device_contexts, func_graph_to_kernel_graphs);
  ParseDeviceContextForReturnNode(device_contexts[0]);
  ParseDeviceContextForCallNode(control_nodes);
  ParseDeviceContextForPartialNode(control_nodes);
}

void ControlNodeParser::ParseDeviceContextForFuncGraph(const std::vector<AnfNodePtr> &control_nodes,
                                                       const std::vector<KernelGraphPtr> &kernel_graphs,
                                                       const std::vector<DeviceContext *> &device_contexts,
                                                       const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs) {
  mindspore::HashMap<KernelGraphPtr, DeviceContext *> kernel_graph_to_device_context;
  for (size_t i = 0; i < kernel_graphs.size(); ++i) {
    kernel_graph_to_device_context[kernel_graphs[i]] = device_contexts[i];
  }
  const auto &default_context = device_contexts[0];

  // Collect the device context type of the parameter in the kernel graph as the type of the real parameters.
  for (const auto &func_graph_to_kernel_graph : func_graph_to_kernel_graphs) {
    const auto &func_graph = func_graph_to_kernel_graph.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    std::vector<KernelWithIndex> front_parameters;
    for (const auto &parameter : func_graph->parameters()) {
      const auto &abstract = parameter->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      for (size_t i = 0; i < AnfAlgo::GetOutputNumByAbstract(abstract); ++i) {
        front_parameters.emplace_back(parameter, i);
      }
    }
    std::vector<const DeviceContext *> parameter_device_contexts(front_parameters.size(), default_context);
    std::map<KernelWithIndex, DeviceContext *> front_parameter_to_device_context;

    for (const auto &kernel_graph_group : func_graph_to_kernel_graph.second) {
      for (const auto &kernel_graph : kernel_graph_group) {
        const auto &backend_parameters = kernel_graph->parameters();

        for (const auto &backend_parameter : backend_parameters) {
          auto front_parameter = KernelWithIndex(kernel_graph->GetFrontAnfByBackendAnf(backend_parameter), 0);
          if (front_parameter.first == nullptr) {
            front_parameter = kernel_graph->GetElementInTupleBackendFrontIndexMap(backend_parameter);
          }
          if (front_parameter.first != nullptr && front_parameter.first->isa<Parameter>()) {
            front_parameter_to_device_context[front_parameter] = kernel_graph_to_device_context[kernel_graph];
          }
        }
      }
    }

    for (size_t i = 0; i < front_parameters.size(); ++i) {
      const auto &front_parameter = front_parameters[i];
      const auto &iter = front_parameter_to_device_context.find(front_parameter);
      if (iter != front_parameter_to_device_context.end()) {
        parameter_device_contexts[i] = iter->second;
      }
    }
    func_graph_to_device_contexts_[func_graph] = parameter_device_contexts;
  }

  // If there is no kernel in funcgraph, the parameter uses the default device context type.
  FuncGraphSet sub_graphs = root_func_graph_->manager()->func_graphs();
  for (auto sub_graph : sub_graphs) {
    if (func_graph_to_device_contexts_.find(sub_graph) == func_graph_to_device_contexts_.end()) {
      func_graph_to_device_contexts_[sub_graph] =
        std::vector<const DeviceContext *>(sub_graph->parameters().size(), default_context);
    }
  }
}

void ControlNodeParser::ParseDeviceContextForPartialNode(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    if (!AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial)) {
      continue;
    }

    MS_EXCEPTION_IF_NULL(control_node);
    const auto &cnode = control_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    if (inputs.size() <= kPartialFuncGraphPos) {
      MS_LOG(EXCEPTION) << "Invalid input size for partial node:" << cnode->DebugString();
    }

    // Fetch the funcgraph in partial node.
    const auto &func_graph_node = inputs[kPartialFuncGraphPos];
    MS_EXCEPTION_IF_NULL(func_graph_node);
    if ((!func_graph_node->isa<ValueNode>()) || (!IsValueNode<FuncGraph>(func_graph_node))) {
      MS_LOG(EXCEPTION) << "Invalid funcgraph node:" << func_graph_node->DebugString()
                        << " for partial node:" << cnode->DebugString();
    }
    const auto &func_graph = GetValueNode<FuncGraphPtr>(func_graph_node);
    MS_EXCEPTION_IF_NULL(func_graph);

    // Fetch the device contexts for the formal parameters in the funcgraph of partial node.
    auto iter = func_graph_to_device_contexts_.find(func_graph);
    if (iter == func_graph_to_device_contexts_.end()) {
      MS_LOG(EXCEPTION) << "Failed to get device contexts for funcgraph:" << func_graph->ToString();
    }

    // Get the device contexts for the real parameters.
    std::vector<const DeviceContext *> device_contexts;
    // In partial node, the first input is always a partial, maybe a funcgraph or a partial node, so we need
    // to insert an empty device context for it.
    device_contexts.emplace_back(nullptr);
    for (size_t i = 0; i < inputs.size() - kPartialInputStartPos; ++i) {
      if (i >= iter->second.size()) {
        MS_LOG(EXCEPTION) << "Invalid device context index:" << i << " for funcgraph:" << func_graph->ToString()
                          << " device context size:" << iter->second.size()
                          << " for partial node:" << cnode->DebugString();
      }
      MS_EXCEPTION_IF_NULL(iter->second[i]);
      device_contexts.emplace_back(iter->second[i]);
    }
    control_node_to_device_contexts_[control_node] = device_contexts;
  }
}

void ControlNodeParser::ParseDeviceContextForCallNode(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (!AnfAlgo::IsCallNode(control_node)) {
      continue;
    }

    // Fetch the device contexts of the funcgraph the node called.
    const auto &func_graphs = FetchFuncGraphbyCallNode(control_node);
    if (func_graphs.empty()) {
      MS_LOG(EXCEPTION) << "Failed to get funcgraph by call node:" << control_node->DebugString();
    }
    const auto &func_graph = *(func_graphs.begin());
    MS_EXCEPTION_IF_NULL(func_graph);
    auto iter = func_graph_to_device_contexts_.find(func_graph);
    if (iter == func_graph_to_device_contexts_.end()) {
      MS_LOG(EXCEPTION) << "Failed to get device contexts for funcgraph:" << func_graph->ToString();
    }

    std::vector<const DeviceContext *> device_contexts;
    // In call node, the first input is always a partial, maybe a funcgraph or a partial node, so we need
    // to insert an empty device context for it.
    device_contexts.emplace_back(nullptr);
    const auto &cnode = control_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    size_t call_input_num = 0;
    for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
      const auto &abstract = inputs[i]->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      call_input_num += AnfAlgo::GetOutputNumByAbstract(abstract);
    }

    if (call_input_num > iter->second.size()) {
      MS_LOG(EXCEPTION) << "Invalid input size:" << call_input_num << " context size:" << iter->second.size()
                        << "for funcgraph" << func_graph->ToString() << " for call node:" << cnode->DebugString();
    }

    // Fetch the device contexts for the real parameters on the call node.
    for (size_t i = iter->second.size() - call_input_num; i < iter->second.size(); ++i) {
      if (i >= iter->second.size()) {
        MS_LOG(EXCEPTION) << "Invalid device context index:" << i << " for funcgraph:" << func_graph->ToString()
                          << " device context size:" << iter->second.size()
                          << " for partial node:" << cnode->DebugString();
      }
      MS_EXCEPTION_IF_NULL(iter->second[i]);
      device_contexts.emplace_back(iter->second[i]);
    }
    control_node_to_device_contexts_[control_node] = device_contexts;
  }
}

void ControlNodeParser::ParseDeviceContextForReturnNode(const DeviceContext *default_context) {
  MS_EXCEPTION_IF_NULL(default_context);
  // Collect the call realationship between funcgraphs.
  FuncGraphCallRelation func_graph_call_relation;
  for (const auto &call_node_to_func_graphs : call_node_to_func_graphs_) {
    const auto &call_node = call_node_to_func_graphs.first;
    MS_EXCEPTION_IF_NULL(call_node);
    const auto &func_graph = call_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    func_graph_call_relation[func_graph].emplace_back(call_node_to_func_graphs.second);
  }

  // Topologically sort all funcgraphs according to the function call relationship.
  const auto &topo_sort_func_graphs = TopoSortForFuncGraph(root_func_graph_, &func_graph_call_relation);

  // Deduces the device context type of funcgraph outputs according to the topological order.
  for (const auto &func_graph : topo_sort_func_graphs) {
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &return_node = func_graph->return_node();
    MS_EXCEPTION_IF_NULL(return_node);
    const auto &cnode = return_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    const auto output_nodes = FetchInputNodeByNode(inputs[kReturnInputPos]);
    std::vector<const DeviceContext *> return_device_contexts;

    for (const auto &output_node : output_nodes) {
      if (output_node.first->isa<Parameter>()) {
        // If the output is parameter, get the device context type from the formal parameter.
        const auto &iter = find(func_graph->parameters().begin(), func_graph->parameters().end(), output_node.first);
        if (iter == func_graph->parameters().end()) {
          MS_LOG(EXCEPTION) << "Invalid parameter:" << output_node.first->DebugString()
                            << " for func_graph:" << func_graph->ToString();
        }
        const auto &func_graph_iter = func_graph_to_device_contexts_.find(func_graph);
        if (func_graph_iter == func_graph_to_device_contexts_.end()) {
          MS_LOG(EXCEPTION) << "Cannot find device context for funcgraph:" << func_graph->ToString();
        }
        MS_EXCEPTION_IF_NULL(func_graph_iter->second[iter - func_graph->parameters().begin()]);
        return_device_contexts.emplace_back(func_graph_iter->second[iter - func_graph->parameters().begin()]);
      } else if (output_node.first->isa<ValueNode>()) {
        // If the output is parameter, used the default context type.
        MS_EXCEPTION_IF_NULL(default_context);
        return_device_contexts.emplace_back(default_context);
      } else if (AnfAlgo::IsCallNode(output_node.first)) {
        // If the output is call node, get the device context type by the output of funcgraph.
        const auto &func_graphs = call_node_to_func_graphs_[output_node.first];
        std::vector<const DeviceContext *> call_device_contexts;
        for (const auto &graph : func_graphs) {
          MS_EXCEPTION_IF_NULL(graph);
          const auto &node = graph->return_node();
          MS_EXCEPTION_IF_NULL(node);
          const auto &iter = control_node_to_device_contexts_.find(node);
          if (iter != control_node_to_device_contexts_.end()) {
            call_device_contexts = iter->second;
            break;
          }
        }
        // Since funcgraph has been topo-sorted according to the calling relationship, when there is a call node in
        // the output, the output type of the funcgraph called by it should have been determined, if not, an exception
        // will be thrown.
        if (call_device_contexts.empty() || call_device_contexts.size() <= output_node.second) {
          MS_LOG(EXCEPTION) << "Cannot find device context for call node:" << output_node.first->DebugString()
                            << " device contexts size:" << call_device_contexts.size()
                            << " index:" << output_node.second;
        }
        MS_EXCEPTION_IF_NULL(call_device_contexts[output_node.second]);
        return_device_contexts.emplace_back(call_device_contexts[output_node.second]);
      } else if (AnfAlgo::CheckPrimitiveType(output_node.first, prim::kPrimPartial)) {
        return_device_contexts.emplace_back(default_context);
      } else if (output_node.first->isa<CNode>()) {
        // If the output is a cnode, get the device context type by the kernel.
        const auto &iter = front_to_backend_kernels_.find(output_node);
        if (iter == front_to_backend_kernels_.end()) {
          MS_LOG(EXCEPTION) << "Cannot find backend kernel for cnode:" << output_node.first->DebugString();
        }
        MS_EXCEPTION_IF_NULL(iter->second.second);
        return_device_contexts.emplace_back(iter->second.second);
      } else {
        MS_LOG(EXCEPTION) << "Invalid node for return:" << output_node.first->DebugString();
      }
    }
    control_node_to_device_contexts_[return_node] = return_device_contexts;
  }
}

void ControlNodeParser::ParseFrontNodeToKernelGraph(const std::vector<KernelGraphPtr> &graphs) {
  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->execution_order().empty()) {
      continue;
    }
    const auto &front_to_backend_nodes = graph->front_backend_anf_map();
    for (const auto &front_to_backend_node : front_to_backend_nodes) {
      front_node_to_kernel_graph_[front_to_backend_node.first] = graph;
    }
  }
}

int ControlNodeParser::FetchBranchIDByCallNode(const AnfNodePtr &call_node) {
  MS_EXCEPTION_IF_NULL(call_node);

  if (call_node_to_branch_id_.find(call_node) == call_node_to_branch_id_.end()) {
    MS_LOG(EXCEPTION) << "Invalid branch id for call_node:" << call_node->DebugString();
  }
  return call_node_to_branch_id_[call_node];
}

KernelGraphPtr ControlNodeParser::FetchKernelGraphByFrontNode(const AnfNodePtr &kernel) {
  const auto &iter = front_node_to_kernel_graph_.find(kernel);
  if (iter == front_node_to_kernel_graph_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool ControlNodeParser::IsCallInputKernelGraph(KernelGraph *const graph) {
  if (call_input_kernel_graphs_.find(graph) == call_input_kernel_graphs_.end()) {
    return false;
  }
  return true;
}

bool ControlNodeParser::IsCallInputKernelGraphGroup(const std::string &group_name) {
  for (const auto &graph_group : kernel_graph_group_infos_) {
    if (group_name.find(graph_group->group_name_) != std ::string::npos) {
      return graph_group->is_call_input_;
    }
  }
  MS_LOG(EXCEPTION) << "Invalid kernel graph group name:" << group_name;
  return false;
}

KernelWithIndex ControlNodeParser::FetchBackendNodeByFrontNode(const KernelWithIndex &node_with_index) {
  const auto &iter = front_to_backend_kernels_.find(node_with_index);
  if (iter != front_to_backend_kernels_.end()) {
    return iter->second.first;
  }
  return {};
}

FuncGraphPtr ControlNodeParser::FetchFuncGraphByKernelGraph(const KernelGraph *const graph) {
  for (const auto &func_graph_to_kernel_graphs : func_graph_to_kernel_graph_groups_) {
    const auto &kernel_graph_groups = func_graph_to_kernel_graphs.second;
    if (std::any_of(kernel_graph_groups.begin(), kernel_graph_groups.end(), [graph](const auto &kernel_graph_group) {
          return std::any_of(kernel_graph_group.begin(), kernel_graph_group.end(),
                             [graph](const auto &kernel_graph) { return kernel_graph.get() == graph; });
        })) {
      return func_graph_to_kernel_graphs.first;
    }
  }
  return nullptr;
}

void ControlNodeParser::FetchFrontValueNode(const std::vector<AnfNodePtr> &control_nodes,
                                            DeviceContext *default_context) {
  MS_EXCEPTION_IF_NULL(default_context);

  for (const auto &formal_to_real_parameter : formal_to_real_parameters_) {
    for (const auto &real_parameter_with_index : formal_to_real_parameter.second) {
      const auto &real_parameter = real_parameter_with_index.first;
      if (!IsFrontValueNode(real_parameter)) {
        continue;
      }

      const auto &iter = front_to_backend_parameters_.find(real_parameter_with_index);
      if (iter != front_to_backend_parameters_.end() && (!iter->second.empty())) {
        front_value_nodes_.emplace(real_parameter_with_index, iter->second.begin()->second);
        CreateDeviceTensorForValueNode(real_parameter_with_index, iter->second.begin()->first,
                                       iter->second.begin()->second);
      } else {
        front_value_nodes_.emplace(real_parameter_with_index, default_context);
        CreateDeviceTensorForFrontNode(real_parameter_with_index, default_context);
      }
    }
  }

  // If the output of funcgraph is a value node, it will eventually be sent to the kernel as a real parameter.
  // These the value nodes also need to create a device address.
  for (const auto &front_to_backend_parameters : front_to_backend_parameters_) {
    const auto &front_node = front_to_backend_parameters.first.first;
    MS_EXCEPTION_IF_NULL(front_node);
    if (IsFrontValueNode(front_node) && (!front_to_backend_parameters.second.empty())) {
      const auto &backend_parameter = front_to_backend_parameters.second.begin()->first;
      const auto &device_context = front_to_backend_parameters.second.begin()->second;
      CreateDeviceTensorForValueNode(front_to_backend_parameters.first, backend_parameter, device_context);
      front_value_nodes_.emplace(front_to_backend_parameters.first, device_context);
    }
  }

  // Create device tensors for those value nodes which direct return by a return node.
  for (const auto &control_node : control_nodes) {
    if ((!AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) && (!AnfAlgo::IsCallNode(control_node))) {
      continue;
    }

    auto input_with_indexs = FetchInputNodeByCNode(control_node);
    auto iter = control_node_to_device_contexts_.find(control_node);
    if (iter == control_node_to_device_contexts_.end() || iter->second.size() < input_with_indexs.size()) {
      MS_LOG(EXCEPTION) << "Invalid device context for control node:" << control_node->DebugString()
                        << " need:" << input_with_indexs.size() << " current:"
                        << (iter == control_node_to_device_contexts_.end() ? "null"
                                                                           : std::to_string(iter->second.size()));
    }
    for (size_t i = 0; i < input_with_indexs.size(); ++i) {
      const auto &input_with_index = input_with_indexs[i];
      if (IsFrontValueNode(input_with_index.first) &&
          front_value_nodes_.find({input_with_index, iter->second[i]}) == front_value_nodes_.end()) {
        CreateDeviceTensorForFrontNode(input_with_index, iter->second[i]);
        front_value_nodes_.emplace(input_with_index, iter->second[i]);
      }
    }
  }
}

void ControlNodeParser::ParseFormalToRealParameter(const std::vector<AnfNodePtr> &control_nodes) {
  FormalToRealParameter formal_to_real_parameters;

  // The actual parameters of the function are divided into two parts:
  // 1. Input of partial node.
  // 2. Input of call node.
  for (const auto &node : control_nodes) {
    if (AnfAlgo::IsCallNode(node)) {
      const auto &cnode = node->cast<CNodePtr>();
      const auto &inputs = cnode->inputs();
      const auto &func_graphs = FetchFuncGraphbyCallNode(node);
      for (const auto func_graph : func_graphs) {
        const auto &parameters = func_graph->parameters();
        for (size_t i = inputs.size() - 1, j = parameters.size() - 1; i >= kCallInputStartPos && j >= 0; --i, --j) {
          MS_EXCEPTION_IF_NULL(inputs[i]);
          MS_EXCEPTION_IF_NULL(parameters[j]);
          if (HasAbstractMonad(inputs[i])) {
            continue;
          }
          AddFormalToRealParameter(parameters[j], inputs[i], call_node_to_func_graphs_, &formal_to_real_parameters);
        }
      }
    } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      const auto &inputs = cnode->inputs();
      if (inputs.size() <= kPartialFuncGraphPos || (!inputs[kPartialFuncGraphPos]->isa<ValueNode>()) ||
          (!IsValueNode<FuncGraph>(inputs[kPartialFuncGraphPos]))) {
        MS_LOG(EXCEPTION) << "Invalid partial node:" << node->DebugString();
      }
      const auto &func_graph = GetValueNode<FuncGraphPtr>(inputs[kPartialFuncGraphPos]);
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &parameters = func_graph->parameters();
      if (inputs.size() - kPartialInputStartPos > parameters.size()) {
        MS_LOG(EXCEPTION) << "Invalid partial input size:" << inputs.size()
                          << " formal parameter size:" << parameters.size();
      }
      for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
        MS_EXCEPTION_IF_NULL(inputs[i]);
        MS_EXCEPTION_IF_NULL(parameters[i - kPartialInputStartPos]);
        if (HasAbstractMonad(inputs[i])) {
          continue;
        }
        AddFormalToRealParameter(parameters[i - kPartialInputStartPos], inputs[i], call_node_to_func_graphs_,
                                 &formal_to_real_parameters);
      }
    }
  }

  // When the real parameter is also a parameter, the corresponding actual parameter needs to be obtained recursively.
  for (const auto &formal_to_real_parameter : formal_to_real_parameters) {
    const auto &formal_parameter = formal_to_real_parameter.first;
    const auto &real_parameters = formal_to_real_parameter.second;
    std::set<KernelWithIndex> total_real_parameters = real_parameters;
    for (const auto &real_parameter : real_parameters) {
      if (real_parameter.first->isa<Parameter>()) {
        std::set<KernelWithIndex> invalid_real_parameter{formal_parameter};
        ParseAllRealParameterByFormalParameter(real_parameter, formal_to_real_parameters, &total_real_parameters,
                                               &invalid_real_parameter);
        real_to_formal_parameters_[real_parameter].emplace(formal_parameter);
      } else {
        total_real_parameters.emplace(real_parameter);
      }
    }
    std::swap(formal_to_real_parameters_[formal_parameter], total_real_parameters);
  }
}

void ControlNodeParser::ParseAllRealParameterByFormalParameter(const KernelWithIndex &formal_parameter,
                                                               const FormalToRealParameter &formal_to_real_parameters,
                                                               std::set<KernelWithIndex> *total_real_parameters,
                                                               std::set<KernelWithIndex> *invalid_real_parameter) {
  if (invalid_real_parameter->find(formal_parameter) != invalid_real_parameter->end()) {
    return;
  }
  invalid_real_parameter->emplace(formal_parameter);

  // Get all the actual parameters corresponding to parameter recursively.
  const auto &dst_iter = formal_to_real_parameters_.find(formal_parameter);
  if (dst_iter != formal_to_real_parameters_.end()) {
    total_real_parameters->insert(dst_iter->second.begin(), dst_iter->second.end());
    return;
  }
  const auto &src_iter = formal_to_real_parameters.find(formal_parameter);
  if (src_iter == formal_to_real_parameters.end()) {
    const auto &func_graph = formal_parameter.first->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    if (func_graph == root_func_graph_) {
      return;
    }
    MS_LOG(EXCEPTION) << "Invalid formal parameter:" << formal_parameter.first->DebugString();
  }
  const auto &real_parameters = src_iter->second;
  for (const auto &real_parameter : real_parameters) {
    MS_EXCEPTION_IF_NULL(real_parameter.first);
    total_real_parameters->emplace(real_parameter);
    if (real_parameter.first->isa<Parameter>()) {
      ParseAllRealParameterByFormalParameter(real_parameter, formal_to_real_parameters, total_real_parameters,
                                             invalid_real_parameter);
    }
  }
}

void ControlNodeParser::ParseControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    CNodePtr cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      break;
    } else if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial)) {
      for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          (void)control_node_parameters_.emplace_back(inputs[i]);
        }
      }
    } else if (cnode->input(0)->isa<CNode>() || IsValueNode<FuncGraph>(cnode->input(0))) {
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          (void)control_node_parameters_.emplace_back(inputs[i]);
        }
      }
    } else if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch)) {
      if (inputs.size() != kSwitchInputNum) {
        MS_LOG(EXCEPTION) << "Invalid switch node:" << AnfAlgo::GetNodeDebugString(control_node);
      }
      if (inputs[kSwitchCondPos]->isa<Parameter>()) {
        (void)control_node_parameters_.emplace_back(inputs[kSwitchCondPos]);
      }
    } else if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      if (inputs.size() != kSwitchLayerInputNum) {
        MS_LOG(EXCEPTION) << "Invalid switch node:" << AnfAlgo::GetNodeDebugString(control_node);
      }
      if (inputs[kSwitchLayerCondPos]->isa<Parameter>()) {
        (void)control_node_parameters_.emplace_back(inputs[kSwitchLayerCondPos]);
      }
    }
  }
}

void ControlNodeParser::CreateBranchIDForCallNode(const std::vector<AnfNodePtr> &control_nodes) {
  int branch_id = kMainBranchID;

  for (const auto &control_node : control_nodes) {
    // Root funcgraph does not need to create a gather actor.
    if (AnfAlgo::IsCallNode(control_node)) {
      call_node_to_branch_id_[control_node] = ++branch_id;
    }
  }
}

void ControlNodeParser::ParseFrontToBackendParameter(const std::vector<KernelGraphPtr> &graphs,
                                                     const std::vector<DeviceContext *> &device_contexts) {
  if (graphs.size() != device_contexts.size()) {
    MS_LOG(EXCEPTION) << "Graph num is not equal to device context num.";
  }

  // Fetch the mapping relationship between front parameters and backend parameters in the kernel graphs.
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    auto device_context = device_contexts[i];
    for (const auto &parameter : graph->input_nodes()) {
      if (HasAbstractMonad(parameter)) {
        continue;
      }
      const auto &front_node = graph->GetFrontAnfByBackendAnf(parameter);
      const auto &front_node_with_index = graph->GetFrontNodeByInternalParameter(parameter);
      const auto &front_tuple_parameter_with_index = graph->GetElementInTupleBackendFrontIndexMap(parameter);
      if (front_node == nullptr && front_node_with_index.first == nullptr &&
          front_tuple_parameter_with_index.first == nullptr) {
        MS_LOG(EXCEPTION) << "Invalid backend parameter:" << parameter->DebugString()
                          << " for kernel graph:" << graph->ToString();
      }

      if (front_node_with_index.first != nullptr) {
        std::set<KernelWithIndex> real_parameters;
        std::set<KernelWithIndex> invalid_call_nodes;
        FetchRealParameterByNode(front_node_with_index, &real_parameters, &invalid_call_nodes,
                                 call_node_to_func_graphs_);
        for (const auto real_parameter : real_parameters) {
          if (real_parameter.first->isa<Parameter>() || real_parameter.first->isa<ValueNode>()) {
            front_to_backend_parameters_[real_parameter].emplace(parameter, device_context);
          }
        }
      } else if (front_tuple_parameter_with_index.first != nullptr) {
        front_to_backend_parameters_[front_tuple_parameter_with_index].emplace(parameter, device_context);
      } else {
        front_to_backend_parameters_[{front_node, 0}].emplace(parameter, device_context);
      }
    }
  }

  // Get the corresponding backend node for the real parameter according to the relationship between real
  // parameter and formal parameter.
  for (const auto &front_to_backend_parameters : front_to_backend_parameters_) {
    const auto &front_parameter = front_to_backend_parameters.first;
    const auto &backend_parameters = front_to_backend_parameters.second;
    const auto &iter = formal_to_real_parameters_.find(front_parameter);
    if (iter != formal_to_real_parameters_.end()) {
      for (const auto &real_parameter_with_index : iter->second) {
        const auto &real_parameter = real_parameter_with_index.first;
        if (real_parameter->isa<Parameter>()) {
          front_to_backend_parameters_[real_parameter_with_index].insert(backend_parameters.begin(),
                                                                         backend_parameters.end());
        }
      }
    }
  }
}

void ControlNodeParser::ParseCallNodeToFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  auto func_graph_analyzer = std::make_shared<FuncGraphAnalyzer>(root_func_graph_);
  func_graph_analyzer->Run();

  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (!AnfAlgo::IsCallNode(control_node)) {
      continue;
    }

    auto func_graphs = func_graph_analyzer->GetCallerFuncGraphs(control_node);
    for (auto func_graph : func_graphs) {
      call_node_to_func_graphs_[control_node].emplace(func_graph);
    }
  }
}

const std::set<FuncGraphPtr> &ControlNodeParser::FetchFuncGraphbyCallNode(const AnfNodePtr &control_node) {
  const auto &iter = call_node_to_func_graphs_.find(control_node);
  if (iter == call_node_to_func_graphs_.end()) {
    MS_LOG(EXCEPTION) << "Invalid call node:" << control_node->DebugString();
  }
  return iter->second;
}

void ControlNodeParser::FetchHostParameterToWeight() {
  for (const auto &pair : real_to_formal_parameters_) {
    std::set<AnfNodePtr> dest_nodes;
    FetchWeightbyHostParameter(pair.first.first, &dest_nodes, real_to_formal_parameters_);
    host_parameter_to_weights_[pair.first.first] = dest_nodes;

    if (std::find(root_graph_parameters_.begin(), root_graph_parameters_.end(), pair.first.first) !=
        root_graph_parameters_.end()) {
      for (auto &sub_front_node : dest_nodes) {
        sub_front_node_to_root_front_node_[sub_front_node] = pair.first.first;
      }
    }
  }
}

void ControlNodeParser::FetchFrontToBackendKernel(const std::vector<KernelGraphPtr> &graphs,
                                                  const std::vector<DeviceContext *> &device_contexts) {
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    const auto &device_context = device_contexts[i];
    MS_EXCEPTION_IF_NULL(graph);
    auto execution_order = graph->execution_order();
    for (auto &kernel : execution_order) {
      auto front_node = graph->GetFrontAnfByBackendAnf(kernel);
      if (front_node != nullptr) {
        for (size_t j = 0; j < AnfAlgo::GetOutputTensorNum(kernel); ++j) {
          front_to_backend_kernels_[{front_node, j}] = {{kernel, j}, device_context};
          MS_LOG(DEBUG) << "Add front to backend kernel, front:" << AnfAlgo::GetNodeDebugString(front_node)
                        << "index:" << j << " addr:" << front_node << " second:" << AnfAlgo::GetNodeDebugString(kernel)
                        << "index:" << j << " addr:" << kernel;
        }
      }
    }

    const auto graph_output_map = graph->graph_output_map();
    for (const auto &output_pair : graph_output_map) {
      if (output_pair.first.first->isa<CNode>()) {
        front_to_backend_kernels_[output_pair.second] = {output_pair.first, device_context};
      }
    }
  }
}

void ControlNodeParser::FetchAutoMonadNode(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    const auto &cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs.empty()) {
      MS_LOG(EXCEPTION) << "Invalid control node:" << AnfAlgo::GetNodeDebugString(control_node);
    }

    if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (AnfAlgo::CheckPrimitiveType(inputs[i], prim::kPrimUpdateState)) {
          const auto &node = FetchSourceNodeByAutoMonad(inputs[i]);
          const auto &iter = front_to_backend_kernels_.find(AnfAlgo::VisitKernelWithReturnType(node, 0));
          if (iter != front_to_backend_kernels_.end()) {
            kernel_to_call_nodes_[iter->second.first.first] = control_node;
          }
        }
      }
    }
  }
}

AnfNodePtr ControlNodeParser::FetchRootGraphFrontNodeBySubFrontNode(const AnfNodePtr &sub_front_node) {
  if (sub_front_node_to_root_front_node_.count(sub_front_node) == 0) {
    return sub_front_node;
  }
  return sub_front_node_to_root_front_node_[sub_front_node];
}

bool IsFirstControlNode(const AnfNodePtr &node, std::set<AnfNodePtr> *checked_nodes,
                        std::set<AnfNodePtr> unrecursion_call_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(checked_nodes);
  if (!node->isa<CNode>() || checked_nodes->find(node) != checked_nodes->end()) {
    return true;
  }
  checked_nodes->emplace(node);

  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if ((AnfAlgo::IsCallNode(input) && unrecursion_call_nodes.find(input) == unrecursion_call_nodes.end()) ||
        (!IsFirstControlNode(input, checked_nodes, unrecursion_call_nodes))) {
      return false;
    }
  }
  return true;
}

void ControlNodeParser::ParseFirstControlNodeForFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    std::set<AnfNodePtr> checked_nodes;
    if (((AnfAlgo::IsCallNode(control_node) &&
          unrecursion_call_nodes_.find(control_node) == unrecursion_call_nodes_.end()) ||
         AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) &&
        IsFirstControlNode(control_node, &checked_nodes, unrecursion_call_nodes_)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      func_graph_to_first_control_nodes_[func_graph].emplace(control_node);
    }
  }
}

void ControlNodeParser::ParseUnRecursionCallNode() {
  std::unordered_map<FuncGraphPtr, std::set<FuncGraphPtr>> func_graph_call_relation;
  // Collect the call relationship between funcgraphs.
  for (const auto &call_node_to_func_graphs : call_node_to_func_graphs_) {
    const auto &call_node = call_node_to_func_graphs.first;
    MS_EXCEPTION_IF_NULL(call_node);
    const auto &func_graph = call_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    func_graph_call_relation[func_graph].insert(call_node_to_func_graphs.second.begin(),
                                                call_node_to_func_graphs.second.end());
  }

  for (const auto &call_node_to_func_graphs : call_node_to_func_graphs_) {
    const auto &call_node = call_node_to_func_graphs.first;
    const auto &dest_func_graph = call_node->func_graph();
    MS_EXCEPTION_IF_NULL(dest_func_graph);
    std::set<FuncGraphPtr> exexution_func_graphs;
    for (const auto &func_graph : call_node_to_func_graphs.second) {
      FetchAllExecutionFunction(func_graph, &exexution_func_graphs, func_graph_call_relation);
    }
    if (exexution_func_graphs.find(dest_func_graph) == exexution_func_graphs.end()) {
      unrecursion_call_nodes_.emplace(call_node);
      MS_LOG(DEBUG) << "Add unrecursion call control node:" << call_node->DebugString();
    }
  }
}

void ControlNodeParser::ParseNeedStackControlNode(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (!AnfAlgo::IsCallNode(control_node)) {
      continue;
    }
    auto input_with_indexs = FetchInputNodeByCNode(control_node);
    for (const auto &input_with_index : input_with_indexs) {
      MS_EXCEPTION_IF_NULL(input_with_index.first);
      // If the call node has call or recursion graph input, a stack created for the call node is required.
      if (!AnfAlgo::IsCallNode(input_with_index.first)) {
        const auto &graph = FetchKernelGraphByFrontNode(input_with_index.first);
        if (graph == nullptr || (!IsRecursionKernelGraph(graph))) {
          continue;
        }
      }
      need_stack_control_nodes_.emplace(control_node);
      MS_LOG(DEBUG) << "Add need stack control node:" << control_node->DebugString();
      break;
    }
  }

  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      auto input_with_indexs = FetchInputNodeByCNode(control_node);
      size_t call_input_num = 0;
      for (auto input_with_index : input_with_indexs) {
        if (AnfAlgo::IsCallNode(input_with_index.first)) {
          ++call_input_num;
        }
      }

      const auto &cnode = control_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      const auto &inputs = cnode->inputs();
      if (inputs.size() <= kReturnInputPos) {
        MS_LOG(EXCEPTION) << "Invalid return node:" << control_node->DebugString();
      }

      if (call_input_num != 0 && (AnfAlgo::CheckPrimitiveType(inputs[kReturnInputPos], prim::kPrimDepend))) {
        need_stack_control_nodes_.emplace(control_node);
      }
    } else if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial)) {
      auto input_with_indexs = FetchInputNodeByCNode(control_node);
      if (std::any_of(input_with_indexs.begin(), input_with_indexs.end(),
                      [this](const auto &input_with_index) { return IsRecursionCallNode(input_with_index.first); })) {
        need_stack_control_nodes_.emplace(control_node);
        MS_LOG(DEBUG) << "Add need stack control node:" << control_node->DebugString();
      }
    }
  }
}

void ControlNodeParser::ParseNeedStackKernelGraph(const KernelGraphToDeviceContext &kernel_graph_to_device_contexts) {
  for (const auto &func_graph_to_kernel_graph_groups : func_graph_to_kernel_graph_groups_) {
    for (const auto &kernel_graph_group : func_graph_to_kernel_graph_groups.second) {
      if (kernel_graph_group.empty()) {
        continue;
      }

      KernelGraphGroupInfoPtr kernel_graph_group_info = std::make_shared<KernelGraphGroupInfo>();
      for (const auto &kernel_graph : kernel_graph_group) {
        MS_EXCEPTION_IF_NULL(kernel_graph);
        if (kernel_graph->execution_order().empty()) {
          continue;
        }
        auto iter = kernel_graph_to_device_contexts.find(kernel_graph);
        if (iter == kernel_graph_to_device_contexts.end()) {
          MS_LOG(EXCEPTION) << "Failed to find device context for kernel graph:" << kernel_graph->ToString();
        }
        // Collect kernel graphs in group.
        kernel_graph_group_info->graphs_.emplace(kernel_graph);

        // Collect inputs in group.
        const auto &real_parameters = kernel_graph->input_nodes();
        for (const auto &parameter : real_parameters) {
          const auto &front_node_with_index = GetFrontNodeByKernelGraph(parameter, kernel_graph.get());
          MS_EXCEPTION_IF_NULL(front_node_with_index.first);
          // If input come from the output of kernel graph belong the same group, it should not be collected in
          // the group inputs.
          if (HasAbstractMonad(front_node_with_index.first) || HasAbstractMonad(parameter) ||
              kernel_graph_group_info->front_output_nodes_.find(front_node_with_index) !=
                kernel_graph_group_info->front_output_nodes_.end()) {
            continue;
          }
          if (AnfAlgo::IsCallNode(front_node_with_index.first)) {
            kernel_graph_group_info->is_call_input_ = true;
          }
          kernel_graph_group_info->front_input_nodes_[front_node_with_index] = iter->second;
        }

        // Collect outputs in group.
        for (const auto &backend_to_front : kernel_graph->graph_output_map()) {
          if (HasAbstractMonad(backend_to_front.second.first) || HasAbstractMonad(backend_to_front.first.first) ||
              AnfAlgo::CheckPrimitiveType(backend_to_front.second.first, prim::kPrimPartial)) {
            continue;
          }
          MS_LOG(DEBUG) << "Kernel graph:" << kernel_graph->ToString()
                        << " add front output node:" << backend_to_front.second.first->DebugString()
                        << " index:" << backend_to_front.second.second
                        << " backend node:" << backend_to_front.first.first->DebugString()
                        << " index:" << backend_to_front.first.second;
          kernel_graph_group_info->front_output_nodes_[backend_to_front.second] = {backend_to_front.first,
                                                                                   iter->second};
        }

        kernel_graphs_to_group_info_[kernel_graph] = kernel_graph_group_info;
        if (kernel_graph_group_info->is_call_input_) {
          call_input_kernel_graphs_.emplace(kernel_graph.get());
        }
      }
      kernel_graph_group_info->group_name_ = "kernel_graph";
      for (const auto &graph : kernel_graph_group_info->graphs_) {
        kernel_graph_group_info->group_name_ += ("_" + std::to_string(graph->graph_id()));
      }
      kernel_graph_group_infos_.emplace(kernel_graph_group_info);
    }
  }
}

void ControlNodeParser::CreateDeviceTensorForRootGraphParameter(DeviceContext *default_context) {
  MS_EXCEPTION_IF_NULL(default_context);
  for (const auto &parameter : root_graph_parameters_) {
    KernelWithIndex parameter_with_index(parameter, 0);
    if (front_to_backend_parameters_.find(parameter_with_index) == front_to_backend_parameters_.end()) {
      CreateDeviceTensorForFrontNode(parameter_with_index, default_context);
      front_to_backend_parameters_[parameter_with_index].emplace(parameter, default_context);
    }
  }
}

std::string ControlNodeParser::FetchGroupNameByKernelGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto group_info_iter = kernel_graphs_to_group_info_.find(graph);
  if (group_info_iter == kernel_graphs_to_group_info_.end()) {
    MS_LOG(EXCEPTION) << "Failed to get kernel graph group info for graph:" << graph->ToString();
  }
  return group_info_iter->second->group_name_;
}
}  // namespace runtime
}  // namespace mindspore
