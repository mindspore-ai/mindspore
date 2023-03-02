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

#include <unordered_map>
#include <map>
#include "runtime/graph_scheduler/control_node_parser.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "include/common/utils/convert_utils.h"
#include "abstract/utils.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "abstract/abstract_function.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace runtime {
namespace {
// Check if node is a value node need to create a device tensor.
bool IsFrontValueNode(const KernelWithIndex &node_with_index) {
  const auto &node = node_with_index.first;
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>() || IsValueNode<FuncGraph>(node) || IsValueNode<Primitive>(node)) {
    return false;
  }

  return true;
}

// Fetch real input node in maketuple.
KernelWithIndex FetchRealInputNode(const KernelWithIndex &node_with_index) {
  const auto &node = node_with_index.first;
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    return node_with_index;
  }

  const auto &abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
  if (output_num <= node_with_index.second) {
    MS_LOG(EXCEPTION) << "Invalid index:" << node_with_index.second << "for tuple node:" << node->DebugString();
  }

  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  size_t real_index = node_with_index.second;
  for (size_t i = kMakeTupleInputStartPos; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    const auto &sub_abstract = inputs[i]->abstract();
    MS_EXCEPTION_IF_NULL(sub_abstract);
    size_t tmp_index = common::AnfAlgo::GetOutputNumByAbstract(sub_abstract);
    // If it is not the output of node, need to subtract the number of inputs of it.
    if (real_index >= tmp_index) {
      real_index -= tmp_index;
      continue;
    }
    return {inputs[i], real_index};
  }
  MS_LOG(EXCEPTION) << "Failed to get real output from node:" << node->DebugString()
                    << " index:" << node_with_index.second;
}

// Fetch all the output index in the sub-abstract of abstract.
std::set<size_t> FetchRealIndexByAbstract(const AbstractBasePtr &abstract, std::vector<size_t> *const indexes) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(indexes);
  AbstractBasePtr dst_abstract = abstract;
  size_t pre_abstract_num = 0;
  std::set<size_t> output_indexs;
  if (indexes->empty()) {
    size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
    for (size_t i = 0; i < output_num; ++i) {
      (void)output_indexs.emplace(i);
    }
    return output_indexs;
  }

  size_t index = indexes->back();
  indexes->pop_back();

  // Fetch the dest abstract by index, and the abstracts num before the dest abstract.
  if (abstract->isa<abstract::AbstractTuple>()) {
    auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_abstract);
    const auto &sub_abstracts = tuple_abstract->elements();
    if (sub_abstracts.size() <= index) {
      MS_LOG(EXCEPTION) << "Invalid index:" << index << " for abstract:" << abstract->ToString();
    }
    for (size_t i = 0; i < index; ++i) {
      pre_abstract_num += common::AnfAlgo::GetOutputNumByAbstract(sub_abstracts[i]);
    }
    dst_abstract = sub_abstracts[index];
  } else {
    if (index != 0) {
      MS_LOG(EXCEPTION) << "Invalid abstract index:" << index << " for abstract:" << abstract->ToString();
    }
  }
  MS_EXCEPTION_IF_NULL(dst_abstract);

  // Fetch real output index.
  auto tmp_indexs = FetchRealIndexByAbstract(dst_abstract, indexes);
  for (auto tmp_index : tmp_indexs) {
    (void)output_indexs.emplace(tmp_index + pre_abstract_num);
  }
  return output_indexs;
}

// Get all the real parameters corresponding to node.
void FetchRealParameterByNode(const KernelWithIndex &node, std::set<KernelWithIndex> *const real_parameters,
                              std::set<KernelWithIndex> *invalid_call_nodes,
                              const mindspore::HashMap<AnfNodePtr, std::set<FuncGraphPtr>> &call_node_to_func_graphs) {
  MS_EXCEPTION_IF_NULL(node.first);
  MS_EXCEPTION_IF_NULL(real_parameters);
  MS_EXCEPTION_IF_NULL(invalid_call_nodes);
  MS_LOG(DEBUG) << "Fetch real parameter by node:" << node.first->DebugString() << " index:" << node.second;
  auto node_with_index = common::AnfAlgo::VisitKernelWithReturnType(node.first, node.second);
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  if (node_with_index.first->isa<ValueNode>() || node_with_index.first->isa<Parameter>()) {
    // If node is a valuenode or parameter, the real parameter is itself.
    MS_LOG(DEBUG) << "Add real parameter:" << node_with_index.first->DebugString()
                  << " index:" << node_with_index.second;
    (void)real_parameters->emplace(node_with_index);
  } else if (common::AnfAlgo::IsCallNode(node_with_index.first)) {
    // If node is a call node, the real parameters are the outputs of funcgraph the node called.
    if (invalid_call_nodes->find(node_with_index) != invalid_call_nodes->end()) {
      return;
    }
    (void)invalid_call_nodes->emplace(node_with_index);
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
  } else if (common::AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimMakeTuple)) {
    // If node is a maketuple node, the real parameters are its total inputs.
    const auto &real_input = FetchRealInputNode(node_with_index);
    MS_EXCEPTION_IF_NULL(real_input.first);
    MS_LOG(DEBUG) << "Real input node:" << real_input.first->DebugString() << " index:" << real_input.second
                  << " for tuple node:" << node_with_index.first->DebugString() << " index:" << node_with_index.second;
    FetchRealParameterByNode(real_input, real_parameters, invalid_call_nodes, call_node_to_func_graphs);
  } else if (common::AnfAlgo::CheckPrimitiveType(node.first, prim::kPrimSwitch)) {
    // If node is a switch node, the real parameters are its both true and false branches.
    const auto cnode = node_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto inputs = cnode->inputs();
    for (size_t i = kSwitchTrueBranchPos; i < inputs.size(); ++i) {
      FetchRealParameterByNode({inputs[i], 0}, real_parameters, invalid_call_nodes, call_node_to_func_graphs);
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimSwitchLayer)) {
    // If node is a switchlyaer node, the real parameters are its total branches.
    const auto &switch_layer_cnode = node_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_layer_cnode);
    const auto &switch_layer_inputs = switch_layer_cnode->inputs();
    if (switch_layer_inputs.size() != kSwitchLayerInputNum ||
        (!common::AnfAlgo::CheckPrimitiveType(switch_layer_inputs[kSwitchLayerBranchPos], prim::kPrimMakeTuple))) {
      MS_LOG(EXCEPTION) << "Invalid switch layer node:" << switch_layer_cnode->DebugString();
    }
    const auto &make_tuple_cnode = switch_layer_inputs[kSwitchLayerBranchPos]->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple_cnode);
    const auto &make_tuple_inputs = make_tuple_cnode->inputs();
    for (size_t i = kSwitchTrueBranchPos; i < make_tuple_inputs.size(); ++i) {
      FetchRealParameterByNode({make_tuple_inputs[i], 0}, real_parameters, invalid_call_nodes,
                               call_node_to_func_graphs);
    }
  } else {
    // If node is a kernel, the real parameter is itself.
    MS_LOG(DEBUG) << "Add real parameter:" << node_with_index.first->DebugString()
                  << " index:" << node_with_index.second;
    (void)real_parameters->emplace(node_with_index);
  }
}

// Topologically sort all funcgraphs according to the function call relationship.
std::vector<FuncGraphPtr> TopoSortForFuncGraph(const FuncGraphPtr &root, FuncGraphCallRelation *const edges) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(edges);
  MS_EXCEPTION_IF_NULL(root->manager());
  std::set<FuncGraphPtr> nodes;
  (void)nodes.emplace(root);

  FuncGraphSet subs = root->manager()->func_graphs();
  for (auto sub : subs) {
    if (sub != root && root != nullptr) {
      (void)nodes.emplace(sub);
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
    (void)result.emplace_back(node);
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
  MS_EXCEPTION_IF_NULL(node_value);
  if (node_value->isa<FuncGraph>() || node_value->isa<Primitive>()) {
    return;
  }

  size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(backend_node, 0);
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(backend_node, 0);
  if (output_type_id == kTypeUnknown) {
    output_type_id = common::AnfAlgo::GetOutputInferDataType(backend_node, 0);
  }

  if (front_node->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    front_node->set_kernel_info(kernel_info);
    std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    kernel_info->set_select_kernel_build_info(builder->Build());
    kernel_info->GetMutableSelectKernelBuildInfo()->SetOutputsKernelObjectType(
      {kernel::KernelObjectType::TUPLE_UNFOLD});
  }

  // Set build info to front node.
  auto backend_kernel_info = static_cast<device::KernelInfo *>(backend_node->kernel_info());
  MS_EXCEPTION_IF_NULL(backend_kernel_info);
  auto backend_build_info = backend_kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(backend_build_info);

  auto front_kernel_info = static_cast<device::KernelInfo *>(front_node->kernel_info());
  MS_EXCEPTION_IF_NULL(front_kernel_info);
  auto front_build_info = front_kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(front_build_info);
  // Set output format and device data type.
  if (front_build_info->GetAllOutputFormats().size() > front_node_with_index.second) {
    front_build_info->SetOutputFormat(backend_build_info->GetOutputFormat(0), front_node_with_index.second);
    front_build_info->SetOutputDeviceType(backend_build_info->GetOutputDeviceType(0), front_node_with_index.second);
  } else {
    auto formats = front_build_info->GetAllOutputFormats();
    auto types = front_build_info->GetAllOutputDeviceTypes();
    for (size_t i = 0; i <= front_node_with_index.second - front_build_info->GetAllOutputFormats().size(); ++i) {
      (void)formats.emplace_back(backend_build_info->GetOutputFormat(0));
      (void)types.emplace_back(backend_build_info->GetOutputDeviceType(0));
    }
    front_build_info->SetOutputsFormat(formats);
    front_build_info->SetOutputsDeviceType(types);
  }

  device::DeviceAddressPtr address = nullptr;
  if (node_value->isa<tensor::Tensor>() && node_value->cast<TensorPtr>()->is_forward_output()) {
    // If is_forward_output, get address from tensor
    auto tensor = node_value->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  } else {
    // Create device tensor.
    std::string output_format = AnfAlgo::GetOutputFormat(backend_node, 0);
    address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, tensor_size, output_format,
                                                                       output_type_id, ShapeVector());
  }
  MS_EXCEPTION_IF_NULL(address);
  MS_LOG(DEBUG) << "Create address for node:" << common::AnfAlgo::GetNodeDebugString(front_node)
                << " index:" << front_node_with_index.second << " addr:" << address << " size:" << tensor_size;
  AnfAlgo::SetOutputAddr(address, front_node_with_index.second, front_node.get());
  UpdateRefCount(address.get(), true);
}

TypeId FetchTypeIdByNode(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  TypeId type_id = kTypeUnknown;
  if (node->isa<ValueNode>() && node->abstract() != nullptr) {
    // For valuenode, fetch type from abstract.
    const auto &abs = FetchAbstractByIndex(node->abstract(), index);
    MS_EXCEPTION_IF_NULL(abs);
    const auto &type = abs->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    if (type->isa<TensorType>()) {
      const auto &tensor_type = type->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_type);
      const auto &element = tensor_type->element();
      type_id = element->type_id();
    } else {
      type_id = type->type_id();
    }
  } else {
    type_id = common::AnfAlgo::GetOutputInferDataType(node, index);
  }
  return type_id;
}

size_t FetchOutputSizeByNode(const AnfNodePtr &node, size_t index, TypeId type_id) {
  MS_EXCEPTION_IF_NULL(node);
  size_t size = GetTypeByte(TypeIdToType(type_id));
  if (node->isa<ValueNode>() && node->abstract() != nullptr) {
    const auto &abs = FetchAbstractByIndex(node->abstract(), index);
    MS_EXCEPTION_IF_NULL(abs);
    const auto &shape_ptr = abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    if (shape_ptr->isa<abstract::Shape>()) {
      const auto &shapes = shape_ptr->cast<abstract::ShapePtr>()->shape();
      size = std::accumulate(shapes.begin(), shapes.end(), size, std::multiplies<int64_t>());
    } else if (abs->isa<abstract::AbstractMonad>() || abs->isa<abstract::AbstractScalar>()) {
      MS_LOG(DEBUG) << "For scalar, the output shape is 1.";
    } else {
      MS_LOG(EXCEPTION) << "Invalid abstract;" << abs->ToString() << " for node:" << node->DebugString()
                        << " index:" << index;
    }
  } else {
    size = AnfAlgo::GetOutputTensorMemSize(node, index);
  }
  return size;
}

// Create a device tensor for front node.
// When the condition input of the switch and switchlayer or the output of a subgraph is a parameter or value node,
// there is no corresponding backend node for this parameter, so a device tensor needs to be created for it.
void CreateDeviceTensorForFrontNode(const KernelWithIndex &front_node_with_index, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &node = front_node_with_index.first;

  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Start create device tensor for front node:" << front_node_with_index.first->DebugString()
                << " index:" << front_node_with_index.second;

  // Create kernel info for front node.
  if (node->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    kernel_info->set_select_kernel_build_info(builder->Build());
    node->set_kernel_info(kernel_info);
  }

  // Set format.
  const auto &kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &builder = kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(builder);

  if (builder->GetAllOutputFormats().size() > front_node_with_index.second) {
    builder->SetOutputFormat(kOpFormat_DEFAULT, front_node_with_index.second);
  } else {
    auto formats = builder->GetAllOutputFormats();
    for (size_t i = 0; i <= front_node_with_index.second - builder->GetAllOutputFormats().size(); ++i) {
      (void)formats.emplace_back(kOpFormat_DEFAULT);
    }
    builder->SetOutputsFormat(formats);
  }

  // Set type.
  TypeId type_id = FetchTypeIdByNode(node, front_node_with_index.second);
  if (builder->GetAllOutputDeviceTypes().size() > front_node_with_index.second) {
    builder->SetOutputDeviceType(type_id, front_node_with_index.second);
  } else {
    auto types = builder->GetAllOutputDeviceTypes();
    for (size_t i = 0; i <= front_node_with_index.second - builder->GetAllOutputDeviceTypes().size(); ++i) {
      (void)types.emplace_back(type_id);
    }
    builder->SetOutputsDeviceType(types);
  }

  // Fetch mem size by shape, the shape is first obtained from the abstract to deal with the scenario where
  // the value node is a multi-level tuple.
  size_t size = FetchOutputSizeByNode(node, front_node_with_index.second, type_id);
  device::DeviceAddressPtr address = nullptr;
  if (node->isa<ValueNode>()) {
    const auto &node_value = node->cast<ValueNodePtr>()->value();
    if (node_value->isa<tensor::Tensor>() && node_value->cast<TensorPtr>()->is_forward_output()) {
      // If is_forward_output, get address from tensor
      auto tensor = node_value->cast<TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    } else {
      // Create device tensor.
      address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, size, kOpFormat_DEFAULT, type_id,
                                                                         ShapeVector());
    }
  } else {
    // Create device tensor.
    address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, size, kOpFormat_DEFAULT, type_id,
                                                                       ShapeVector());
  }
  MS_EXCEPTION_IF_NULL(address);
  MS_LOG(INFO) << "Create address for node that has no corresponding backend node:"
               << common::AnfAlgo::GetNodeDebugString(node) << " addr:" << address << " size:" << size
               << ", type id:" << type_id;
  AnfAlgo::SetOutputAddr(address, front_node_with_index.second, node.get());
  UpdateRefCount(address.get(), true);
}

// Fetch all funcgraph by a seed graph, if a calls b, b calls c, and c calls a, return a set of a, b, c.
void FetchAllExecutionFunction(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *const checked_funcgraphs,
                               const std::unordered_map<FuncGraphPtr, std::set<FuncGraphPtr>> &call_relation) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(checked_funcgraphs);
  if (checked_funcgraphs->find(func_graph) != checked_funcgraphs->end()) {
    return;
  }
  (void)checked_funcgraphs->emplace(func_graph);
  auto iter = call_relation.find(func_graph);
  if (iter == call_relation.end()) {
    return;
  }

  for (const auto &called_func_graph : iter->second) {
    MS_EXCEPTION_IF_NULL(called_func_graph);
    FetchAllExecutionFunction(called_func_graph, checked_funcgraphs, call_relation);
  }
}

bool IsValidMonadNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->isa<ValueNode>() || node->isa<Parameter>() || common::AnfAlgo::IsCallNode(node);
}

// Fetch all inputs of node.
std::vector<KernelWithIndex> FetchInputNodeByNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (HasAbstractMonad(node)) {
    const auto &real_node_with_index = common::AnfAlgo::VisitKernelWithReturnType(node, 0);
    const auto &real_node = real_node_with_index.first;
    MS_EXCEPTION_IF_NULL(real_node);
    if (IsValidMonadNode(real_node)) {
      return {real_node_with_index};
    }
    MS_LOG(EXCEPTION) << "Invalid monad node:" << real_node->DebugString();
  }

  // The node is divided into the following types:
  // 1. depend and load.
  const auto &node_with_index =
    common::AnfAlgo::VisitKernelWithReturnType(node, 0, false, {prim::kPrimTupleGetItem, prim::kPrimMakeTuple});
  auto real_node = node_with_index.first;
  size_t real_index = node_with_index.second;
  MS_EXCEPTION_IF_NULL(real_node);
  std::vector<KernelWithIndex> results;

  // 2. Tuple node.
  const PrimitiveSet expand_prims{prim::kPrimMakeTuple};
  // The MakeTuple/MakeSparse node need expand and recurse.
  if (IsOneOfPrimitiveCNode(real_node, expand_prims)) {
    const auto &cnode = real_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    for (size_t i = kMakeTupleInputStartPos; i < inputs.size(); ++i) {
      const auto &sub_results = FetchInputNodeByNode(inputs[i]);
      (void)results.insert(results.end(), sub_results.begin(), sub_results.end());
    }
    return results;
  }

  // 3. One output node.
  const auto &abstract = real_node->abstract();
  if (abstract == nullptr) {
    MS_LOG(WARNING) << "Empty abstract for node:" << real_node->DebugString();
    (void)results.emplace_back(common::AnfAlgo::VisitKernelWithReturnType(real_node, real_index));
    return results;
  }

  // 4 Other.
  if (common::AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimTupleGetItem)) {
    std::vector<size_t> index_stack;
    auto get_item_src_node = common::AnfAlgo::GetTupleIndexes(real_node, &index_stack);
    MS_EXCEPTION_IF_NULL(get_item_src_node);
    if (index_stack.empty()) {
      const auto &sub_results = FetchInputNodeByNode(get_item_src_node);
      (void)results.insert(results.end(), sub_results.begin(), sub_results.end());
      return results;
    }
    auto get_item_src_abstract = get_item_src_node->abstract();
    MS_EXCEPTION_IF_NULL(get_item_src_abstract);
    auto indexes = FetchRealIndexByAbstract(get_item_src_abstract, &index_stack);
    (void)std::transform(indexes.begin(), indexes.end(), std::back_inserter(results),
                         [&get_item_src_node](const auto &index) { return KernelWithIndex(get_item_src_node, index); });
    return results;
  }

  size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
  for (size_t i = 0; i < output_num; ++i) {
    (void)results.emplace_back(real_node, i);
  }
  return results;
}

// Add formal parameter and real parameter into realationship map.
void AddFormalToRealParameter(const AnfNodePtr &formal_parameter, const AnfNodePtr &real_parameter,
                              const CallNodeToFuncGraph &call_node_to_func_graphs,
                              FormalToRealParameter *const formal_to_real_parameters) {
  MS_EXCEPTION_IF_NULL(formal_parameter);
  MS_EXCEPTION_IF_NULL(real_parameter);
  MS_EXCEPTION_IF_NULL(formal_to_real_parameters);
  auto abstract = formal_parameter->abstract();
  if (abstract == nullptr) {
    MS_LOG(EXCEPTION) << "Empty abstract for parameter:" << formal_parameter->DebugString();
  }
  size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);

  for (size_t i = 0; i < output_num; ++i) {
    std::set<KernelWithIndex> real_parameters;
    std::set<KernelWithIndex> invalid_call_nodes;
    FetchRealParameterByNode({real_parameter, i}, &real_parameters, &invalid_call_nodes, call_node_to_func_graphs);
    if (real_parameters.empty()) {
      MS_LOG(EXCEPTION) << "Failed to find real parameter for formal parameter:" << real_parameter->DebugString();
    }

    for (const auto &parameter : real_parameters) {
      MS_EXCEPTION_IF_NULL(parameter.first);
      MS_LOG(DEBUG) << "Add formal parameter:" << formal_parameter->DebugString() << " index:" << i
                    << " to real parameter:" << parameter.first->DebugString() << " index:" << parameter.second;
    }
    (*formal_to_real_parameters)[{formal_parameter, i}].insert(real_parameters.begin(), real_parameters.end());
  }
}

// Recursively traverse the input to confirm whether there is an input of recursive call.
bool IsFirstControlNode(const AnfNodePtr &node, std::set<AnfNodePtr> *checked_nodes,
                        const std::set<AnfNodePtr> &unrecursion_call_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(checked_nodes);
  if (!node->isa<CNode>() || checked_nodes->find(node) != checked_nodes->end()) {
    return true;
  }
  (void)checked_nodes->emplace(node);

  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if ((common::AnfAlgo::IsCallNode(input) && unrecursion_call_nodes.find(input) == unrecursion_call_nodes.end()) ||
        (!IsFirstControlNode(input, checked_nodes, unrecursion_call_nodes))) {
      return false;
    }
  }
  return true;
}

// Check if src_node depends on dst_node.
bool IsTopoDependNode(const AnfNodePtr &src_node, const AnfNodePtr &dst_node, std::set<AnfNodePtr> *checked_node) {
  MS_EXCEPTION_IF_NULL(src_node);
  MS_EXCEPTION_IF_NULL(dst_node);
  MS_EXCEPTION_IF_NULL(checked_node);
  if (src_node == dst_node) {
    return true;
  }
  if (!src_node->isa<CNode>() || checked_node->find(src_node) != checked_node->end()) {
    return false;
  }

  (void)checked_node->emplace(src_node);
  const auto &cnode = src_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (IsTopoDependNode(input, dst_node, checked_node)) {
      return true;
    }
  }
  return false;
}
}  // namespace

bool IsInvalidPartial(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }

  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  if (inputs.size() <= kPartialFuncGraphPos) {
    return false;
  }

  if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    return false;
  }
  if (IsDeadNode(inputs[kPartialFuncGraphPos])) {
    return true;
  }
  return false;
}

KernelWithIndex FetchRealNodeByGetItem(const KernelWithIndex &node_with_index) {
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  std::vector<size_t> index_stack{node_with_index.second};

  const auto &get_item_src_node = common::AnfAlgo::GetTupleIndexes(node_with_index.first, &index_stack);
  MS_EXCEPTION_IF_NULL(get_item_src_node);
  const auto &get_item_src_abstract = get_item_src_node->abstract();
  MS_EXCEPTION_IF_NULL(get_item_src_abstract);
  auto indexes = FetchRealIndexByAbstract(get_item_src_abstract, &index_stack);
  if (indexes.empty()) {
    MS_LOG(EXCEPTION) << "Failed to find index for node:" << get_item_src_node;
  }
  if (indexes.size() > 1) {
    MS_LOG(WARNING) << "Output size:" << indexes.size() << " for node:" << get_item_src_node->DebugString()
                    << " more than 1";
  }
  return {get_item_src_node, *(indexes.begin())};
}

bool IsCsrNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCSRTensorGetIndptr) ||
         common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCSRTensorGetIndices) ||
         common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCSRTensorGetValues) ||
         common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCSRTensorGetDenseShape);
}

bool IsCooNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCOOTensorGetIndices) ||
         common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCOOTensorGetValues) ||
         common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCOOTensorGetDenseShape);
}

KernelWithIndex GetFrontNodeByKernelGraph(const AnfNodePtr &backend_node, const KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(graph);
  const auto &front_node = graph->GetFrontAnfByBackendAnf(backend_node);
  if (front_node != nullptr) {
    MS_LOG(DEBUG) << "Front node:" << front_node->DebugString() << " index:0"
                  << " for backend node:" << backend_node->DebugString();
    return {front_node, 0};
  }
  const auto &front_node_with_index = graph->GetFrontNodeByInternalParameter(backend_node);
  if (front_node_with_index.first != nullptr) {
    MS_LOG(DEBUG) << "Internal front node:" << front_node_with_index.first->DebugString()
                  << " index:" << front_node_with_index.second << " for backend node:" << backend_node->DebugString();
    return front_node_with_index;
  }
  const auto &front_tuple_node_with_index = graph->GetElementInTupleBackendFrontIndexMap(backend_node);
  if (front_tuple_node_with_index.first == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot find front node for backend node:" << backend_node->DebugString()
                      << " in graph:" << graph->ToString();
  }
  MS_LOG(DEBUG) << "Tuple front node:" << front_tuple_node_with_index.first->DebugString()
                << " index:" << front_tuple_node_with_index.second;
  return front_tuple_node_with_index;
}

std::vector<KernelWithIndex> FetchInputNodeByCNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Fetch input node for:" << node->DebugString();
  if (!node->isa<CNode>()) {
    MS_LOG(DEBUG) << "Empty input node for:" << node->DebugString();
    return {};
  }

  std::vector<KernelWithIndex> results;
  // The first input of normal cnode is the primitive of node, and the real input starts from the second input,
  // but in control flow, the call node has no primitive, and the 0th input is funcgraph or partial.
  size_t input_start_pos = kCNodeInputStartPos;
  if (common::AnfAlgo::IsCallNode(node)) {
    input_start_pos = 0;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto inputs = cnode->inputs();

  // The first branch of the input of the switch node is the true branch, and the second is the false branch.
  // But in switch actor, since the false value is 0, it corresponds to the first branch. Therefore, the input
  // of the switch node needs to exchange the positions of the two branches. So deal separately.
  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch)) {
    if (inputs.size() != kSwitchInputNum) {
      MS_LOG(EXCEPTION) << "Invalid switch node:" << node->DebugString();
    }
    (void)results.emplace_back(common::AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchCondPos], 0));
    (void)results.emplace_back(common::AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchFalseBranchPos], 0));
    (void)results.emplace_back(common::AnfAlgo::VisitKernelWithReturnType(inputs[kSwitchTrueBranchPos], 0));
    return results;
  }

  for (size_t i = input_start_pos; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    const auto &sub_results = FetchInputNodeByNode(inputs[i]);
    (void)results.insert(results.end(), sub_results.begin(), sub_results.end());
  }
  return results;
}

abstract::AbstractBasePtr FetchAbstractByIndex(const AbstractBasePtr &abstract, size_t index) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>() || abstract->cast<abstract::AbstractSequencePtr>()->dynamic_len()) {
    if (index != 0) {
      MS_LOG(EXCEPTION) << "Invalid abstract index:" << index << " for abstract:" << abstract->ToString();
    }
    return abstract;
  }

  auto tuple_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  const auto &sub_abstracts = tuple_abstract->elements();
  size_t real_index = index;
  for (const auto &sub_abstract : sub_abstracts) {
    size_t tmp_index = common::AnfAlgo::GetOutputNumByAbstract(sub_abstract);
    if (real_index >= tmp_index) {
      real_index -= tmp_index;
      continue;
    }
    return FetchAbstractByIndex(sub_abstract, real_index);
  }
  MS_LOG(EXCEPTION) << "Invalid abstract index:" << index << " for abstract:" << abstract->ToString();
}

bool IsPartialInput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &abstract = node->abstract();
  if (abstract != nullptr) {
    if (abstract->isa<abstract::AbstractFunction>()) {
      return true;
    }
    return false;
  }

  if (!node->isa<CNode>()) {
    return false;
  }

  // If the abstract is empty and the node is a cnode, check its true branch.
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  const auto &inputs = cnode->inputs();
  if (inputs.size() < kSwitchTrueBranchIndex + 1) {
    MS_LOG(EXCEPTION) << "Invalid switch node:" << node->DebugString();
  }
  const auto &branch_node = inputs[kSwitchTrueBranchIndex];
  MS_EXCEPTION_IF_NULL(branch_node);
  const auto &branch_abstract = branch_node->abstract();
  // If abstract is empty, the default is true.
  if (branch_abstract == nullptr) {
    MS_LOG(WARNING) << "Failed to get abstract by true branch input of switch node:" << node->DebugString();
    return true;
  }

  if (branch_abstract->isa<abstract::AbstractFunction>()) {
    return true;
  } else if (branch_abstract->isa<abstract::AbstractSequence>()) {
    // In switch layer, the true branch input is a make tuple.
    auto sequence_abstract = branch_abstract->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(sequence_abstract);
    const auto &sub_abstracts = sequence_abstract->elements();
    if (sub_abstracts.empty() || sub_abstracts[0] == nullptr) {
      MS_LOG(WARNING) << "Failed to get abstract by true branch input of switch node:" << node->DebugString();
      return true;
    }
    if (sub_abstracts[0]->isa<abstract::AbstractFunction>()) {
      return true;
    }
  }
  return false;
}

// Fetch the depend nodes according to the monad node.
void FetchRealDependNodeByAutoMonad(const AnfNodePtr &node, std::set<AnfNodePtr> *const depend_nodes) {
  // Find the real input node, include the monad node and make tuple node.
  const std::vector<PrimitivePtr> return_types = {prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad,
                                                  prim::kPrimMakeTuple};
  const auto &node_with_index = common::AnfAlgo::VisitKernelWithReturnType(node, 0, false, return_types);
  auto real_node = node_with_index.first;
  MS_EXCEPTION_IF_NULL(real_node);
  if (!real_node->isa<CNode>()) {
    return;
  }

  const auto &real_cnode = real_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(real_cnode);
  const auto &real_inputs = real_cnode->inputs();

  // Make tuple node needs to be expanded.
  if (common::AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimMakeTuple)) {
    for (size_t i = 1; i < real_inputs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(real_inputs[i]);
      FetchRealDependNodeByAutoMonad(real_inputs[i], depend_nodes);
    }
    return;
  }

  const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> recursion_prims = {
    prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad, prim::kPrimMakeTuple};
  if (common::AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimDepend) ||
      common::AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimLoad)) {
    FetchRealDependNodeByAutoMonad(real_inputs[kDependAttachNodeIndex], depend_nodes);
    // The real input may be this scene:  depend/load --> load/depend, so need add the control arrow for real input
    // node in this scene.
    if (IsOneOfPrimitiveCNode(real_inputs[kRealInputIndexInDepend], recursion_prims)) {
      FetchRealDependNodeByAutoMonad(real_inputs[kRealInputIndexInDepend], depend_nodes);
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimUpdateState)) {
    for (size_t i = kUpdateStateRealInput; i < real_inputs.size(); ++i) {
      FetchRealDependNodeByAutoMonad(real_inputs[i], depend_nodes);
    }
  } else {
    (void)depend_nodes->emplace(real_node);
  }
}

// Get all the depend nodes of node in side effect.
std::vector<AnfNodePtr> FetchAllMonadNodeByNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return {};
  }
  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimUpdateState) ||
      common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend) ||
      common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimLoad)) {
    return {node};
  }

  std::vector<AnfNodePtr> results;
  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (const auto &input : cnode->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      const auto &result = FetchAllMonadNodeByNode(input);
      (void)results.insert(results.end(), result.begin(), result.end());
    }
  }
  return results;
}

void ControlNodeParser::Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
                              const std::vector<DeviceContext *> &device_contexts, const FuncGraphPtr &root_graph,
                              const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs) {
  if (graphs.size() != device_contexts.size()) {
    MS_LOG(EXCEPTION) << "Graph num is not equal to device context, graph:" << graphs.size()
                      << " device context num:" << device_contexts.size();
  }

  if (control_nodes.size() <= 1) {
    MS_LOG(DEBUG) << "Control node parser is not inited.";
    return;
  }
  MS_LOG(DEBUG) << "Control node parse start.";

  // Fetch default device context.
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  DeviceContext *default_context = nullptr;
  if (device_contexts.empty()) {
    default_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
  } else {
    default_context = device_contexts[0];
  }
  MS_EXCEPTION_IF_NULL(default_context);

  KernelGraphToDeviceContext kernel_graph_to_device_contexts;
  for (size_t i = 0; i < graphs.size(); ++i) {
    kernel_graph_to_device_contexts[graphs[i]] = device_contexts[i];
  }

  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    MS_LOG(DEBUG) << "Print control node:" << control_node->DebugString();
  }

  is_inited_ = true;

  root_func_graph_ = root_graph;

  root_graph_parameters_ = root_graph->parameters();

  func_graph_to_kernel_graph_groups_ = func_graph_to_kernel_graphs;
  for (const auto &func_graph_to_kernel_graph_groups : func_graph_to_kernel_graph_groups_) {
    for (const auto &kernel_graph_group : func_graph_to_kernel_graph_groups.second) {
      for (const auto &kernel_graph : kernel_graph_group) {
        MS_EXCEPTION_IF_NULL(func_graph_to_kernel_graph_groups.first);
        MS_EXCEPTION_IF_NULL(kernel_graph);
        MS_LOG(DEBUG) << "Funcgraph to kernel graph, func:" << func_graph_to_kernel_graph_groups.first->ToString()
                      << " kernel_graph:" << kernel_graph->ToString();
      }
    }
  }

  CreateBranchIDForCallNode(control_nodes);

  ParseFrontNodeToKernelGraph(graphs);

  ParseCallNodeToFuncGraph(control_nodes);

  ParseUnRecursionCallNode();

  InsertDependForParallelCall(control_nodes);

  ParseKernelGraphGroup(kernel_graph_to_device_contexts);

  ParseNodeLevel(control_nodes);

  ParseNeedStackControlNode(control_nodes);

  ParseFormalToRealParameter(control_nodes);

  ParseFrontToBackendParameter(graphs, device_contexts);

  CreateDeviceTensorForRootGraphParameter(default_context);

  ParseFrontToBackendKernel(graphs, device_contexts);

  ParseDeviceContext(control_nodes, graphs, device_contexts, default_context, func_graph_to_kernel_graphs);

  FetchFrontValueNode(control_nodes, default_context);

  ParseControlNodeParameter(control_nodes);

  ParseFirstControlNodeAndKernelGraphForFuncGraph(control_nodes);
  MS_LOG(DEBUG) << "Control node parse end.";
}

// Fetch all the funcgraph recursively that the call node will call.
void FetchAllCalledFuncGraph(const AnfNodePtr &call_node, std::set<FuncGraphPtr> *called_graphs,
                             const CallNodeToFuncGraph &call_node_to_func_graphs,
                             const FuncGraphToCallNode &func_graph_to_call_nodes) {
  MS_EXCEPTION_IF_NULL(call_node);
  MS_EXCEPTION_IF_NULL(called_graphs);
  const auto &call_iter = call_node_to_func_graphs.find(call_node);
  if (call_iter == call_node_to_func_graphs.end()) {
    return;
  }
  for (const auto &func_graph : call_iter->second) {
    MS_EXCEPTION_IF_NULL(func_graph);
    if (called_graphs->find(func_graph) != called_graphs->end()) {
      continue;
    }
    (void)called_graphs->emplace(func_graph);
    const auto &graph_iter = func_graph_to_call_nodes.find(func_graph);
    if (graph_iter == func_graph_to_call_nodes.end()) {
      continue;
    }

    // Fetch the funcgraph recursively.
    for (const auto &node : graph_iter->second) {
      FetchAllCalledFuncGraph(node, called_graphs, call_node_to_func_graphs, func_graph_to_call_nodes);
    }
  }
}

tensor::TensorPtr ControlNodeParser::CreateTensorForValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  tensor::TensorPtr tensor = nullptr;
  if (value->isa<Monad>()) {
    tensor = std::make_shared<tensor::Tensor>(int8_t('U'), TypeIdToType(kNumberTypeInt8));
  } else if (value->isa<Scalar>()) {
    const auto scalar_value = value->cast<ScalarPtr>();
    MS_EXCEPTION_IF_NULL(scalar_value);
    tensor = ScalarToTensor(scalar_value);
  } else {
    MS_LOG(EXCEPTION) << "Invalid value:" << value->ToString();
  }
  control_node_tensors_.emplace_back(tensor);
  return tensor;
}

bool ControlNodeParser::IsParallelCallRecursionGraph(const AnfNodePtr &call_node1, const AnfNodePtr &call_node2,
                                                     const FuncGraphToCallNode &func_graph_to_call_nodes) {
  // Fetch all funcgraphs the two call nodes will call both.
  std::set<FuncGraphPtr> called_graphs_1;
  FetchAllCalledFuncGraph(call_node1, &called_graphs_1, call_node_to_func_graphs_, func_graph_to_call_nodes);
  std::set<FuncGraphPtr> called_graphs_2;
  FetchAllCalledFuncGraph(call_node2, &called_graphs_2, call_node_to_func_graphs_, func_graph_to_call_nodes);
  std::vector<FuncGraphPtr> common_called_graphs;
  (void)std::set_intersection(called_graphs_1.begin(), called_graphs_1.end(), called_graphs_2.begin(),
                              called_graphs_2.end(), std::back_inserter(common_called_graphs));

  // Check for recursive calls in funcgraph.
  for (const auto &func_graph : common_called_graphs) {
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &iter = func_graph_to_call_nodes.find(func_graph);
    if (iter == func_graph_to_call_nodes.end()) {
      continue;
    }
    for (const auto &call_node : iter->second) {
      MS_EXCEPTION_IF_NULL(call_node);
      if (IsRecursionCallNode(call_node)) {
        MS_LOG(INFO) << "Call node:" << call_node1->DebugString() << " and:" << call_node2->DebugString()
                     << " would call the same recursion in graph:" << func_graph
                     << " which has a recursion call:" << call_node->DebugString();
        return true;
      }
    }
  }
  return false;
}

void ControlNodeParser::InsertDependForParallelCall(const std::vector<AnfNodePtr> &control_nodes) {
  MS_LOG(INFO) << "InsertDependForParallelCall start";
  // Fetch call node in funcgraph.
  FuncGraphToCallNode func_graph_to_call_nodes;
  for (const auto &control_node : control_nodes) {
    if (common::AnfAlgo::IsCallNode(control_node)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      (void)func_graph_to_call_nodes[func_graph].emplace(control_node);
    }
  }

  std::vector<AnfNodePtr> call_nodes;
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (!common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      if (common::AnfAlgo::IsCallNode(control_node)) {
        // Fetch all the call nodes in the same graph.
        (void)call_nodes.emplace_back(control_node);
      }
      continue;
    }

    // Check whether there is a topology relationship between call nodes.
    for (size_t i = 0; i < call_nodes.size(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        std::set<AnfNodePtr> checked_nodes;
        if (IsTopoDependNode(call_nodes[i], call_nodes[j], &checked_nodes) ||
            (!IsParallelCallRecursionGraph(call_nodes[i], call_nodes[j], func_graph_to_call_nodes))) {
          continue;
        }
        // If there is no topological relationship between call nodes, and the same recursive graph will be called
        // at the same time, then a depend node needs to be inserted between call nodes.
        auto func_graph = call_nodes[i]->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        auto cnode = call_nodes[i]->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        const auto &inputs = cnode->inputs();
        MS_EXCEPTION_IF_NULL(inputs[0]);

        // Create a depend node.
        std::vector<AnfNodePtr> depend_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                                 cnode->input(0), call_nodes[j]};
        auto new_depend = func_graph->NewCNode(depend_inputs);
        new_depend->set_abstract(cnode->input(0)->abstract());

        // Set depend node to call input.
        std::vector<AnfNodePtr> new_call_inputs{new_depend};
        for (size_t k = 1; k < inputs.size(); ++k) {
          (void)new_call_inputs.emplace_back(inputs[k]);
        }
        cnode->set_inputs(new_call_inputs);
        MS_LOG(INFO) << "Add depend node:" << new_depend->DebugString()
                     << " for call node:" << call_nodes[i]->DebugString() << " and:" << call_nodes[j]->DebugString();
      }
    }
    call_nodes.clear();
  }
  MS_LOG(INFO) << "InsertDependForParallelCall end";
}

bool ControlNodeParser::IsControlFlowDataArrow(const KernelGraphPtr &graph, const AnfNodePtr &backend_node) {
  MS_EXCEPTION_IF_NULL(graph);
  // Has no control flow node.
  if (!IsInited()) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(backend_node);
  if (!backend_node->isa<Parameter>()) {
    return false;
  }
  auto parameter_node = backend_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter_node);

  // Parameter input should be linked to its entrance actor.
  auto front_node = graph->GetFrontAnfByBackendAnf(backend_node);
  auto internal_node_with_index = graph->GetFrontNodeByInternalParameter(backend_node);
  front_node = (front_node != nullptr ? front_node : internal_node_with_index.first);
  if (front_node == nullptr) {
    auto front_node_with_index = graph->GetElementInTupleBackendFrontIndexMap(backend_node);
    front_node = front_node_with_index.first;
  }
  MS_EXCEPTION_IF_NULL(front_node);
  const auto &real_front_node = common::AnfAlgo::VisitKernelWithReturnType(front_node, 0).first;
  if (real_front_node != nullptr && real_front_node->isa<ValueNode>() && (!HasAbstractMonad(real_front_node))) {
    // If the real front node is a value node, we have two situations:
    // 1. if the value in value node is a tensor, it should be set into device tensor store by graph scheduler;
    // 2. if the value is a monad state, it should be converted to control arrow, which should link by control
    //    node scheduler.
    MS_LOG(DEBUG) << "Front node:" << real_front_node->DebugString()
                  << " of backend node:" << backend_node->DebugString() << " is a valuenode.";
    return false;
  }

  // If parameter is a weight node in root funcgraph, it should be set to kernel actor directly.
  if (IsRootGraphPersistentDeviceTensor(front_node)) {
    MS_LOG(DEBUG) << "backend node:" << backend_node->DebugString()
                  << " front node:" << (front_node == nullptr ? "null" : front_node->DebugString());
    return false;
  }

  // If the input front node and graph not in same graph group, the input arrow should be link to the exit actor
  // of the graph.
  if (!IsSameKernelGraphGroup(front_node, graph)) {
    return true;
  }

  // If the graph has a call input, all of its inputs in the graph should be linked to its stack actor.
  if (IsCallInputKernelGraph(graph.get())) {
    // If the input come from a kernel graph belong the same group, it should be linked by internal parameter.
    if (front_node != nullptr && (IsSameKernelGraphGroup(front_node, graph) || front_node->isa<ValueNode>())) {
      return false;
    }
    return true;
  }

  return (front_node != nullptr && front_node->isa<Parameter>());
}

bool ControlNodeParser::IsRootGraphPersistentDeviceTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPersistentDeviceTensor(node)) {
    return false;
  }

  // No control flow.
  if (!is_inited_) {
    return true;
  }

  // Maybe the load node, need fetch the real parameter node.
  auto real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({node, 0}).first;
  MS_EXCEPTION_IF_NULL(real_node);
  return find(root_graph_parameters_.begin(), root_graph_parameters_.end(), real_node) != root_graph_parameters_.end();
}

bool ControlNodeParser::IsNeedStackControlNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!(node->isa<CNode>())) {
    return false;
  }

  return need_stack_control_nodes_.find(node) != need_stack_control_nodes_.end();
}

bool ControlNodeParser::IsRecursionCallNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::IsCallNode(node)) {
    return false;
  }
  return find(unrecursion_call_nodes_.begin(), unrecursion_call_nodes_.end(), node) == unrecursion_call_nodes_.end();
}

bool ControlNodeParser::IsRecursionKernelGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto group_info_iter = kernel_graphs_to_group_info_.find(graph);
  if (group_info_iter == kernel_graphs_to_group_info_.end()) {
    MS_LOG(EXCEPTION) << "Invalid kernel graph:" << graph->ToString();
  }
  MS_EXCEPTION_IF_NULL(group_info_iter->second);
  if (!group_info_iter->second->need_stack_) {
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
                                           DeviceContext *default_context,
                                           const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs) {
  MS_EXCEPTION_IF_NULL(default_context);
  ParseDeviceContextForFuncGraph(kernel_graphs, device_contexts, default_context, func_graph_to_kernel_graphs);
  ParseDeviceContextForReturnNode(default_context);
  ParseDeviceContextForCallNode(control_nodes);
  ParseDeviceContextForPartialNode(control_nodes);
}

void ControlNodeParser::ParseDeviceContextForFuncGraph(const std::vector<KernelGraphPtr> &kernel_graphs,
                                                       const std::vector<DeviceContext *> &device_contexts,
                                                       DeviceContext *default_context,
                                                       const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs) {
  MS_EXCEPTION_IF_NULL(default_context);
  if (device_contexts.size() != kernel_graphs.size()) {
    MS_LOG(EXCEPTION) << "Invalid device context size:" << device_contexts.size()
                      << " graph size:" << kernel_graphs.size();
  }
  mindspore::HashMap<KernelGraphPtr, DeviceContext *> kernel_graph_to_device_context;
  for (size_t i = 0; i < kernel_graphs.size(); ++i) {
    kernel_graph_to_device_context[kernel_graphs[i]] = device_contexts[i];
  }

  // Collect the device context type of the parameter in the kernel graph as the type of the real parameters.
  for (const auto &func_graph_to_kernel_graph : func_graph_to_kernel_graphs) {
    const auto &func_graph = func_graph_to_kernel_graph.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    std::vector<KernelWithIndex> front_parameters;
    for (const auto &parameter : func_graph->parameters()) {
      const auto &abstract = parameter->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      for (size_t i = 0; i < common::AnfAlgo::GetOutputNumByAbstract(abstract); ++i) {
        (void)front_parameters.emplace_back(parameter, i);
      }
    }
    std::vector<const DeviceContext *> parameter_device_contexts(front_parameters.size(), default_context);
    std::map<KernelWithIndex, DeviceContext *> front_parameter_to_device_context;

    for (const auto &kernel_graph_group : func_graph_to_kernel_graph.second) {
      for (const auto &kernel_graph : kernel_graph_group) {
        MS_EXCEPTION_IF_NULL(kernel_graph);
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
  MS_EXCEPTION_IF_NULL(root_func_graph_);
  MS_EXCEPTION_IF_NULL(root_func_graph_->manager());
  FuncGraphSet sub_graphs = root_func_graph_->manager()->func_graphs();
  for (auto sub_graph : sub_graphs) {
    MS_EXCEPTION_IF_NULL(sub_graph);
    if (func_graph_to_device_contexts_.find(sub_graph) == func_graph_to_device_contexts_.end()) {
      size_t output_num = 0;
      for (const auto &parameter : sub_graph->parameters()) {
        const auto &abstract = parameter->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        output_num += common::AnfAlgo::GetOutputNumByAbstract(abstract);
      }
      func_graph_to_device_contexts_[sub_graph] = std::vector<const DeviceContext *>(output_num, default_context);
    }
  }
}

void ControlNodeParser::ParseDeviceContextForPartialNode(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    if (!common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial)) {
      continue;
    }

    MS_EXCEPTION_IF_NULL(control_node);
    const auto &cnode = control_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    if (inputs.size() <= kPartialFuncGraphPos) {
      MS_LOG(EXCEPTION) << "Invalid input size for partial node:" << cnode->DebugString();
    }
    auto &func_node = inputs[kPartialFuncGraphPos];
    // Ignore if the node is 'Partial(DeadNode,)'.
    if (IsDeadNode(func_node)) {
      MS_LOG(DEBUG) << "Ignore partial dead node:" << cnode->DebugString();
      continue;
    }
    // Fetch the funcgraph in partial node.
    const auto &func_graph = GetValueNode<FuncGraphPtr>(func_node);
    if (func_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid funcgraph node:" << func_node->DebugString()
                        << " for partial node:" << cnode->DebugString();
    }

    // Fetch the device contexts for the formal parameters in the funcgraph of partial node.
    auto iter = func_graph_to_device_contexts_.find(func_graph);
    if (iter == func_graph_to_device_contexts_.end()) {
      MS_LOG(EXCEPTION) << "Failed to get device contexts for funcgraph:" << func_graph->ToString();
    }

    size_t input_num = 0;
    for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(inputs[i]);
      const auto &abstract = inputs[i]->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      input_num += common::AnfAlgo::GetOutputNumByAbstract(abstract);
    }
    if (input_num > iter->second.size()) {
      MS_LOG(EXCEPTION) << "Invalid input num:" << input_num << " for funcgraph:" << func_graph->ToString()
                        << " device context size:" << iter->second.size()
                        << " for partial node:" << cnode->DebugString();
    }

    // Get the device contexts for the real parameters.
    std::vector<const DeviceContext *> device_contexts;
    // In partial node, the first input is always a partial, maybe a funcgraph or a partial node, so we need
    // to insert an empty device context for it.
    (void)device_contexts.emplace_back(nullptr);
    for (size_t i = 0; i < input_num; ++i) {
      MS_EXCEPTION_IF_NULL(iter->second[i]);
      (void)device_contexts.emplace_back(iter->second[i]);
    }
    control_node_to_device_contexts_[control_node] = device_contexts;
  }
}

void ControlNodeParser::ParseDeviceContextForCallNode(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (!common::AnfAlgo::IsCallNode(control_node)) {
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
    (void)device_contexts.emplace_back(nullptr);
    const auto &cnode = control_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    size_t call_input_num = 0;
    for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(inputs[i]);
      const auto &abstract = inputs[i]->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      call_input_num += common::AnfAlgo::GetOutputNumByAbstract(abstract);
    }

    if (call_input_num > iter->second.size()) {
      MS_LOG(EXCEPTION) << "Invalid input size:" << call_input_num << " context size:" << iter->second.size()
                        << "for funcgraph" << func_graph->ToString() << " for call node:" << cnode->DebugString();
    }

    // Fetch the device contexts for the real parameters on the call node.
    for (size_t i = iter->second.size() - call_input_num; i < iter->second.size(); ++i) {
      MS_EXCEPTION_IF_NULL(iter->second[i]);
      (void)device_contexts.emplace_back(iter->second[i]);
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
    (void)func_graph_call_relation[func_graph].emplace_back(call_node_to_func_graphs.second);
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
    if (inputs.size() <= kReturnInputPos) {
      MS_LOG(EXCEPTION) << "Invalid return node:" << cnode->DebugString();
    }
    const auto output_nodes = FetchInputNodeByNode(inputs[kReturnInputPos]);
    std::vector<const DeviceContext *> return_device_contexts;

    for (const auto &output_node : output_nodes) {
      MS_EXCEPTION_IF_NULL(output_node.first);
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
        size_t index = LongToSize(iter - func_graph->parameters().begin());
        MS_EXCEPTION_IF_NULL(func_graph_iter->second[index]);
        (void)return_device_contexts.emplace_back(func_graph_iter->second[index]);
      } else if (output_node.first->isa<ValueNode>()) {
        // If the output is parameter, used the default context type.
        (void)return_device_contexts.emplace_back(default_context);
      } else if (common::AnfAlgo::IsCallNode(output_node.first)) {
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
        (void)return_device_contexts.emplace_back(call_device_contexts[output_node.second]);
      } else if (common::AnfAlgo::CheckPrimitiveType(output_node.first, prim::kPrimPartial) ||
                 common::AnfAlgo::CheckPrimitiveType(output_node.first, prim::kPrimSwitch)) {
        (void)return_device_contexts.emplace_back(default_context);
      } else if (output_node.first->isa<CNode>()) {
        // If the output is a cnode, get the device context type by the kernel.
        const auto &iter = front_to_backend_kernels_.find(output_node);
        if (iter == front_to_backend_kernels_.end()) {
          MS_LOG(EXCEPTION) << "Cannot find backend kernel for cnode:" << output_node.first->DebugString();
        }
        MS_EXCEPTION_IF_NULL(iter->second.second);
        (void)return_device_contexts.emplace_back(iter->second.second);
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
      MS_LOG(DEBUG) << "Add front node:" << front_to_backend_node.first->DebugString()
                    << " for kernel graph:" << graph->ToString();
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
    MS_EXCEPTION_IF_NULL(graph_group);
    if (group_name.find(graph_group->group_name_) != std ::string::npos) {
      return graph_group->need_stack_;
    }
  }
  MS_LOG(EXCEPTION) << "Invalid kernel graph group name:" << group_name;
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

NodeWithIndexToContext ControlNodeParser::FetchBackendParameterWithContextByFrontParameter(
  const KernelWithIndex &front_parameter_with_index) {
  MS_EXCEPTION_IF_NULL(front_parameter_with_index.first);
  const auto &iter = front_to_backend_parameters_.find(front_parameter_with_index);
  if (iter == front_to_backend_parameters_.end()) {
    return {};
  }

  for (const auto &node_with_index_to_context : iter->second) {
    const auto &node = node_with_index_to_context.first.first;
    MS_EXCEPTION_IF_NULL(node);
    if (AnfAlgo::GetOutputTensorMemSize(node, node_with_index_to_context.first.second) != 0) {
      return node_with_index_to_context;
    }
    MS_LOG(DEBUG) << "Backend node:" << node->DebugString()
                  << " for front node:" << front_parameter_with_index.first->DebugString()
                  << " index:" << front_parameter_with_index.second << " output size is 0.";
  }
  return {};
}

void ControlNodeParser::CreateDeviceTensors(const std::vector<AnfNodePtr> &control_nodes,
                                            const DeviceContext *const default_context) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch) ||
        common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      auto input_with_indexs = FetchInputNodeByCNode(control_node);
      for (size_t i = 0; i < input_with_indexs.size(); ++i) {
        MS_EXCEPTION_IF_NULL(input_with_indexs[i].first);
        if (IsFrontValueNode(input_with_indexs[i])) {
          CreateDeviceTensorForFrontNode(input_with_indexs[i], default_context);
          (void)front_value_nodes_.emplace(input_with_indexs[i], default_context);
        }
      }
      continue;
    }

    if ((!common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) &&
        (!common::AnfAlgo::IsCallNode(control_node))) {
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
      if (IsFrontValueNode(input_with_index) &&
          front_value_nodes_.find({input_with_index, iter->second[i]}) == front_value_nodes_.end()) {
        MS_EXCEPTION_IF_NULL(input_with_index.first);
        MS_LOG(DEBUG) << "Create device tensor for value node:" << input_with_index.first->DebugString()
                      << " index:" << i << " in control node:" << control_node->DebugString();
        const auto &node_with_index_with_context = FetchBackendParameterWithContextByFrontParameter(input_with_index);
        if (node_with_index_with_context.first.first != nullptr) {
          CreateDeviceTensorForValueNode(input_with_index, node_with_index_with_context.first.first,
                                         node_with_index_with_context.second);
          (void)front_value_nodes_.emplace(input_with_index, node_with_index_with_context.second);
        } else {
          CreateDeviceTensorForFrontNode(input_with_index, default_context);
          (void)front_value_nodes_.emplace(input_with_index, default_context);
        }
      }
    }
  }
}

void ControlNodeParser::FetchFrontValueNode(const std::vector<AnfNodePtr> &control_nodes,
                                            const DeviceContext *const default_context) {
  MS_EXCEPTION_IF_NULL(default_context);

  for (const auto &formal_to_real_parameter : formal_to_real_parameters_) {
    for (const auto &real_parameter_with_index : formal_to_real_parameter.second) {
      if (!IsFrontValueNode(real_parameter_with_index)) {
        continue;
      }

      const auto &node_with_index_to_context =
        FetchBackendParameterWithContextByFrontParameter(real_parameter_with_index);

      if (node_with_index_to_context.first.first != nullptr) {
        (void)front_value_nodes_.emplace(real_parameter_with_index, node_with_index_to_context.second);
        CreateDeviceTensorForValueNode(real_parameter_with_index, node_with_index_to_context.first.first,
                                       node_with_index_to_context.second);
      } else {
        (void)front_value_nodes_.emplace(real_parameter_with_index, default_context);
        CreateDeviceTensorForFrontNode(real_parameter_with_index, default_context);
      }
    }
  }

  // Create device tensors for those value nodes which direct return by a return node.
  CreateDeviceTensors(control_nodes, default_context);
  for (const auto &front_node : front_value_nodes_) {
    MS_EXCEPTION_IF_NULL(front_node.first.first);
    MS_LOG(DEBUG) << "Print front value node:" << front_node.first.first->DebugString()
                  << " addr:" << front_node.first.first << " index:" << front_node.first.second;
  }
}

void ControlNodeParser::ParseFormalToRealParameter(const std::vector<AnfNodePtr> &control_nodes) {
  FormalToRealParameter formal_to_real_parameters;

  // The actual parameters of the function are divided into two parts:
  // 1. Input of partial node.
  // 2. Input of call node.
  for (const auto &node : control_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (common::AnfAlgo::IsCallNode(node)) {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      const auto &inputs = cnode->inputs();
      const auto &func_graphs = FetchFuncGraphbyCallNode(node);
      for (const auto &func_graph : func_graphs) {
        MS_EXCEPTION_IF_NULL(func_graph);
        const auto &parameters = func_graph->parameters();
        for (int i = SizeToInt(inputs.size()) - 1, j = SizeToInt(parameters.size()) - 1; i >= 1 && j >= 0; --i, --j) {
          MS_EXCEPTION_IF_NULL(inputs[IntToSize(i)]);
          MS_EXCEPTION_IF_NULL(parameters[IntToSize(j)]);
          AddFormalToRealParameter(parameters[IntToSize(j)], inputs[IntToSize(i)], call_node_to_func_graphs_,
                                   &formal_to_real_parameters);
        }
      }
    } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      const auto &inputs = cnode->inputs();
      if (inputs.size() <= kPartialFuncGraphPos) {
        MS_LOG(EXCEPTION) << "Invalid input size for partial node:" << node->DebugString();
      }
      auto &func_node = inputs[kPartialFuncGraphPos];
      MS_EXCEPTION_IF_NULL(func_node);
      // Ignore if the node is 'Partial(DeadNode,)'.
      if (IsDeadNode(func_node)) {
        MS_LOG(DEBUG) << "Ignore partial dead node:" << node->DebugString();
        continue;
      }
      const auto &func_graph = GetValueNode<FuncGraphPtr>(func_node);
      if (func_graph == nullptr) {
        MS_LOG(EXCEPTION) << "Invalid funcgraph node:" << func_node->DebugString()
                          << " for partial node:" << node->DebugString();
      }
      const auto &parameters = func_graph->parameters();
      if (inputs.size() - kPartialInputStartPos > parameters.size()) {
        MS_LOG(EXCEPTION) << "Invalid partial input size:" << inputs.size()
                          << " formal parameter size:" << parameters.size();
      }
      for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
        MS_EXCEPTION_IF_NULL(inputs[i]);
        MS_EXCEPTION_IF_NULL(parameters[i - kPartialInputStartPos]);
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
      MS_EXCEPTION_IF_NULL(real_parameter.first);
      if (real_parameter.first->isa<Parameter>()) {
        std::set<KernelWithIndex> invalid_real_parameter{formal_parameter};
        ParseAllRealParameterByFormalParameter(real_parameter, formal_to_real_parameters, &total_real_parameters,
                                               &invalid_real_parameter);
        (void)real_to_formal_parameters_[real_parameter].emplace(formal_parameter);
      } else {
        (void)total_real_parameters.emplace(real_parameter);
      }
    }
    std::swap(formal_to_real_parameters_[formal_parameter], total_real_parameters);
  }

  for (const auto &formal_to_real : formal_to_real_parameters_) {
    for (const auto &real_parameter : formal_to_real.second) {
      MS_EXCEPTION_IF_NULL(formal_to_real.first.first);
      MS_EXCEPTION_IF_NULL(real_parameter.first);
      MS_LOG(DEBUG) << "Print formal to real node, formal:" << formal_to_real.first.first->DebugString()
                    << " real:" << real_parameter.first->DebugString() << " index:" << real_parameter.second;
    }
  }
}

void ControlNodeParser::ParseAllRealParameterByFormalParameter(const KernelWithIndex &formal_parameter,
                                                               const FormalToRealParameter &formal_to_real_parameters,
                                                               std::set<KernelWithIndex> *const total_real_parameters,
                                                               std::set<KernelWithIndex> *invalid_real_parameter) {
  MS_EXCEPTION_IF_NULL(formal_parameter.first);
  MS_EXCEPTION_IF_NULL(total_real_parameters);
  MS_EXCEPTION_IF_NULL(invalid_real_parameter);
  if (invalid_real_parameter->find(formal_parameter) != invalid_real_parameter->end()) {
    return;
  }
  (void)invalid_real_parameter->emplace(formal_parameter);

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
    MS_LOG(EXCEPTION) << "Invalid formal parameter:" << formal_parameter.first->DebugString()
                      << ", maybe there is no call node for funcgraph:"
                      << (formal_parameter.first->func_graph() == nullptr
                            ? "null"
                            : formal_parameter.first->func_graph()->ToString());
  }
  const auto &real_parameters = src_iter->second;
  for (const auto &real_parameter : real_parameters) {
    MS_EXCEPTION_IF_NULL(real_parameter.first);
    (void)total_real_parameters->emplace(real_parameter);
    if (real_parameter.first->isa<Parameter>()) {
      ParseAllRealParameterByFormalParameter(real_parameter, formal_to_real_parameters, total_real_parameters,
                                             invalid_real_parameter);
    }
  }
}

void ControlNodeParser::ParseControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      break;
    }

    const auto &inputs = FetchInputNodeByCNode(control_node);
    for (size_t i = 0; i < inputs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(inputs[i].first);
      MS_LOG(DEBUG) << "Control node:" << control_node->DebugString()
                    << " input node:" << inputs[i].first->DebugString() << " index:" << inputs[i].second;
      if (inputs[i].first->isa<Parameter>()) {
        MS_LOG(DEBUG) << "Control node:" << control_node->DebugString()
                      << " input parameter:" << inputs[i].first->DebugString() << " index:" << inputs[i].second;
        (void)control_node_parameters_.emplace_back(inputs[i]);
        // Set Dynamic shape flag for parameter.
        const auto &parameter = inputs[i].first->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(parameter);
        const auto &base_shape = parameter->Shape();
        if (base_shape == nullptr) {
          continue;
        }
        if ((base_shape->isa<abstract::Shape>() && base_shape->IsDynamic()) ||
            base_shape->isa<abstract::DynamicSequenceShape>()) {
          MS_LOG(INFO) << "Set dynamic shape flag to parameter:" << parameter->DebugString();
          parameter->set_has_dynamic_shape(true);
        }
      }
    }
  }
}

void ControlNodeParser::CreateBranchIDForCallNode(const std::vector<AnfNodePtr> &control_nodes) {
  int branch_id = kMainBranchID;

  for (const auto &control_node : control_nodes) {
    // Root funcgraph does not need to create a gather actor.
    if (common::AnfAlgo::IsCallNode(control_node)) {
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
    MS_EXCEPTION_IF_NULL(graph);
    MS_EXCEPTION_IF_NULL(device_context);
    for (const auto &parameter : graph->input_nodes()) {
      MS_EXCEPTION_IF_NULL(parameter);
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
        for (const auto &real_parameter : real_parameters) {
          MS_EXCEPTION_IF_NULL(real_parameter.first);
          if (real_parameter.first->isa<Parameter>() || real_parameter.first->isa<ValueNode>()) {
            (void)front_to_backend_parameters_[real_parameter].emplace(KernelWithIndex(parameter, 0), device_context);
            MS_LOG(DEBUG) << "Add front node:" << real_parameter.first->DebugString()
                          << " index:" << real_parameter.second
                          << " for backend parameter:" << parameter->DebugString();
          }
        }
      } else if (front_tuple_parameter_with_index.first != nullptr) {
        (void)front_to_backend_parameters_[front_tuple_parameter_with_index].emplace(KernelWithIndex(parameter, 0),
                                                                                     device_context);
      } else {
        (void)front_to_backend_parameters_[{front_node, 0}].emplace(KernelWithIndex(parameter, 0), device_context);
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
        MS_EXCEPTION_IF_NULL(real_parameter);
        if (real_parameter->isa<Parameter>()) {
          front_to_backend_parameters_[real_parameter_with_index].insert(backend_parameters.begin(),
                                                                         backend_parameters.end());
        }
      }
    }
  }
  for (const auto &front_to_backend_parameters : front_to_backend_parameters_) {
    for (const auto &backend_parameter : front_to_backend_parameters.second) {
      MS_EXCEPTION_IF_NULL(front_to_backend_parameters.first.first);
      MS_EXCEPTION_IF_NULL(backend_parameter.first.first);
      MS_LOG(DEBUG) << "Print front to backend parameter, front:"
                    << front_to_backend_parameters.first.first->DebugString()
                    << " index:" << front_to_backend_parameters.first.second
                    << " backend:" << backend_parameter.first.first->DebugString()
                    << " index:" << backend_parameter.first.second << " node addr:" << backend_parameter.first.first;
    }
  }
}

void ControlNodeParser::ParseCallNodeToFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (!common::AnfAlgo::IsCallNode(control_node)) {
      continue;
    }

    const auto &cnode = control_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &func_graphs = abstract::GetFuncGraphsFromAbs(cnode->input(0));
    if (func_graphs.empty()) {
      MS_LOG(EXCEPTION) << "Get func graphs from abstract failed.";
    }
    for (auto func_graph : func_graphs) {
      (void)call_node_to_func_graphs_[control_node].emplace(func_graph);
    }
  }
}

const std::set<FuncGraphPtr> &ControlNodeParser::FetchFuncGraphbyCallNode(const AnfNodePtr &control_node) {
  MS_EXCEPTION_IF_NULL(control_node);
  const auto &iter = call_node_to_func_graphs_.find(control_node);
  if (iter == call_node_to_func_graphs_.end()) {
    MS_LOG(EXCEPTION) << "Invalid call node:" << control_node->DebugString();
  }
  return iter->second;
}

void ControlNodeParser::ParseFrontToBackendKernel(const std::vector<KernelGraphPtr> &graphs,
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
          MS_LOG(DEBUG) << "Add front to backend kernel, front:" << common::AnfAlgo::GetNodeDebugString(front_node)
                        << "index:" << j << " addr:" << front_node
                        << " second:" << common::AnfAlgo::GetNodeDebugString(kernel) << "index:" << j
                        << " addr:" << kernel;
        }
      }
    }

    const auto graph_output_map = graph->graph_output_map();
    for (const auto &output_pair : graph_output_map) {
      MS_EXCEPTION_IF_NULL(output_pair.first.first);
      if (output_pair.first.first->isa<CNode>()) {
        front_to_backend_kernels_[output_pair.second] = {output_pair.first, device_context};
      }
    }
  }
  for (const auto &front_to_backend_kernels : front_to_backend_kernels_) {
    MS_EXCEPTION_IF_NULL(front_to_backend_kernels.first.first);
    MS_EXCEPTION_IF_NULL(front_to_backend_kernels.second.first.first);
    MS_LOG(DEBUG) << "Print front to backend kernel, front node:" << front_to_backend_kernels.first.first->DebugString()
                  << " front index:" << front_to_backend_kernels.first.second
                  << " backend node:" << front_to_backend_kernels.second.first.first->DebugString()
                  << " backend index:" << front_to_backend_kernels.second.first.second;
  }
}

void ControlNodeParser::ParseFirstControlNodeAndKernelGraphForFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    const auto &func_graph = control_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    // In the funcgraph with recursive call node, the call node is marked as level1, and the entrance actor is
    // notified to send data after the call node execute ends. At this time, it is necessary to ensure that the
    // data of all actors in the graph has been processed, so all control nodes of level0 need link control arrow
    // to entrance actor.
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch)) {
      auto iter = node_to_level_.find(control_node);
      if (iter != node_to_level_.end() && iter->second == 0 && (!IsPartialInput(control_node))) {
        (void)func_graph_to_first_control_nodes_[func_graph].emplace(control_node);
      }
    }

    std::set<AnfNodePtr> checked_nodes;
    if (((common::AnfAlgo::IsCallNode(control_node) &&
          unrecursion_call_nodes_.find(control_node) == unrecursion_call_nodes_.end()) ||
         common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) &&
        IsFirstControlNode(control_node, &checked_nodes, unrecursion_call_nodes_)) {
      (void)func_graph_to_first_control_nodes_[func_graph].emplace(control_node);
      MS_LOG(DEBUG) << "Add first control node:" << control_node->DebugString()
                    << " for funcgraph:" << func_graph->ToString();
      if (!common::AnfAlgo::IsCallNode(control_node)) {
        continue;
      }

      // If there is a recursive call node in the funcgraph, the kernel graph of the topo sort before the call node
      // needs to be executed before the call recursion, that is, the kernel graph whose level is less than the call
      // node needs to link a control arrow to the corresponding entry actor.
      // Fetch the level of control node.
      const auto &level_iter = node_to_level_.find(control_node);
      if (level_iter == node_to_level_.end()) {
        MS_LOG(WARNING) << "Failed to get level for call node:" << control_node->DebugString();
        continue;
      }

      // Fetch all of the kernel graph group info whose level less than the control node.
      const auto &graph_group_iter = func_graph_to_kernel_graph_groups_.find(func_graph);
      if (graph_group_iter == func_graph_to_kernel_graph_groups_.end()) {
        continue;
      }
      for (const auto &kernel_graphs : graph_group_iter->second) {
        // Fetch one graph from the group.
        KernelGraphPtr dst_graph = nullptr;
        for (const auto &graph : kernel_graphs) {
          MS_EXCEPTION_IF_NULL(graph);
          if (graph->execution_order().empty()) {
            continue;
          }
          dst_graph = graph;
          break;
        }
        if (dst_graph == nullptr) {
          continue;
        }

        // Fetch the group info.
        const auto &group_info_iter = kernel_graphs_to_group_info_.find(dst_graph);
        if (group_info_iter == kernel_graphs_to_group_info_.end()) {
          MS_LOG(EXCEPTION) << "Failed to get group info for kernel_graph:" << dst_graph->ToString();
        }
        MS_EXCEPTION_IF_NULL(group_info_iter->second);
        if (group_info_iter->second->level_ < level_iter->second) {
          MS_LOG(DEBUG) << "Kernel graph group;" << group_info_iter->second->group_name_
                        << " need link control to entrance of funcgraph:" << func_graph->ToString();
          (void)func_graph_to_first_kernel_graphs_[func_graph].emplace(group_info_iter->second);
        }
      }
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
    MS_EXCEPTION_IF_NULL(call_node);
    const auto &dest_func_graph = call_node->func_graph();
    MS_EXCEPTION_IF_NULL(dest_func_graph);
    std::set<FuncGraphPtr> exexution_func_graphs;
    for (const auto &func_graph : call_node_to_func_graphs.second) {
      FetchAllExecutionFunction(func_graph, &exexution_func_graphs, func_graph_call_relation);
    }
    if (exexution_func_graphs.find(dest_func_graph) == exexution_func_graphs.end()) {
      (void)unrecursion_call_nodes_.emplace(call_node);
      MS_LOG(DEBUG) << "Add unrecursion call control node:" << call_node->DebugString();
    }
  }
}

bool ControlNodeParser::IsCallNodeNeedStack(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  std::set<AnfNodePtr> depend_nodes;

  // Fetch all the side effect inputs of call node.
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    std::vector<AnfNodePtr> monad_nodes = FetchAllMonadNodeByNode(input);
    for (const auto &monad_node : monad_nodes) {
      FetchRealDependNodeByAutoMonad(monad_node, &depend_nodes);
    }
  }

  // Fetch all the data inputs of call node.
  auto input_with_indexs = FetchInputNodeByCNode(node);
  (void)std::for_each(
    input_with_indexs.begin(), input_with_indexs.end(),
    [&depend_nodes](const auto &input_with_index) { (void)depend_nodes.emplace(input_with_index.first); });

  // Check if the call node need a stack.
  for (const auto &depend_node : depend_nodes) {
    MS_EXCEPTION_IF_NULL(depend_node);
    // If the call node has call or recursion graph input, a stack created for the call node is required.
    if (!common::AnfAlgo::IsCallNode(depend_node)) {
      if (!depend_node->isa<CNode>()) {
        continue;
      }
      const auto &graph = FetchKernelGraphByFrontNode(depend_node);
      if (graph == nullptr || (!IsRecursionKernelGraph(graph))) {
        continue;
      }
    }
    return true;
  }
  return false;
}

void ControlNodeParser::ParseNeedStackControlNode(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (common::AnfAlgo::IsCallNode(control_node) && IsCallNodeNeedStack(control_node)) {
      (void)need_stack_control_nodes_.emplace(control_node);
      MS_LOG(DEBUG) << "Add need stack control node:" << control_node->DebugString();
    }
  }

  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (IsInvalidPartial(control_node)) {
      continue;
    }

    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      auto input_with_indexs = FetchInputNodeByCNode(control_node);
      size_t call_input_num = 0;
      for (auto input_with_index : input_with_indexs) {
        if (common::AnfAlgo::IsCallNode(input_with_index.first)) {
          ++call_input_num;
        }
      }

      const auto &cnode = control_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      const auto &inputs = cnode->inputs();
      if (inputs.size() <= kReturnInputPos) {
        MS_LOG(EXCEPTION) << "Invalid return node:" << control_node->DebugString();
      }

      if ((!IsInputInSameLevel(control_node)) ||
          (call_input_num != 0 && (common::AnfAlgo::CheckPrimitiveType(inputs[kReturnInputPos], prim::kPrimDepend)))) {
        (void)need_stack_control_nodes_.emplace(control_node);
        MS_LOG(DEBUG) << "Add need stack control node:" << control_node->DebugString();
      }
    } else if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial) ||
               common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch) ||
               common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      if (!IsInputInSameLevel(control_node)) {
        (void)need_stack_control_nodes_.emplace(control_node);
        MS_LOG(DEBUG) << "Add need stack control node:" << control_node->DebugString();
      }
    }
  }
}

void CollectEffectiveInputByGraph(const KernelGraphPtr &graph, const DeviceContext *const device_context,
                                  KernelGraphGroupInfo *const kernel_graph_group_info) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(kernel_graph_group_info);

  const auto &outputs = kernel_graph_group_info->front_output_nodes_;
  const auto &monad_outputs = kernel_graph_group_info->monad_outputs_;
  const auto &real_parameters = graph->input_nodes();
  for (const auto &parameter : real_parameters) {
    MS_EXCEPTION_IF_NULL(parameter);
    auto front_node_with_index = GetFrontNodeByKernelGraph(parameter, graph.get());
    MS_EXCEPTION_IF_NULL(front_node_with_index.first);
    // If input come from the output of kernel graph belong the same group, it should not be collected in
    // the group inputs.
    if (HasAbstractMonad(front_node_with_index.first) || HasAbstractMonad(parameter) ||
        outputs.find(front_node_with_index) != outputs.end() || front_node_with_index.first->isa<ValueNode>()) {
      // The monad input is used to link the control arrow of the graph. If it comes from other graphs in the same
      // group, it is not used as the monad input of the group.
      if ((HasAbstractMonad(front_node_with_index.first) || HasAbstractMonad(parameter)) &&
          monad_outputs.find(front_node_with_index) == monad_outputs.end()) {
        (void)kernel_graph_group_info->monad_inputs_.emplace(front_node_with_index.first);
        MS_LOG(DEBUG) << "Kernel graph:" << graph->ToString()
                      << " add front monad input node:" << front_node_with_index.first->DebugString();
      }
      continue;
    }
    if (common::AnfAlgo::IsCallNode(front_node_with_index.first)) {
      kernel_graph_group_info->need_stack_ = true;
    }
    MS_LOG(DEBUG) << "Kernel graph:" << graph->ToString()
                  << " add front input node:" << front_node_with_index.first->DebugString()
                  << " index:" << front_node_with_index.second << " backend node:" << parameter->DebugString()
                  << " index:0";
    kernel_graph_group_info->front_input_nodes_[front_node_with_index] = device_context;
  }
}

void CollectEffectiveOutputByGraph(const KernelGraphPtr &graph, DeviceContext *const device_context,
                                   FrontToBackendKernelWithContext *const outputs,
                                   std::set<KernelWithIndex> *monad_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(monad_outputs);

  for (const auto &backend_to_front : graph->graph_output_map()) {
    MS_EXCEPTION_IF_NULL(backend_to_front.first.first);
    MS_EXCEPTION_IF_NULL(backend_to_front.second.first);
    if (HasAbstractMonad(backend_to_front.second.first) || HasAbstractMonad(backend_to_front.first.first) ||
        backend_to_front.first.first->isa<Parameter>() ||
        common::AnfAlgo::CheckPrimitiveType(backend_to_front.second.first, prim::kPrimPartial) ||
        backend_to_front.second.first->isa<ValueNode>()) {
      if (HasAbstractMonad(backend_to_front.second.first) || HasAbstractMonad(backend_to_front.first.first)) {
        (void)monad_outputs->emplace(backend_to_front.second);
      }
      continue;
    }

    // Skip the function input.
    const auto &abstract = backend_to_front.second.first->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = FetchAbstractByIndex(abstract, backend_to_front.second.second);
    MS_EXCEPTION_IF_NULL(real_abstract);
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      continue;
    }

    MS_LOG(DEBUG) << "Kernel graph:" << graph->ToString()
                  << " add front output node:" << backend_to_front.second.first->DebugString()
                  << " index:" << backend_to_front.second.second
                  << " backend node:" << backend_to_front.first.first->DebugString()
                  << " index:" << backend_to_front.first.second;
    (*outputs)[backend_to_front.second] = {backend_to_front.first, device_context};
  }
}

void ControlNodeParser::ParseKernelGraphGroup(const KernelGraphToDeviceContext &kernel_graph_to_device_contexts) {
  for (const auto &func_graph_to_kernel_graph_groups : func_graph_to_kernel_graph_groups_) {
    for (const auto &kernel_graph_group : func_graph_to_kernel_graph_groups.second) {
      if (kernel_graph_group.empty()) {
        continue;
      }

      KernelGraphGroupInfoPtr kernel_graph_group_info = std::make_shared<KernelGraphGroupInfo>();
      MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
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
        (void)kernel_graph_group_info->graphs_.emplace(kernel_graph);

        // Collect inputs in group.
        CollectEffectiveInputByGraph(kernel_graph, iter->second, kernel_graph_group_info.get());

        // Collect outputs in group.
        CollectEffectiveOutputByGraph(kernel_graph, iter->second, &(kernel_graph_group_info->front_output_nodes_),
                                      &(kernel_graph_group_info->monad_outputs_));

        kernel_graphs_to_group_info_[kernel_graph] = kernel_graph_group_info;
      }
      kernel_graph_group_info->group_name_ = "kernel_graph";
      for (const auto &graph : kernel_graph_group_info->graphs_) {
        if (kernel_graph_group_info->need_stack_) {
          MS_LOG(DEBUG) << "Add call input kernel graph:" << graph->ToString();
          (void)call_input_kernel_graphs_.emplace(graph.get());
        }
        kernel_graph_group_info->group_name_ += ("_" + std::to_string(graph->graph_id()));
      }
      MS_LOG(DEBUG) << "Add kernel graph info for group:" << kernel_graph_group_info->group_name_;
      (void)kernel_graph_group_infos_.emplace(kernel_graph_group_info);
    }
  }
}

size_t ControlNodeParser::ParseControlNodeLevel(const AnfNodePtr &node, std::set<AnfNodePtr> *checked_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(checked_nodes);
  if (!node->isa<CNode>() || checked_nodes->find(node) != checked_nodes->end()) {
    return 0;
  }
  (void)checked_nodes->emplace(node);

  auto iter = node_to_level_.find(node);
  if (iter != node_to_level_.end()) {
    return iter->second;
  }

  size_t level = 0;
  const auto &kernel_graph = FetchKernelGraphByFrontNode(node);
  if (kernel_graph == nullptr) {
    // If the kernel graph is not found, it means that the input does not come from the kernel graph, then
    // just continue to traverse the input.
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    for (const auto &input : inputs) {
      size_t tmp_level = ParseControlNodeLevel(input, checked_nodes);
      level = (tmp_level > level ? tmp_level : level);
    }
    return level;
  }

  // If the input comes from the kernel graph, you need to check all the graph's input, not just the node's input.
  auto group_info_iter = kernel_graphs_to_group_info_.find(kernel_graph);
  if (group_info_iter == kernel_graphs_to_group_info_.end()) {
    MS_LOG(EXCEPTION) << "Failed to get kernel graph group info for graph:" << kernel_graph->ToString();
  }
  MS_EXCEPTION_IF_NULL(group_info_iter->second);
  const auto &inputs = group_info_iter->second->front_input_nodes_;
  for (const auto &input : inputs) {
    const auto &input_node = input.first.first;
    size_t tmp_level = ParseControlNodeLevel(input_node, checked_nodes);
    level = (tmp_level > level ? tmp_level : level);
  }
  return level;
}

void ControlNodeParser::ParseNodeLevel(const std::vector<AnfNodePtr> &control_nodes) {
  size_t level = 0;
  // 1. Parse levels of control nodes.
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      node_to_level_[control_node] = level;
      MS_LOG(DEBUG) << "Add level:" << level << " for node:" << control_node->DebugString();
      level = 0;
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &parameters = func_graph->parameters();
      for (const auto &parameter : parameters) {
        MS_EXCEPTION_IF_NULL(parameter);
        MS_LOG(DEBUG) << "Add level:" << level << " for node:" << parameter->DebugString();
        node_to_level_[parameter] = level;
      }
      continue;
    } else if (IsRecursionCallNode(control_node)) {
      ++level;
      MS_LOG(DEBUG) << "Add level:" << level << " for node:" << control_node->DebugString();
      node_to_level_[control_node] = level;
    } else {
      std::set<AnfNodePtr> checked_nodes;
      node_to_level_[control_node] = ParseControlNodeLevel(control_node, &checked_nodes);
      MS_LOG(DEBUG) << "Add level:" << node_to_level_[control_node] << " for node:" << control_node->DebugString();
    }
  }

  // 2. Parse the levels of kernel graph outputs.
  for (const auto &kernel_graph_group_info : kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
    level = 0;
    for (const auto &front_input_node : kernel_graph_group_info->front_input_nodes_) {
      const auto &input_node = front_input_node.first.first;
      auto iter = node_to_level_.find(input_node);
      if (iter != node_to_level_.end() && level < iter->second) {
        level = iter->second;
      }
    }
    for (const auto &front_output_node : kernel_graph_group_info->front_output_nodes_) {
      MS_EXCEPTION_IF_NULL(front_output_node.second.first.first);
      if (front_output_node.second.first.first->isa<Parameter>()) {
        continue;
      }
      const auto &output_node = front_output_node.first.first;
      MS_EXCEPTION_IF_NULL(output_node);
      MS_LOG(DEBUG) << "Add level:" << level << " for node:" << output_node->DebugString();
      node_to_level_[output_node] = level;
    }
  }

  // Parse the levels of kernel graph groups.
  for (const auto &kernel_graph_group_info : kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
    size_t max_level = 0;
    for (const auto &front_input_node : kernel_graph_group_info->front_input_nodes_) {
      const auto &input_node = front_input_node.first.first;
      MS_EXCEPTION_IF_NULL(input_node);
      auto iter = node_to_level_.find(input_node);
      if (iter == node_to_level_.end()) {
        MS_LOG(EXCEPTION) << "Failed to get level by input node:" << input_node->DebugString()
                          << " for kernel graph:" << kernel_graph_group_info->group_name_;
      }
      max_level = (max_level > iter->second ? max_level : iter->second);
    }
    if (max_level > 0) {
      kernel_graph_group_info->need_stack_ = true;
      kernel_graph_group_info->level_ = max_level;
      for (const auto &kernel_graph : kernel_graph_group_info->graphs_) {
        (void)call_input_kernel_graphs_.emplace(kernel_graph.get());
      }
    }
    MS_LOG(DEBUG) << "Kernel graph group:" << kernel_graph_group_info->group_name_
                  << " need stack:" << kernel_graph_group_info->need_stack_
                  << " level:" << kernel_graph_group_info->level_;
  }
}

bool ControlNodeParser::IsInputInSameLevel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return true;
  }

  auto input_with_indexes = FetchInputNodeByCNode(node);
  size_t level = SIZE_MAX;
  for (const auto &input_with_index : input_with_indexes) {
    auto input_node = input_with_index.first;
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<ValueNode>()) {
      continue;
    }
    auto iter = node_to_level_.find(input_node);
    if (iter == node_to_level_.end()) {
      MS_LOG(EXCEPTION) << "Failed to find level by input:" << input_node->DebugString()
                        << " for node:" << node->DebugString();
    }
    if (level == SIZE_MAX) {
      level = iter->second;
      continue;
    }
    if (level != iter->second) {
      return false;
    }
  }
  return true;
}

void ControlNodeParser::CreateDeviceTensorForRootGraphParameter(DeviceContext *const default_context) {
  MS_EXCEPTION_IF_NULL(default_context);
  for (const auto &parameter : root_graph_parameters_) {
    MS_EXCEPTION_IF_NULL(parameter);
    const auto &abstract = parameter->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
    for (size_t i = 0; i < output_num; ++i) {
      KernelWithIndex parameter_with_index(parameter, i);
      if (front_to_backend_parameters_.find(parameter_with_index) == front_to_backend_parameters_.end()) {
        MS_LOG(DEBUG) << "Create device tensor for root graph parameter:" << parameter->DebugString();
        CreateDeviceTensorForFrontNode(parameter_with_index, default_context);
        (void)front_to_backend_parameters_[parameter_with_index].emplace(parameter_with_index, default_context);
      }
    }
  }
}

std::string ControlNodeParser::FetchGroupNameByKernelGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto group_info_iter = kernel_graphs_to_group_info_.find(graph);
  if (group_info_iter == kernel_graphs_to_group_info_.end()) {
    MS_LOG(EXCEPTION) << "Failed to get kernel graph group info for graph:" << graph->ToString();
  }
  MS_EXCEPTION_IF_NULL(group_info_iter->second);
  return group_info_iter->second->group_name_;
}
}  // namespace runtime
}  // namespace mindspore
