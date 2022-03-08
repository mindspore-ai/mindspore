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

#include "runtime/graph_scheduler/control_node_parser.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "include/common/utils/convert_utils.h"
#include "abstract/utils.h"
#include "ir/tensor.h"
#include "abstract/abstract_function.h"

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
  return {};
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
    MS_LOG(DEBUG) << "Real input node:" << real_input.first->DebugString() << " index:" << real_input.second
                  << " for tuple node:" << node_with_index.first->DebugString() << " index:" << node_with_index.second;
    FetchRealParameterByNode(real_input, real_parameters, invalid_call_nodes, call_node_to_func_graphs);
  } else if (common::AnfAlgo::CheckPrimitiveType(node.first, prim::kPrimSwitch)) {
    // If node is a switch node, the real parameters are its both true and false branches.
    const auto cnode = node_with_index.first->cast<CNodePtr>();
    const auto inputs = cnode->inputs();
    for (size_t i = kSwitchTrueBranchPos; i < inputs.size(); ++i) {
      FetchRealParameterByNode({inputs[i], 0}, real_parameters, invalid_call_nodes, call_node_to_func_graphs);
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimSwitchLayer)) {
    // If node is a switchlyaer node, the real parameters are its total branches.
    const auto &switch_layer_cnode = node_with_index.first->cast<CNodePtr>();
    const auto &switch_layer_inputs = switch_layer_cnode->inputs();
    if (switch_layer_inputs.size() != kSwitchLayerInputNum ||
        (!common::AnfAlgo::CheckPrimitiveType(switch_layer_inputs[kSwitchLayerBranchPos], prim::kPrimMakeTuple))) {
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
    MS_LOG(DEBUG) << "Add real parameter:" << node_with_index.first->DebugString()
                  << " index:" << node_with_index.second;
    (void)real_parameters->emplace(node_with_index);
  }
}

// Topologically sort all funcgraphs according to the function call relationship.
std::vector<FuncGraphPtr> TopoSortForFuncGraph(const FuncGraphPtr &root, FuncGraphCallRelation *const edges) {
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
  if (node_value->isa<FuncGraph>() || node_value->isa<Primitive>()) {
    return;
  }

  size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(backend_node, 0);
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(backend_node, 0);
  if (output_type_id == kTypeUnknown) {
    output_type_id = common::AnfAlgo::GetOutputInferDataType(backend_node, 0);
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
  MS_LOG(DEBUG) << "Create address for node:" << common::AnfAlgo::GetNodeDebugString(front_node) << " addr:" << address
                << " size:" << tensor_size;
  AnfAlgo::SetOutputAddr(address, front_node_with_index.second, front_node.get());
  UpdateRefCount(address.get(), true);
}

// Create a device tensor for front node.
// When the condition input of the switch and switchlayer or the output of a subgraph is a parameter or value node,
// there is no corresponding backend node for this parameter, so a device tensor needs to be created for it.
void CreateDeviceTensorForFrontNode(const KernelWithIndex &front_node_with_index, const DeviceContext *device_context) {
  MS_LOG(DEBUG) << "Start create device tensor for front node:" << front_node_with_index.first->DebugString();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &node = front_node_with_index.first;

  TypeId type_id = common::AnfAlgo::GetOutputInferDataType(node, 0);
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
  const PrimitiveSet expand_prims{prim::kPrimMakeTuple, prim::kPrimMakeCSRTensor, prim::kPrimMakeCOOTensor};
  // The MakeTuple/MakeSparse node need expand and recurse.
  if (IsOneOfPrimitiveCNode(real_node, expand_prims)) {
    const auto &cnode = real_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    for (size_t i = kMakeTupleInputStartPos; i < inputs.size(); ++i) {
      const auto &sub_results = FetchInputNodeByNode(inputs[i]);
      (void)results.insert(results.end(), sub_results.begin(), sub_results.end());
    }
    return results;
  }

  // 3. kPrimMakeCSRTensor.
  if (IsCsrNode(real_node) || IsCooNode(real_node)) {
    const auto &cnode = real_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs.size() <= kMakeTensorInputStartPos) {
      MS_LOG(EXCEPTION) << "Invalid make csr tensor node:" << cnode->DebugString();
    }

    // Fetch output put index.
    const auto &prim_node = inputs[0]->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_node);
    const auto &prim_value = prim_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(prim_value);
    const auto &src_node = inputs[kMakeTensorInputStartPos];
    MS_EXCEPTION_IF_NULL(src_node);
    const auto iter = sparse_attr_map.find(prim_value->name());
    // Csr node from the make csr tensor node.
    if (common::AnfAlgo::CheckPrimitiveType(src_node, prim::kPrimMakeCSRTensor) ||
        common::AnfAlgo::CheckPrimitiveType(src_node, prim::kPrimMakeCOOTensor)) {
      const auto &make_tensor_cnode = src_node->cast<CNodePtr>();
      const auto &make_tensor_inputs = make_tensor_cnode->inputs();
      if (make_tensor_inputs.size() <= kMakeCSRTensorInputNum) {
        MS_LOG(EXCEPTION) << "Invalid make csr tensor node:" << cnode->DebugString();
      }
      const auto &sub_results =
        FetchInputNodeByNode(make_tensor_inputs[LongToSize(iter->second) + kMakeTensorInputStartPos]);
      (void)results.insert(results.end(), sub_results.begin(), sub_results.end());
    } else {
      // Csr node from parameter or call node.
      auto abstract = src_node->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      std::vector<size_t> index_stack{LongToSize(iter->second)};
      auto real_indexs = FetchRealIndexByAbstract(abstract, &index_stack);
      (void)std::transform(real_indexs.begin(), real_indexs.end(), std::back_inserter(results),
                           [&src_node](const auto &index) { return KernelWithIndex(src_node, index); });
    }
    return results;
  }

  // 4. One output node.
  const auto &abstract = real_node->abstract();
  if (abstract == nullptr) {
    MS_LOG(WARNING) << "Empty abstract for node:" << real_node->DebugString();
    (void)results.emplace_back(common::AnfAlgo::VisitKernelWithReturnType(real_node, real_index));
    return results;
  }

  // 5 Other.
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
  auto abstract = formal_parameter->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);

  for (size_t i = 0; i < output_num; ++i) {
    std::set<KernelWithIndex> real_parameters;
    std::set<KernelWithIndex> invalid_call_nodes;
    FetchRealParameterByNode({real_parameter, i}, &real_parameters, &invalid_call_nodes, call_node_to_func_graphs);
    if (real_parameters.empty()) {
      MS_LOG(EXCEPTION) << "Failed to find real parameter for formal parameter:" << real_parameter->DebugString();
    }

    for (const auto &parameter : real_parameters) {
      MS_LOG(DEBUG) << "Add formal parameter:" << formal_parameter->DebugString() << " index:" << i
                    << " to real parameter:" << parameter.first->DebugString() << " index:" << parameter.second;
    }
    (*formal_to_real_parameters)[{formal_parameter, i}].insert(real_parameters.begin(), real_parameters.end());
  }
}

// Recursively traverse the input to confirm whether there is an input of recursive call.
bool IsFirstControlNode(const AnfNodePtr &node, std::set<AnfNodePtr> *checked_nodes,
                        std::set<AnfNodePtr> unrecursion_call_nodes) {
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
}  // namespace

KernelWithIndex FetchRealNodeByGetItem(const KernelWithIndex &node_with_index) {
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  std::vector<size_t> index_stack{node_with_index.second};

  const auto &get_item_src_node = common::AnfAlgo::GetTupleIndexes(node_with_index.first, &index_stack);
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

  if (abstract->isa<abstract::AbstractCOOTensor>()) {
    auto coo_abs = abstract->cast<abstract::AbstractCOOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_abs);
    if (index == kCooTensorIndicesIndex) {
      return coo_abs->indices();
    } else if (index == kCooTensorValuesIndex) {
      return coo_abs->values();
    } else if (index >= kCooTensorDenseShapeIndex) {
      return FetchAbstractByIndex(coo_abs->dense_shape(), index - kCooTensorDenseShapeIndex);
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
    size_t tmp_index = common::AnfAlgo::GetOutputNumByAbstract(sub_abstract);
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
  MS_LOG(DEBUG) << "Control node parse start.";
  KernelGraphToDeviceContext kernel_graph_to_device_contexts;
  for (size_t i = 0; i < graphs.size(); ++i) {
    kernel_graph_to_device_contexts[graphs[i]] = device_contexts[i];
  }

  for (const auto &control_node : control_nodes) {
    MS_LOG(DEBUG) << "Print control node:" << control_node->DebugString();
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

  ParseKernelGraphGroup(kernel_graph_to_device_contexts);

  ParseNodeLevel(control_nodes);

  ParseNeedStackControlNode(control_nodes);

  ParseFormalToRealParameter(control_nodes);

  ParseFrontToBackendParameter(graphs, device_contexts);

  CreateDeviceTensorForRootGraphParameter(device_contexts[0]);

  ParseFrontToBackendKernel(graphs, device_contexts);

  ParseDeviceContext(control_nodes, graphs, device_contexts, func_graph_to_kernel_graphs);

  FetchFrontValueNode(control_nodes, device_contexts[0]);

  ParseControlNodeParameter(control_nodes);

  ParseFirstControlNodeAndKernelGraphForFuncGraph(control_nodes);
  MS_LOG(DEBUG) << "Control node parse end.";
}

bool ControlNodeParser::IsControlFlowDataArrow(const KernelGraphPtr &graph, const AnfNodePtr &backend_node) {
  MS_EXCEPTION_IF_NULL(graph);
  // Has no control flow node.
  if (!IsInited()) {
    return false;
  }

  if (graph->is_executing_sink()) {
    MS_LOG(ERROR) << "Not support the execution sink fully in the control flow.";
    return true;
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
  // If parameter is a weight node in root funcgraph, it should be set to kernel actor directly.
  if (IsRootGraphPersistentDeviceTensor(front_node)) {
    MS_LOG(DEBUG) << "backend node:" << backend_node->DebugString()
                  << " front node:" << (front_node == nullptr ? "null" : front_node->DebugString());
    return false;
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

  return find(root_graph_parameters_.begin(), root_graph_parameters_.end(), node) != root_graph_parameters_.end();
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
                                           const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs) {
  if (device_contexts.empty()) {
    MS_LOG(EXCEPTION) << "Invalid device contexts.";
  }

  ParseDeviceContextForFuncGraph(kernel_graphs, device_contexts, func_graph_to_kernel_graphs);
  ParseDeviceContextForReturnNode(device_contexts[0]);
  ParseDeviceContextForCallNode(control_nodes);
  ParseDeviceContextForPartialNode(control_nodes);
}

void ControlNodeParser::ParseDeviceContextForFuncGraph(const std::vector<KernelGraphPtr> &kernel_graphs,
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
      for (size_t i = 0; i < common::AnfAlgo::GetOutputNumByAbstract(abstract); ++i) {
        (void)front_parameters.emplace_back(parameter, i);
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
    auto func_value = GetValueNode<StringImmPtr>(func_node);
    if (func_value != nullptr && func_value->value() == kDeadNodeName) {
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
        size_t index = LongToSize(iter - func_graph->parameters().begin());
        MS_EXCEPTION_IF_NULL(func_graph_iter->second[index]);
        (void)return_device_contexts.emplace_back(func_graph_iter->second[index]);
      } else if (output_node.first->isa<ValueNode>()) {
        // If the output is parameter, used the default context type.
        MS_EXCEPTION_IF_NULL(default_context);
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
    if (group_name.find(graph_group->group_name_) != std ::string::npos) {
      return graph_group->need_stack_;
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

NodeWithContext ControlNodeParser::FetchBackendParameterWithContextByFrontParameter(
  const KernelWithIndex &front_parameter_with_index) {
  const auto &iter = front_to_backend_parameters_.find(front_parameter_with_index);
  if (iter == front_to_backend_parameters_.end()) {
    return {};
  }

  for (const auto &node_with_context : iter->second) {
    MS_EXCEPTION_IF_NULL(node_with_context.first);
    if (AnfAlgo::GetOutputTensorMemSize(node_with_context.first, 0) != 0) {
      return node_with_context;
    }
    MS_LOG(DEBUG) << "Backend node:" << node_with_context.first->DebugString()
                  << " for front node:" << front_parameter_with_index.first->DebugString()
                  << " index:" << front_parameter_with_index.second << " output size is 0.";
  }
  return {};
}

void ControlNodeParser::FetchFrontValueNode(const std::vector<AnfNodePtr> &control_nodes,
                                            const DeviceContext *const default_context) {
  MS_EXCEPTION_IF_NULL(default_context);

  for (const auto &formal_to_real_parameter : formal_to_real_parameters_) {
    for (const auto &real_parameter_with_index : formal_to_real_parameter.second) {
      if (!IsFrontValueNode(real_parameter_with_index)) {
        continue;
      }

      const auto &backend_node_with_context =
        FetchBackendParameterWithContextByFrontParameter(real_parameter_with_index);
      if (backend_node_with_context.first != nullptr) {
        (void)front_value_nodes_.emplace(real_parameter_with_index, backend_node_with_context.second);
        CreateDeviceTensorForValueNode(real_parameter_with_index, backend_node_with_context.first,
                                       backend_node_with_context.second);
      } else {
        (void)front_value_nodes_.emplace(real_parameter_with_index, default_context);
        CreateDeviceTensorForFrontNode(real_parameter_with_index, default_context);
      }
    }
  }

  // Create device tensors for those value nodes which direct return by a return node.
  for (const auto &control_node : control_nodes) {
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
        MS_LOG(DEBUG) << "Create device tensor for value node:" << input_with_index.first->DebugString()
                      << " index:" << i << " in control node:" << control_node->DebugString();
        const auto &backend_node_with_context = FetchBackendParameterWithContextByFrontParameter(input_with_index);
        if (backend_node_with_context.first != nullptr) {
          CreateDeviceTensorForValueNode(input_with_index, backend_node_with_context.first,
                                         backend_node_with_context.second);
          (void)front_value_nodes_.emplace(input_with_index, iter->second[i]);
        } else {
          CreateDeviceTensorForFrontNode(input_with_index, default_context);
          (void)front_value_nodes_.emplace(input_with_index, default_context);
        }
      }
    }
  }
  for (const auto &front_node : front_value_nodes_) {
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
    if (common::AnfAlgo::IsCallNode(node)) {
      const auto &cnode = node->cast<CNodePtr>();
      const auto &inputs = cnode->inputs();
      const auto &func_graphs = FetchFuncGraphbyCallNode(node);
      for (const auto func_graph : func_graphs) {
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
      // Ignore if the node is 'Partial(DeadNode,)'.
      auto func_value = GetValueNode<StringImmPtr>(func_node);
      if (func_value != nullptr && func_value->value() == kDeadNodeName) {
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
      MS_LOG(DEBUG) << "Print formal to real node, formal:" << formal_to_real.first.first->DebugString()
                    << " real:" << real_parameter.first->DebugString() << " index:" << real_parameter.second;
    }
  }
}

void ControlNodeParser::ParseAllRealParameterByFormalParameter(const KernelWithIndex &formal_parameter,
                                                               const FormalToRealParameter &formal_to_real_parameters,
                                                               std::set<KernelWithIndex> *const total_real_parameters,
                                                               std::set<KernelWithIndex> *invalid_real_parameter) {
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
    MS_LOG(EXCEPTION) << "Invalid formal parameter:" << formal_parameter.first->DebugString();
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
    CNodePtr cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      break;
    } else if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial)) {
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
    } else if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch)) {
      if (inputs.size() != kSwitchInputNum) {
        MS_LOG(EXCEPTION) << "Invalid switch node:" << common::AnfAlgo::GetNodeDebugString(control_node);
      }
      if (inputs[kSwitchCondPos]->isa<Parameter>()) {
        (void)control_node_parameters_.emplace_back(inputs[kSwitchCondPos]);
      }
    } else if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      if (inputs.size() != kSwitchLayerInputNum) {
        MS_LOG(EXCEPTION) << "Invalid switch node:" << common::AnfAlgo::GetNodeDebugString(control_node);
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
    for (const auto &parameter : graph->input_nodes()) {
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
            (void)front_to_backend_parameters_[real_parameter].emplace(parameter, device_context);
            MS_LOG(DEBUG) << "Add front node:" << real_parameter.first->DebugString()
                          << " index:" << real_parameter.second
                          << " for backend parameter:" << parameter->DebugString();
          }
        }
      } else if (front_tuple_parameter_with_index.first != nullptr) {
        (void)front_to_backend_parameters_[front_tuple_parameter_with_index].emplace(parameter, device_context);
      } else {
        (void)front_to_backend_parameters_[{front_node, 0}].emplace(parameter, device_context);
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
  for (const auto &front_to_backend_parameters : front_to_backend_parameters_) {
    for (const auto &backend_parameter : front_to_backend_parameters.second) {
      MS_LOG(DEBUG) << "Print front to backend parameter, front:"
                    << front_to_backend_parameters.first.first->DebugString()
                    << " index:" << front_to_backend_parameters.first.second
                    << " backend:" << backend_parameter.first->DebugString()
                    << " node addr:" << backend_parameter.first;
    }
  }
}

FuncGraphPtr GetFuncGraph(const abstract::AbstractBasePtr &abs, const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_CHECK_FAIL(abs != nullptr, "Null abstract, current node: " + anf_node->DebugString());
  if (abs->isa<abstract::FuncGraphAbstractClosure>()) {
    auto abs_func_graph = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
    if (!abs_func_graph->specialized()) {
      MS_LOG(INFO) << "Unspecilized func graph abstract: " << abs_func_graph->ToString()
                   << ", node: " << anf_node->DebugString();
    }
    return abs_func_graph->func_graph();
  }

  if (abs->isa<abstract::PartialAbstractClosure>()) {
    auto abs_partial_closure = abs->cast<abstract::PartialAbstractClosurePtr>();
    auto abs_func = abs_partial_closure->fn();
    return GetFuncGraph(abs_func, anf_node);
  }
  MS_LOG(EXCEPTION) << "Unexpected abs: " << abs->ToString();
}

std::vector<FuncGraphPtr> GetFuncGraphs(const AnfNodePtr &anf_node) {
  if (IsValueNode<FuncGraph>(anf_node)) {
    return {GetValueNode<FuncGraphPtr>(anf_node)};
  }
  auto abs = anf_node->abstract();
  MS_EXCEPTION_IF_CHECK_FAIL(abs != nullptr, "Null abstract of node: " + anf_node->DebugString());
  if (!abs->isa<abstract::AbstractFunction>()) {
    MS_LOG(EXCEPTION) << "Unexpected abs: " << abs->ToString() << ", anf_node: " << anf_node->DebugString();
  }
  auto abs_func = abs->cast<abstract::AbstractFunctionPtr>();
  std::vector<FuncGraphPtr> ret;
  if (abs->isa<abstract::AbstractFuncUnion>()) {
    auto visit_func = [&ret, &anf_node](const abstract::AbstractFuncAtomPtr &poss) {
      ret.emplace_back(GetFuncGraph(poss, anf_node));
    };
    abs_func->Visit(visit_func);
  } else {
    ret.emplace_back(GetFuncGraph(abs_func, anf_node));
  }
  return ret;
}

void ControlNodeParser::ParseCallNodeToFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (!common::AnfAlgo::IsCallNode(control_node)) {
      continue;
    }
    std::vector<FuncGraphPtr> func_graphs;
    func_graphs = GetFuncGraphs(control_node->cast<CNodePtr>()->input(0));

    for (auto func_graph : func_graphs) {
      (void)call_node_to_func_graphs_[control_node].emplace(func_graph);
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
        for (size_t j = 0; j < common::AnfAlgo::GetOutputTensorNum(kernel); ++j) {
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
      if (output_pair.first.first->isa<CNode>()) {
        front_to_backend_kernels_[output_pair.second] = {output_pair.first, device_context};
      }
    }
  }
  for (const auto &front_to_backend_kernels : front_to_backend_kernels_) {
    MS_LOG(DEBUG) << "Print front to backend kernel, front node:" << front_to_backend_kernels.first.first->DebugString()
                  << " front index:" << front_to_backend_kernels.first.second
                  << " backend node:" << front_to_backend_kernels.second.first.first->DebugString()
                  << " backend index:" << front_to_backend_kernels.second.first.second;
  }
}

void ControlNodeParser::ParseFirstControlNodeAndKernelGraphForFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    std::set<AnfNodePtr> checked_nodes;
    if (((common::AnfAlgo::IsCallNode(control_node) &&
          unrecursion_call_nodes_.find(control_node) == unrecursion_call_nodes_.end()) ||
         common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) &&
        IsFirstControlNode(control_node, &checked_nodes, unrecursion_call_nodes_)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
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
        if (group_info_iter->second->level_ < level_iter->second) {
          MS_LOG(DEBUG) << "Kernel graph group;" << group_info_iter->second->group_name_
                        << " need link control to entrance of funcgraph:" << func_graph->ToString();
          func_graph_to_first_kernel_graphs_[func_graph].emplace(group_info_iter->second);
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

  auto input_with_indexs = FetchInputNodeByCNode(node);
  for (const auto &input_with_index : input_with_indexs) {
    MS_EXCEPTION_IF_NULL(input_with_index.first);
    // If the call node has call or recursion graph input, a stack created for the call node is required.
    if (!common::AnfAlgo::IsCallNode(input_with_index.first)) {
      if (!input_with_index.first->isa<CNode>()) {
        continue;
      }
      const auto &graph = FetchKernelGraphByFrontNode(input_with_index.first);
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

void CollectEffectiveInputByGraph(const KernelGraphPtr &graph, const FrontToBackendKernelWithContext &outputs,
                                  DeviceContext *const device_context,
                                  std::map<KernelWithIndex, const DeviceContext *> *const inputs,
                                  bool *const need_stack) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(need_stack);

  const auto &real_parameters = graph->input_nodes();
  for (const auto &parameter : real_parameters) {
    auto front_node_with_index = GetFrontNodeByKernelGraph(parameter, graph.get());
    MS_EXCEPTION_IF_NULL(front_node_with_index.first);
    // If input come from the output of kernel graph belong the same group, it should not be collected in
    // the group inputs.
    if (HasAbstractMonad(front_node_with_index.first) || HasAbstractMonad(parameter) ||
        outputs.find(front_node_with_index) != outputs.end() || front_node_with_index.first->isa<ValueNode>()) {
      continue;
    }
    if (common::AnfAlgo::IsCallNode(front_node_with_index.first)) {
      (*need_stack) = true;
    }
    MS_LOG(DEBUG) << "Kernel graph:" << graph->ToString()
                  << " add front input node:" << front_node_with_index.first->DebugString()
                  << " index:" << front_node_with_index.second << " backend node:" << parameter->DebugString()
                  << " index:0";
    (*inputs)[front_node_with_index] = device_context;
  }
}

void CollectEffectiveOutputByGraph(const KernelGraphPtr &graph, DeviceContext *const device_context,
                                   FrontToBackendKernelWithContext *const outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(outputs);

  for (const auto &backend_to_front : graph->graph_output_map()) {
    if (HasAbstractMonad(backend_to_front.second.first) || HasAbstractMonad(backend_to_front.first.first) ||
        backend_to_front.first.first->isa<Parameter>() ||
        common::AnfAlgo::CheckPrimitiveType(backend_to_front.second.first, prim::kPrimPartial) ||
        backend_to_front.second.first->isa<ValueNode>()) {
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
        CollectEffectiveInputByGraph(kernel_graph, kernel_graph_group_info->front_output_nodes_, iter->second,
                                     &(kernel_graph_group_info->front_input_nodes_),
                                     &(kernel_graph_group_info->need_stack_));

        // Collect outputs in group.
        CollectEffectiveOutputByGraph(kernel_graph, iter->second, &(kernel_graph_group_info->front_output_nodes_));

        kernel_graphs_to_group_info_[kernel_graph] = kernel_graph_group_info;
        if (kernel_graph_group_info->need_stack_) {
          (void)call_input_kernel_graphs_.emplace(kernel_graph.get());
        }
      }
      kernel_graph_group_info->group_name_ = "kernel_graph";
      for (const auto &graph : kernel_graph_group_info->graphs_) {
        kernel_graph_group_info->group_name_ += ("_" + std::to_string(graph->graph_id()));
      }
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

  const auto &inputs = group_info_iter->second->front_input_nodes_;
  for (const auto &input : inputs) {
    const auto &node = input.first.first;
    size_t tmp_level = ParseControlNodeLevel(node, checked_nodes);
    level = (tmp_level > level ? tmp_level : level);
  }
  return level;
}

void ControlNodeParser::ParseNodeLevel(const std::vector<AnfNodePtr> &control_nodes) {
  size_t level = 0;
  // 1. Parse levels of control nodes.
  for (const auto &control_node : control_nodes) {
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      node_to_level_[control_node] = level;
      MS_LOG(DEBUG) << "Add level:" << level << " for node:" << control_node->DebugString();
      level = 0;
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &parameters = func_graph->parameters();
      for (const auto &parameter : parameters) {
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
    level = 0;
    for (const auto &front_input_node : kernel_graph_group_info->front_input_nodes_) {
      const auto &input_node = front_input_node.first.first;
      auto iter = node_to_level_.find(input_node);
      if (iter != node_to_level_.end() && level < iter->second) {
        level = iter->second;
      }
    }
    for (const auto &front_output_node : kernel_graph_group_info->front_output_nodes_) {
      if (front_output_node.second.first.first->isa<Parameter>()) {
        continue;
      }
      const auto &output_node = front_output_node.first.first;
      MS_LOG(DEBUG) << "Add level:" << level << " for node:" << output_node->DebugString();
      node_to_level_[output_node] = level;
    }
  }

  // Parse the levels of kernel graph groups.
  for (const auto &kernel_graph_group_info : kernel_graph_group_infos_) {
    size_t max_level = 0;
    for (const auto &front_input_node : kernel_graph_group_info->front_input_nodes_) {
      const auto &input_node = front_input_node.first.first;
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
    KernelWithIndex parameter_with_index(parameter, 0);
    if (front_to_backend_parameters_.find(parameter_with_index) == front_to_backend_parameters_.end()) {
      MS_LOG(DEBUG) << "Create device tensor for root graph parameter:" << parameter->DebugString();
      CreateDeviceTensorForFrontNode(parameter_with_index, default_context);
      (void)front_to_backend_parameters_[parameter_with_index].emplace(parameter, default_context);
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
