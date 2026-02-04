/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/runtime/control_node_parser.h"
#include <unordered_map>
#include <functional>
#include <map>
#include <algorithm>
#include <utility>
#include <memory>
#include <vector>
#include <set>
#include <string>
#include "primitive/sparse_tensor_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "backend/ge_backend/utils/device_address_utils.h"
#include "utils/convert_utils.h"
#include "abstract/utils.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "ir/tensor_new.h"
#include "ir/dtype/tensor_type.h"
#include "abstract/abstract_function.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"

namespace mindspore {
namespace ge_backend {
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
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid index:" << node_with_index.second
                                      << "for tuple node:" << node->DebugString();
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
  MS_LOG_WITH_NODE(EXCEPTION, node) << "Failed to get real output from node:" << node->DebugString()
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
  if (abstract->isa<abstract::AbstractSequence>()) {
    auto sequence_abstract = abstract->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(sequence_abstract);
    const auto &sub_abstracts = sequence_abstract->elements();
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
      MS_LOG(DEBUG) << "Invalid call node:" << node_with_index.first->DebugString();
      return;
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
      MS_LOG_WITH_NODE(EXCEPTION, switch_layer_cnode)
        << "Invalid switch layer node:" << switch_layer_cnode->DebugString();
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

TypeId FetchTypeIdByNode(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  TypeId type_id = kTypeUnknown;
  if (node->isa<ValueNode>() && node->abstract() != nullptr) {
    // For valuenode, fetch type from abstract.
    const auto &abs = common::AnfAlgo::FetchAbstractByIndex(node->abstract(), index);
    MS_EXCEPTION_IF_NULL(abs);
    const auto &type = abs->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    if (type->isa<TensorType>()) {
      const auto &tensor_type = type->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_type);
      const auto &element = tensor_type->element();
      MS_EXCEPTION_IF_NULL(element);
      type_id = element->type_id();
    } else if (common::AnfAlgo::IsDynamicSequence(node)) {
      const auto &sequence_abs = abs->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(sequence_abs);
      if (sequence_abs->dynamic_len_element_abs() == nullptr) {
        type_id = type->type_id();
      } else {
        if (sequence_abs->dynamic_len_element_abs()->isa<abstract::AbstractTensor>()) {
          const auto &tensor_abs = sequence_abs->dynamic_len_element_abs()->cast<abstract::AbstractTensorPtr>();
          MS_EXCEPTION_IF_NULL(tensor_abs);
          MS_EXCEPTION_IF_NULL(tensor_abs->element());
          const auto &tensor_element_type = tensor_abs->element()->BuildType();
          MS_EXCEPTION_IF_NULL(tensor_element_type);
          return tensor_element_type->type_id();
        }
        const auto &element_type = sequence_abs->dynamic_len_element_abs()->BuildType();
        MS_EXCEPTION_IF_NULL(element_type);
        type_id = element_type->type_id();
      }
    } else {
      type_id = type->type_id();
    }
  } else {
    type_id = common::AnfAlgo::GetOutputInferDataType(node, index);
  }
  return type_id;
}

size_t FetchOutputSizeByValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<Scalar>()) {
    return GetTypeByte(value->type());
  } else if (value->isa<tensor::Tensor>()) {
    const auto &tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    return tensor->Size();
  } else if (value->isa<ValueSequence>()) {
    const auto &value_sequence = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_sequence);
    if (value_sequence->size() == 0) {
      return 0;
    }
    size_t size = 0;
    for (const auto &sub_value : value_sequence->value()) {
      MS_EXCEPTION_IF_NULL(sub_value);
      size += FetchOutputSizeByValue(sub_value);
    }
    return size;
  } else {
    MS_LOG(EXCEPTION) << "Invalid value:" << value->ToString();
  }
}

size_t FetchOutputSizeByNode(const AnfNodePtr &node, size_t index, TypeId type_id) {
  MS_EXCEPTION_IF_NULL(node);
  size_t size = GetTypeByte(TypeIdToType(type_id));
  if (node->isa<ValueNode>() && node->abstract() != nullptr) {
    const auto &abs = common::AnfAlgo::FetchAbstractByIndex(node->abstract(), index);
    MS_EXCEPTION_IF_NULL(abs);
    const auto &shape_ptr = abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    if (shape_ptr->isa<abstract::Shape>()) {
      const auto &shapes = shape_ptr->cast<abstract::ShapePtr>()->shape();
      size = std::accumulate(shapes.begin(), shapes.end(), size, std::multiplies<int64_t>());
    } else if (shape_ptr->isa<abstract::DynamicSequenceShape>()) {
      const auto &value_node = node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      const auto &value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      size = FetchOutputSizeByValue(value);
      MS_LOG(INFO) << "Abstract;" << abs->ToString() << " for node:" << node->DebugString() << " index:" << index
                   << " shape:" << shape_ptr->ToString() << " size:" << size;
    } else if (abs->isa<abstract::AbstractMonad>() || abs->isa<abstract::AbstractScalar>()) {
      MS_LOG(DEBUG) << "For scalar, the output shape is 1.";
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid abstract;" << abs->ToString() << " for node:" << node->DebugString()
                                        << " index:" << index << " shape:" << shape_ptr->ToString();
    }
  } else {
    size = AnfAlgo::GetOutputTensorMemSize(node, index);
  }
  return size;
}

// Create a device tensor for the front node.
// Get the output format and select kernel build info from the backend node corresponding to the front node to
// create the device address.
void CreateDeviceTensorForValueNode(const KernelWithIndex &front_node_with_index, const AnfNodePtr &backend_node) {
  MS_EXCEPTION_IF_NULL(backend_node);
  const auto &front_node = front_node_with_index.first;
  MS_EXCEPTION_IF_NULL(front_node);

  const auto &node_value = front_node->cast<ValueNodePtr>()->value();
  MS_EXCEPTION_IF_NULL(node_value);
  if (node_value->isa<FuncGraph>() || node_value->isa<Primitive>() ||
      (node_value->isa<ValueSequence>() && node_value->cast<ValueSequencePtr>()->size() == 0)) {
    return;
  }

  size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(backend_node, 0);
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(backend_node, 0);
  if (output_type_id == kTypeUnknown) {
    output_type_id = common::AnfAlgo::GetOutputInferDataType(backend_node, 0);
  }
  if (front_node->abstract() != nullptr && front_node->abstract()->isa<abstract::AbstractSequence>() &&
      front_node->abstract()->cast<abstract::AbstractSequencePtr>()->dynamic_len()) {
    tensor_size = FetchOutputSizeByNode(front_node, front_node_with_index.second, output_type_id);
  }
  CreateBuildInfoForFrontNode(front_node_with_index, backend_node);
  // Create device tensor.
  std::string output_format = AnfAlgo::GetOutputFormat(backend_node, 0);

  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {backend_node, 0}, nullptr, tensor_size, output_format, output_type_id, ShapeVector(), device_name, device_id);
  kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(backend_node));
  auto address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(address);
  MS_LOG(DEBUG) << "Create address for front node:" << front_node->DebugString()
                << " backend node:" << backend_node->DebugString() << " index:" << front_node_with_index.second
                << " addr:" << address << " size:" << tensor_size;
  AnfAlgo::SetOutputAddr(address, kernel_tensor->tensor_storage_info(), front_node_with_index.second, front_node);
  auto node_kernel_tensor = AnfAlgo::GetOutputKernelTensor(front_node, front_node_with_index.second, false);
  UpdateRefCount(node_kernel_tensor, true);
}

// Create a device tensor for front node.
// When the condition input of the switch and switchlayer or the output of a subgraph is a parameter or value node,
// there is no corresponding backend node for this parameter, so a device tensor needs to be created for it.
void CreateDeviceTensorForFrontNode(const KernelWithIndex &front_node_with_index) {
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

  if (node->isa<ValueNode>()) {
    const auto &node_value = node->cast<ValueNodePtr>()->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<ValueSequence>() && node_value->cast<ValueSequencePtr>()->size() == 0) {
      return;
    }
  }

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

  const auto &abstract = AnfAlgo::GetNodeAbstractByIndex(front_node_with_index.first, front_node_with_index.second);
  bool is_map_parameter = abstract != nullptr && abstract->isa<abstract::AbstractMapTensor>();
  if (is_map_parameter) {
    backend::ge_backend::DeviceAddressUtils::CreateDeviceAddressByMapTensorNode(front_node_with_index.first,
                                                                                front_node_with_index.second);
    UpdateRefCount(AnfAlgo::GetOutputKernelTensor(front_node_with_index.first, front_node_with_index.second), true);
    return;
  }

  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  // Fetch mem size by shape, the shape is first obtained from the abstract to deal with the scenario where
  // the value node is a multi-level tuple.
  size_t size = FetchOutputSizeByNode(node, front_node_with_index.second, type_id);
  device::DeviceAddressPtr address{nullptr};
  TensorStorageInfoPtr storage_info{nullptr};
  if (node->isa<ValueNode>()) {
    const auto &node_value = node->cast<ValueNodePtr>()->value();
    MS_EXCEPTION_IF_NULL(node_value);
    // Create device tensor.
    const auto &sub_abstract = common::AnfAlgo::FetchAbstractByIndex(node->abstract(), front_node_with_index.second);
    MS_EXCEPTION_IF_NULL(sub_abstract);
    const auto &kernel_tensor =
      AnfAlgo::CreateKernelTensor(sub_abstract->BuildShape(), sub_abstract->BuildType(), sub_abstract->BuildValue(),
                                  nullptr, size, kOpFormat_DEFAULT, type_id, ShapeVector(), device_name, device_id);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(node));
    address = kernel_tensor->device_address();
    storage_info = kernel_tensor->tensor_storage_info();
  } else {
    // Create device tensor.
    const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {node, front_node_with_index.second}, nullptr, size, kOpFormat_DEFAULT, type_id, ShapeVector(), device_name,
      device_id);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(node));
    address = kernel_tensor->device_address();
    storage_info = kernel_tensor->tensor_storage_info();
  }
  MS_EXCEPTION_IF_NULL(address);
  MS_LOG(INFO) << "Create address for node that has no corresponding backend node:"
               << common::AnfAlgo::GetNodeDebugString(node) << " addr:" << address << " size:" << size
               << ", type id:" << type_id;
  AnfAlgo::SetOutputAddr(address, storage_info, front_node_with_index.second, node);
  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, front_node_with_index.second, false);
  UpdateRefCount(kernel_tensor, true);
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
    MS_LOG_WITH_NODE(EXCEPTION, real_node) << "Invalid monad node:" << real_node->DebugString();
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
    MS_LOG(DEBUG) << "Empty abstract for node:" << real_node->DebugString();
    (void)results.emplace_back(common::AnfAlgo::VisitKernelWithReturnType(real_node, real_index));
    return results;
  }

  // 4 Other.
  if (common::AnfAlgo::CheckPrimitiveType(real_node, prim::kPrimTupleGetItem)) {
    if (real_node->cast<CNodePtr>()->HasAttr(kAttrReplaceRealKernelInBackend) && real_node->abstract() != nullptr) {
      size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(real_node->abstract());
      MS_LOG(INFO) << "Fetch an tuple get item with replace flag:" << real_node->DebugString()
                   << " output num:" << output_num;
      for (size_t i = 0; i < output_num; ++i) {
        (void)results.emplace_back(real_node, i);
      }
      return results;
    }
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

// Add formal parameter and real parameter into relationship map.
void AddFormalToRealParameter(const AnfNodePtr &formal_parameter, const AnfNodePtr &real_parameter,
                              const CallNodeToFuncGraph &call_node_to_func_graphs,
                              FormalToRealParameter *const formal_to_real_parameters) {
  MS_EXCEPTION_IF_NULL(formal_parameter);
  MS_EXCEPTION_IF_NULL(real_parameter);
  MS_EXCEPTION_IF_NULL(formal_to_real_parameters);
  auto abstract = formal_parameter->abstract();
  if (abstract == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, formal_parameter) << "Empty abstract for parameter:" << formal_parameter->DebugString();
  }
  size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);

  for (size_t i = 0; i < output_num; ++i) {
    std::set<KernelWithIndex> real_parameters;
    std::set<KernelWithIndex> invalid_call_nodes;
    FetchRealParameterByNode({real_parameter, i}, &real_parameters, &invalid_call_nodes, call_node_to_func_graphs);
    if (real_parameters.empty()) {
      MS_LOG(DEBUG) << "Failed to find real parameter for formal parameter:" << real_parameter->DebugString();
      continue;
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

bool IsValidBackendParameter(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (node->abstract() == nullptr) {
    return true;
  }
  if (node->abstract()->isa<abstract::AbstractAny>()) {
    return false;
  }
  const auto &shape = node->abstract()->BuildShape();
  if (shape == nullptr || shape->IsDynamic()) {
    return false;
  }
  return true;
}
}  // namespace
void CreateBuildInfoForFrontNode(const KernelWithIndex &front_node_with_index, const AnfNodePtr &backend_node) {
  MS_EXCEPTION_IF_NULL(front_node_with_index.first);
  MS_EXCEPTION_IF_NULL(backend_node);
  const auto &front_node = front_node_with_index.first;
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
}

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
    MS_LOG_WITH_NODE(EXCEPTION, get_item_src_node) << "Failed to find index for node:" << get_item_src_node;
  }
  if (indexes.size() > 1) {
    MS_LOG(DEBUG) << "Output size:" << indexes.size() << " for node:" << get_item_src_node->DebugString()
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
    MS_LOG_WITH_NODE(EXCEPTION, backend_node)
      << "Cannot find front node for backend node:" << backend_node->DebugString() << " in graph:" << graph->ToString();
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
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid switch node:" << node->DebugString();
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
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid switch node:" << node->DebugString();
  }
  const auto &branch_node = inputs[kSwitchTrueBranchIndex];
  MS_EXCEPTION_IF_NULL(branch_node);
  const auto &branch_abstract = branch_node->abstract();
  // If abstract is empty, the default is true.
  if (branch_abstract == nullptr) {
    MS_LOG(DEBUG) << "Failed to get abstract by true branch input of switch node:" << node->DebugString();
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
      MS_LOG(DEBUG) << "Failed to get abstract by true branch input of switch node:" << node->DebugString();
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
    MS_EXCEPTION_IF_NULL(depend_nodes);
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
    for (auto &weak_input : cnode->weak_inputs()) {
      auto input = weak_input.lock();
      MS_EXCEPTION_IF_NULL(input);
      const auto &result = FetchAllMonadNodeByNode(input);
      (void)results.insert(results.end(), result.begin(), result.end());
    }
  }
  return results;
}

void ControlNodeParser::Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
                              const FuncGraphPtr &root_graph,
                              const FuncGraphToKernelGraphGroup &func_graph_to_kernel_graphs) {
  if (control_nodes.size() <= 1) {
    MS_LOG(DEBUG) << "Control node parser is not inited.";
    return;
  }
  MS_LOG(INFO) << "Control node parse start.";

  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    MS_LOG(DEBUG) << "Print control node:" << control_node->DebugString();
  }

  is_inited_ = true;

  root_func_graph_ = root_graph;

  root_graph_parameters_ = root_graph->parameters();
  for (const auto &parameter : root_graph_parameters_) {
    MS_LOG(DEBUG) << "Root graph parameter:" << (parameter == nullptr ? "null" : parameter->DebugString());
  }
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

  ParseKernelGraphGroup();

  ParseNodeLevel(control_nodes);

  ParseNeedStackControlNode(control_nodes);

  ParseFormalToRealParameter(control_nodes);

  ParseFrontToBackendParameter(graphs);

  CreateDeviceTensorForRootGraphParameter();

  ParseFrontToBackendKernel(graphs);

  FetchFrontValueNode(control_nodes);

  ParseControlNodeParameter(control_nodes);

  ParseFirstControlNodeAndKernelGraphForFuncGraph(control_nodes);

  ParseDynamicLenFormalParameter(control_nodes);

  ParserSinglePartialFuncgraph(control_nodes);
  MS_LOG(INFO) << "Control node parse end.";
}

namespace {
void GetArgumentIndexForDynamicLenParameter(const abstract::AbstractBasePtr &argument_abs, size_t argument_index,
                                            const abstract::AbstractBasePtr &parameter_abs,
                                            mindspore::HashMap<size_t, size_t> *indexes) {
  if (argument_abs == nullptr || parameter_abs == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(indexes);
  if ((!argument_abs->isa<abstract::AbstractSequence>()) || (!parameter_abs->isa<abstract::AbstractSequence>())) {
    return;
  }
  const auto &arg_seq_abs = argument_abs->cast<abstract::AbstractSequencePtr>();
  const auto &para_seq_abs = parameter_abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(arg_seq_abs);
  MS_EXCEPTION_IF_NULL(para_seq_abs);
  if (arg_seq_abs->dynamic_len() && para_seq_abs->dynamic_len()) {
    return;
  }
  if ((!arg_seq_abs->dynamic_len()) && para_seq_abs->dynamic_len()) {
    MS_LOG(DEBUG) << "Add argument index:" << argument_index << " size:" << arg_seq_abs->size();
    (*indexes)[argument_index] = arg_seq_abs->size();
    return;
  }
  if (arg_seq_abs->dynamic_len() || para_seq_abs->dynamic_len() || arg_seq_abs->size() != para_seq_abs->size()) {
    MS_LOG(EXCEPTION) << "Invalid dynamic len flag for argument abstract:" << arg_seq_abs->ToString()
                      << " parameter abstract:" << para_seq_abs->ToString();
  }
  size_t start_index = argument_index;
  for (size_t i = 0; i < arg_seq_abs->size(); ++i) {
    GetArgumentIndexForDynamicLenParameter(arg_seq_abs->elements()[i], start_index, para_seq_abs->elements()[i],
                                           indexes);
    start_index += common::AnfAlgo::GetOutputNumByAbstract(arg_seq_abs->elements()[i]);
  }
}

void PrintGraphGroupInfo(const std::set<KernelGraphGroupInfoPtr> &kernel_graph_group_infos) {
  for (const auto &group : kernel_graph_group_infos) {
    MS_EXCEPTION_IF_NULL(group);
    for (const auto &graph : group->graphs_) {
      MS_EXCEPTION_IF_NULL(graph);
      MS_LOG(WARNING) << "Group:" << group->group_name_ << " graph:" << graph->ToString() << " level:" << group->level_;
      for (const auto &input : group->front_input_nodes_) {
        MS_EXCEPTION_IF_NULL(input.first);
        MS_LOG(WARNING) << "Input node:" << input.first->DebugString()
                        << " full name:" << input.first->fullname_with_scope() << " node ptr:" << input.first
                        << " index:" << input.second;
      }
      for (const auto &output : group->front_output_nodes_) {
        MS_EXCEPTION_IF_NULL(output.first.first);
        MS_EXCEPTION_IF_NULL(output.second.first);
        MS_LOG(WARNING) << "Output front node:" << output.first.first->DebugString()
                        << " full name:" << output.first.first->fullname_with_scope()
                        << " node ptr:" << output.first.first << " index:" << output.first.second
                        << " backend node:" << output.second.first->DebugString()
                        << " full name:" << output.second.first->fullname_with_scope()
                        << " node ptr:" << output.second.first << " index:" << output.second.second;
      }
    }
  }
}
}  // namespace

void ControlNodeParser::ParseDynamicLenFormalParameterByCallNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &func_graphs = abstract::GetFuncGraphsFromCallNode(cnode);
  if (func_graphs.empty()) {
    MS_LOG(EXCEPTION) << "Get func_graph from abstract failed.";
  }
  mindspore::HashMap<size_t, size_t> sequence_indexes;
  for (auto func_graph : func_graphs) {
    MS_EXCEPTION_IF_NULL(func_graph);
    // Check the consistency of return outputs and call outputs.
    MS_EXCEPTION_IF_NULL(func_graph->return_node());
    mindspore::HashMap<size_t, size_t> return_sequence_indexes;
    GetArgumentIndexForDynamicLenParameter(func_graph->return_node()->abstract(), 0, node->abstract(),
                                           &return_sequence_indexes);
    if (!return_sequence_indexes.empty()) {
      return_to_call_with_dynamic_sequence_index_[func_graph->return_node()][node] = return_sequence_indexes;
    }
    // Check the consistency of arguments and parameters.
    if (cnode->inputs().empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Invalid cnode:" << cnode->DebugString();
    }
    size_t args_num = cnode->size() - 1;
    size_t para_num = func_graph->parameters().size();
    MS_LOG(DEBUG) << "for call node:" << cnode->DebugString() << " arg size:" << args_num << " para size:" << para_num;
    if (args_num > para_num) {
      MS_LOG(EXCEPTION) << "Invalid args num:" << args_num << " for funcgraph:" << func_graph->ToString()
                        << " parameters num:" << func_graph->parameters().size();
    }
    size_t start_index = 1;
    for (size_t i = 0; i < args_num; ++i) {
      MS_EXCEPTION_IF_NULL(cnode->input(i + 1));
      MS_EXCEPTION_IF_NULL((func_graph->parameters())[i + para_num - args_num]);
      MS_LOG(DEBUG) << "Check formal parameter:" << cnode->input(i + 1)->DebugString()
                    << " real node:" << (func_graph->parameters())[i + para_num - args_num]->DebugString();
      GetArgumentIndexForDynamicLenParameter(cnode->input(i + 1)->abstract(), start_index,
                                             (func_graph->parameters())[i + para_num - args_num]->abstract(),
                                             &sequence_indexes);
      start_index += common::AnfAlgo::GetOutputNumByAbstract(cnode->input(i + 1)->abstract());
    }
    if (!sequence_indexes.empty()) {
      for (const auto &pair : sequence_indexes) {
        MS_LOG(DEBUG) << "Add dynamic len formal parameter for call node:" << node->DebugString()
                      << " funcgraph:" << func_graph->ToString() << " argument index:" << pair.first
                      << " size:" << pair.second;
      }
      control_node_to_funcgraph_with_dynamic_sequence_index_[node][func_graph.get()] = sequence_indexes;
    }
  }
}

void ControlNodeParser::ParseDynamicLenFormalParameterByPartial(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  size_t input_num = cnode->size();
  if (input_num <= kPartialFuncGraphPos || cnode->input(kPartialFuncGraphPos) == nullptr ||
      (!cnode->input(kPartialFuncGraphPos)->isa<ValueNode>())) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid partial node:" << node->DebugString();
  }
  const auto &func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kPartialFuncGraphPos));
  if (func_graph == nullptr) {
    MS_LOG(DEBUG) << "Failed to get funcgraph in partial node:" << node->DebugString();
    return;
  }
  if (func_graph->parameters().size() < input_num - kPartialInputStartPos) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Invalid args num:" << input_num - kPartialInputStartPos
                                       << " in partial node:" << cnode->DebugString()
                                       << " for fungraph:" << func_graph->ToString()
                                       << " parameter num:" << func_graph->parameters().size();
  }
  size_t start_index = 1;
  mindspore::HashMap<size_t, size_t> sequence_indexes;
  for (size_t i = kPartialInputStartPos; i < input_num; ++i) {
    MS_EXCEPTION_IF_NULL(cnode->input(i));
    MS_EXCEPTION_IF_NULL(func_graph->parameters()[i - kPartialInputStartPos]);
    GetArgumentIndexForDynamicLenParameter(cnode->input(i)->abstract(), start_index,
                                           func_graph->parameters()[i - kPartialInputStartPos]->abstract(),
                                           &sequence_indexes);
    start_index += common::AnfAlgo::GetOutputNumByAbstract(cnode->input(i)->abstract());
  }
  if (!sequence_indexes.empty()) {
    mindspore::HashMap<size_t, size_t> new_sequence_indexes;
    for (const auto &index_pair : sequence_indexes) {
      new_sequence_indexes[index_pair.first + 1] = index_pair.second;
    }
    control_node_to_funcgraph_with_dynamic_sequence_index_[node][func_graph.get()] = new_sequence_indexes;
  }
}

void ControlNodeParser::ParseDynamicLenFormalParameter(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &node : control_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (common::AnfAlgo::IsCallNode(node)) {
      ParseDynamicLenFormalParameterByCallNode(node);
    } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      ParseDynamicLenFormalParameterByPartial(node);
    }
  }
  for (const auto &node_to_func_with_index : control_node_to_funcgraph_with_dynamic_sequence_index_) {
    const auto &node = node_to_func_with_index.first;
    MS_EXCEPTION_IF_NULL(node);
    for (const auto &func_with_index : node_to_func_with_index.second) {
      const auto &func_graph = func_with_index.first;
      MS_EXCEPTION_IF_NULL(func_graph);
      for (const auto &indexes : func_with_index.second) {
        MS_LOG(DEBUG) << "Node:" << node->DebugString() << " func_graph:" << func_graph->ToString()
                      << " start index:" << indexes.first << " size:" << indexes.second;
      }
    }
  }
  for (const auto &node_to_call_with_index : return_to_call_with_dynamic_sequence_index_) {
    const auto &node = node_to_call_with_index.first;
    MS_EXCEPTION_IF_NULL(node);
    for (const auto &call_with_index : node_to_call_with_index.second) {
      const auto &call = call_with_index.first;
      MS_EXCEPTION_IF_NULL(call);
      for (const auto &indexes : call_with_index.second) {
        MS_LOG(DEBUG) << "Node:" << node->DebugString() << " call node:" << call->DebugString()
                      << " start index:" << indexes.first << " size:" << indexes.second;
      }
    }
  }
}

bool IsSameInputNum(const std::vector<AnfNodePtr> &nodes) {
  if (nodes.size() <= 1) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(nodes[0]);
  const auto &first_node = nodes[0]->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(first_node);
  size_t input_num = first_node->size();
  for (size_t i = 1; i < nodes.size(); ++i) {
    MS_EXCEPTION_IF_NULL(nodes[i]);
    const auto &cnode = nodes[i]->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (input_num != cnode->size()) {
      return false;
    }
  }
  return true;
}

void ControlNodeParser::ParserSinglePartialFuncgraph(const std::vector<AnfNodePtr> &control_nodes) {
  std::unordered_map<FuncGraphPtr, std::vector<AnfNodePtr>> func_graph_to_call_node;
  for (const auto &pair : call_node_to_func_graphs_) {
    for (const auto &func_graph : pair.second) {
      func_graph_to_call_node[func_graph].emplace_back(pair.first);
    }
  }

  std::unordered_map<FuncGraphPtr, std::vector<AnfNodePtr>> func_graph_to_partial_node;
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (!common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial)) {
      continue;
    }
    const auto &cnode = control_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() <= 1) {
      continue;
    }
    const auto &graph_value_node = cnode->input(1);
    if (graph_value_node == nullptr || !IsValueNode<FuncGraph>(graph_value_node)) {
      continue;
    }
    const auto &func_graph = GetValueNode<FuncGraphPtr>(graph_value_node);
    if (func_graph == nullptr) {
      continue;
    }
    func_graph_to_partial_node[func_graph].emplace_back(control_node);
  }

  for (const auto &pair : func_graph_to_call_node) {
    const auto &func_graph = pair.first;
    if (func_graph == nullptr || pair.second.empty() || !IsSameInputNum(pair.second) ||
        func_graph_to_partial_node.find(func_graph) == func_graph_to_partial_node.end() ||
        func_graph_to_partial_node[func_graph].size() != 1) {
      continue;
    }
    const auto &partial_node = func_graph_to_partial_node[func_graph][0];
    MS_EXCEPTION_IF_NULL(partial_node);
    const auto &partial_cnode = partial_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_cnode);
    size_t partial_arg_size = partial_cnode->size() - 2;
    size_t call_arg_size = pair.second[0]->cast<CNodePtr>()->size() - 1;
    size_t para_size = func_graph->parameters().size();
    if (partial_arg_size + call_arg_size != para_size) {
      MS_LOG(WARNING) << "Invalid args size for partial:" << partial_cnode->DebugString()
                      << " and call node:" << pair.second[0]->DebugString() << " parameter size:" << para_size;
      continue;
    }
    func_graph_to_partial_node_[func_graph] = partial_node;
    MS_LOG(INFO) << "Add single partial:" << partial_node->DebugString() << " to funcgraph:" << func_graph->ToString();
  }
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
    tensor = tensor::from_scalar(int8_t('U'), TypeIdToType(kNumberTypeInt8));
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
        if ((!IsParallelCallRecursionGraph(call_nodes[i], call_nodes[j], func_graph_to_call_nodes_)) ||
            IsTopoDependNode(call_nodes[i], call_nodes[j], &checked_nodes)) {
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
        MS_EXCEPTION_IF_NULL(new_depend);
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
  return unrecursion_call_nodes_.find(node) == unrecursion_call_nodes_.end();
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
    const auto &node = front_input_node.first;
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
    MS_LOG_WITH_NODE(EXCEPTION, call_node) << "Invalid branch id for call_node:" << call_node->DebugString();
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
    return iter->second;
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

KernelWithIndex ControlNodeParser::FetchBackendParameterWithContextByFrontParameter(
  const KernelWithIndex &front_parameter_with_index) {
  MS_EXCEPTION_IF_NULL(front_parameter_with_index.first);
  const auto &iter = front_to_backend_parameters_.find(front_parameter_with_index);
  if (iter == front_to_backend_parameters_.end()) {
    return {};
  }

  for (const auto &node_with_index : iter->second) {
    const auto &node = node_with_index.first;
    MS_EXCEPTION_IF_NULL(node);
    if (AnfAlgo::GetOutputTensorMemSize(node, node_with_index.second) != 0) {
      return node_with_index;
    }
    const auto &abstract =
      AnfAlgo::GetNodeAbstractByIndex(front_parameter_with_index.first, front_parameter_with_index.second);
    if (abstract != nullptr && abstract->isa<abstract::AbstractMapTensor>()) {
      return node_with_index;
    }
    MS_LOG(DEBUG) << "Backend node:" << node->DebugString()
                  << " for front node:" << front_parameter_with_index.first->DebugString()
                  << " index:" << front_parameter_with_index.second << " output size is 0.";
  }
  return {};
}

void ControlNodeParser::CreateDeviceTensors(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch) ||
        common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      auto input_with_indexs = FetchInputNodeByCNode(control_node);
      for (size_t i = 0; i < input_with_indexs.size(); ++i) {
        MS_EXCEPTION_IF_NULL(input_with_indexs[i].first);
        if (IsFrontValueNode(input_with_indexs[i])) {
          CreateDeviceTensorForFrontNode(input_with_indexs[i]);
          (void)front_value_nodes_.emplace(input_with_indexs[i]);
        }
      }
      continue;
    }

    if ((!common::AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) &&
        (!common::AnfAlgo::IsCallNode(control_node))) {
      continue;
    }

    auto input_with_indexs = FetchInputNodeByCNode(control_node);
    for (size_t i = 0; i < input_with_indexs.size(); ++i) {
      const auto &input_with_index = input_with_indexs[i];
      if (IsFrontValueNode(input_with_index) && front_value_nodes_.find(input_with_index) == front_value_nodes_.end()) {
        MS_EXCEPTION_IF_NULL(input_with_index.first);
        MS_LOG(DEBUG) << "Create device tensor for value node:" << input_with_index.first->DebugString()
                      << " index:" << i << " in control node:" << control_node->DebugString();
        const auto &node_with_index = FetchBackendParameterWithContextByFrontParameter(input_with_index);
        const auto &backend_node = node_with_index.first;
        if (IsValidBackendParameter(backend_node)) {
          CreateDeviceTensorForValueNode(input_with_index, backend_node);
          (void)front_value_nodes_.emplace(input_with_index);
        } else {
          CreateDeviceTensorForFrontNode(input_with_index);
          (void)front_value_nodes_.emplace(input_with_index);
        }
      }
    }
  }
}

void ControlNodeParser::FetchFrontValueNode(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &formal_to_real_parameter : formal_to_real_parameters_) {
    for (const auto &real_parameter_with_index : formal_to_real_parameter.second) {
      if (!IsFrontValueNode(real_parameter_with_index)) {
        continue;
      }

      const auto &node_with_index = FetchBackendParameterWithContextByFrontParameter(real_parameter_with_index);
      const auto &backend_node = node_with_index.first;
      if (IsValidBackendParameter(backend_node)) {
        (void)front_value_nodes_.emplace(real_parameter_with_index);
        CreateDeviceTensorForValueNode(real_parameter_with_index, backend_node);
      } else {
        (void)front_value_nodes_.emplace(real_parameter_with_index);
        CreateDeviceTensorForFrontNode(real_parameter_with_index);
      }
    }
  }

  // Create device tensors for those value nodes which direct return by a return node.
  CreateDeviceTensors(control_nodes);
  for (const auto &front_node : front_value_nodes_) {
    MS_EXCEPTION_IF_NULL(front_node.first);
    MS_LOG(DEBUG) << "Print front value node:" << front_node.first->DebugString() << " addr:" << front_node.first
                  << " index:" << front_node.second;
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
        MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid input size for partial node:" << node->DebugString();
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
        MS_LOG_WITH_NODE(EXCEPTION, node)
          << "Invalid funcgraph node:" << func_node->DebugString() << " for partial node:" << node->DebugString();
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
    MS_LOG(DEBUG) << "Invalid formal parameter:" << formal_parameter.first->DebugString()
                  << ", maybe there is no call node for funcgraph:"
                  << (formal_parameter.first->func_graph() == nullptr
                        ? "null"
                        : formal_parameter.first->func_graph()->ToString());
    return;
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
      MS_LOG(DEBUG) << "control node:" << control_node->DebugString()
                    << " branch id:" << call_node_to_branch_id_[control_node];
    }
  }
}

void ControlNodeParser::ParseFrontToBackendParameter(const std::vector<KernelGraphPtr> &graphs) {
  // Fetch the mapping relationship between front parameters and backend parameters in the kernel graphs.
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    MS_EXCEPTION_IF_NULL(graph);
    for (const auto &parameter : graph->input_nodes()) {
      MS_EXCEPTION_IF_NULL(parameter);
      const auto &front_node = graph->GetFrontAnfByBackendAnf(parameter);
      const auto &front_node_with_index = graph->GetFrontNodeByInternalParameter(parameter);
      const auto &front_tuple_parameter_with_index = graph->GetElementInTupleBackendFrontIndexMap(parameter);
      if (front_node == nullptr && front_node_with_index.first == nullptr &&
          front_tuple_parameter_with_index.first == nullptr) {
        MS_LOG_WITH_NODE(EXCEPTION, parameter)
          << "Invalid backend parameter:" << parameter->DebugString() << " for kernel graph:" << graph->ToString();
      }

      if (front_node_with_index.first != nullptr) {
        std::set<KernelWithIndex> real_parameters;
        std::set<KernelWithIndex> invalid_call_nodes;
        FetchRealParameterByNode(front_node_with_index, &real_parameters, &invalid_call_nodes,
                                 call_node_to_func_graphs_);
        for (const auto &real_parameter : real_parameters) {
          MS_EXCEPTION_IF_NULL(real_parameter.first);
          if (real_parameter.first->isa<Parameter>() || real_parameter.first->isa<ValueNode>()) {
            (void)front_to_backend_parameters_[real_parameter].emplace(KernelWithIndex(parameter, 0));
            MS_LOG(DEBUG) << "Add front node:" << real_parameter.first->DebugString()
                          << " index:" << real_parameter.second
                          << " for backend parameter:" << parameter->DebugString();
          }
        }
      } else if (front_tuple_parameter_with_index.first != nullptr) {
        (void)front_to_backend_parameters_[front_tuple_parameter_with_index].emplace(KernelWithIndex(parameter, 0));
      } else {
        (void)front_to_backend_parameters_[{front_node, 0}].emplace(KernelWithIndex(parameter, 0));
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
      MS_EXCEPTION_IF_NULL(backend_parameter.first);
      MS_LOG(DEBUG) << "Print front to backend parameter, front:"
                    << front_to_backend_parameters.first.first->DebugString()
                    << " index:" << front_to_backend_parameters.first.second
                    << " backend:" << backend_parameter.first->DebugString() << " index:" << backend_parameter.second
                    << " node addr:" << backend_parameter.first;
    }
  }
}

void ControlNodeParser::ParseCallNodeToFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);
    if (!common::AnfAlgo::IsCallNode(control_node)) {
      continue;
    }

    const auto &belong_func_graph = control_node->func_graph();
    MS_EXCEPTION_IF_NULL(belong_func_graph);
    (void)func_graph_to_call_nodes_[belong_func_graph].emplace(control_node);

    const auto &cnode = control_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &func_graphs = abstract::GetFuncGraphsFromCallNode(cnode);
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
    MS_LOG_WITH_NODE(EXCEPTION, control_node) << "Invalid call node:" << control_node->DebugString();
  }
  return iter->second;
}

void ControlNodeParser::ParseFrontToBackendKernel(const std::vector<KernelGraphPtr> &graphs) {
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    MS_EXCEPTION_IF_NULL(graph);
    auto execution_order = graph->execution_order();
    for (auto &kernel : execution_order) {
      auto front_node = graph->GetFrontAnfByBackendAnf(kernel);
      if (front_node != nullptr) {
        for (size_t j = 0; j < AnfAlgo::GetOutputTensorNum(kernel); ++j) {
          front_to_backend_kernels_[{front_node, j}] = {kernel, j};
          MS_LOG(DEBUG) << "Add front to backend kernel, front:" << common::AnfAlgo::GetNodeDebugString(front_node)
                        << "index:" << j << " addr:" << front_node
                        << " second:" << common::AnfAlgo::GetNodeDebugString(kernel) << "index:" << j
                        << " addr:" << kernel;
        }
      }
    }

    for (const auto &output_pair : graph->front_node_to_graph_output_map()) {
      MS_EXCEPTION_IF_NULL(output_pair.second.first);
      auto real_node = output_pair.second.first;
      // get realnode for callinline
      if (real_node->isa<CNode>()) {
        front_to_backend_kernels_[output_pair.first] = output_pair.second;
      }
    }
  }
  for (const auto &front_to_backend_kernels : front_to_backend_kernels_) {
    MS_EXCEPTION_IF_NULL(front_to_backend_kernels.first.first);
    MS_EXCEPTION_IF_NULL(front_to_backend_kernels.second.first);
    MS_LOG(DEBUG) << "Print front to backend kernel, front node:" << front_to_backend_kernels.first.first->DebugString()
                  << " front index:" << front_to_backend_kernels.first.second
                  << " backend node:" << front_to_backend_kernels.second.first->DebugString()
                  << " backend index:" << front_to_backend_kernels.second.second;
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
        MS_LOG(DEBUG) << "Failed to get level for call node:" << control_node->DebugString();
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
        MS_LOG_WITH_NODE(EXCEPTION, control_node) << "Invalid return node:" << control_node->DebugString();
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

void CollectEffectiveInputByGraph(const KernelGraphPtr &graph, KernelGraphGroupInfo *const kernel_graph_group_info) {
  MS_EXCEPTION_IF_NULL(graph);
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
                  << " fullname:" << front_node_with_index.first->fullname_with_scope()
                  << " node ptr:" << front_node_with_index.first << " index:" << front_node_with_index.second
                  << " backend node:" << parameter->DebugString() << " index:0";
    kernel_graph_group_info->front_input_nodes_.insert(front_node_with_index);
  }
}

void CollectEffectiveOutputByGraph(const KernelGraphPtr &graph,
                                   std::map<KernelWithIndex, KernelWithIndex> *const outputs,
                                   std::set<KernelWithIndex> *monad_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(monad_outputs);

  for (const auto &front_to_backend : graph->front_node_to_graph_output_map()) {
    MS_EXCEPTION_IF_NULL(front_to_backend.first.first);
    MS_EXCEPTION_IF_NULL(front_to_backend.second.first);
    if (HasAbstractMonad(front_to_backend.second.first) || HasAbstractMonad(front_to_backend.first.first) ||
        front_to_backend.second.first->isa<Parameter>() ||
        common::AnfAlgo::CheckPrimitiveType(front_to_backend.first.first, prim::kPrimPartial) ||
        front_to_backend.first.first->isa<ValueNode>()) {
      if (HasAbstractMonad(front_to_backend.first.first) || HasAbstractMonad(front_to_backend.second.first)) {
        MS_LOG(DEBUG) << "Kernel graph:" << graph->ToString() << " add monad output node:"
                      << (front_to_backend.first.first != nullptr ? front_to_backend.first.first->DebugString()
                                                                  : "null")
                      << " index:" << front_to_backend.first.second;
        (void)monad_outputs->emplace(front_to_backend.first);
      }
      continue;
    }

    // Skip the function input.
    const auto &abstract = front_to_backend.first.first->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, front_to_backend.first.second);
    MS_EXCEPTION_IF_NULL(real_abstract);
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      continue;
    }

    MS_LOG(DEBUG) << "Kernel graph:" << graph->ToString()
                  << " add front output node:" << front_to_backend.first.first->DebugString()
                  << " full name:" << front_to_backend.first.first->fullname_with_scope()
                  << " node ptr:" << front_to_backend.first.first << " index:" << front_to_backend.first.second
                  << " backend node:" << front_to_backend.second.first->DebugString()
                  << " full name:" << front_to_backend.second.first->fullname_with_scope()
                  << " node ptr:" << front_to_backend.second.first << " index:" << front_to_backend.second.second;
    (*outputs)[front_to_backend.first] = {front_to_backend.second};
  }
}

void ControlNodeParser::ParseKernelGraphGroup() {
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

        // Collect kernel graphs in group.
        (void)kernel_graph_group_info->graphs_.emplace(kernel_graph);

        // Collect inputs in group.
        CollectEffectiveInputByGraph(kernel_graph, kernel_graph_group_info.get());

        // Collect outputs in group.
        CollectEffectiveOutputByGraph(kernel_graph, &(kernel_graph_group_info->front_output_nodes_),
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
    const auto &input_node = input.first;
    size_t tmp_level = ParseControlNodeLevel(input_node, checked_nodes);
    level = (tmp_level > level ? tmp_level : level);
  }
  return level;
}

namespace {
AnfNodePtr GetRealOutputNode(const KernelWithIndex &front_pair, const KernelWithIndex &backend_pair) {
  if (front_pair.first == nullptr || backend_pair.first == nullptr) {
    return nullptr;
  }
  if (common::AnfAlgo::CheckPrimitiveType(backend_pair.first, prim::kPrimLoad) &&
      common::AnfAlgo::CheckPrimitiveType(front_pair.first, prim::kPrimLoad)) {
    const auto &backend_cnode = backend_pair.first->cast<CNodePtr>();
    const auto &front_cnode = front_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(backend_cnode);
    MS_EXCEPTION_IF_NULL(front_cnode);
    if (backend_cnode->inputs().size() > 1 && backend_cnode->input(1) != nullptr &&
        backend_cnode->input(1)->isa<CNode>() && front_cnode->inputs().size() > 1 && front_cnode->input(1) != nullptr &&
        front_cnode->input(1)->isa<CNode>()) {
      return front_cnode->input(1);
    }
  }
  return nullptr;
}
}  // namespace

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
      const auto &input_node = front_input_node.first;
      auto iter = node_to_level_.find(input_node);
      if (iter != node_to_level_.end() && level < iter->second) {
        level = iter->second;
      }
    }
    for (const auto &front_output_node : kernel_graph_group_info->front_output_nodes_) {
      MS_EXCEPTION_IF_NULL(front_output_node.second.first);
      if (front_output_node.second.first->isa<Parameter>()) {
        continue;
      }
      const auto &output_node = front_output_node.first.first;
      MS_EXCEPTION_IF_NULL(output_node);
      MS_LOG(DEBUG) << "Add level:" << level << " for node:" << output_node->DebugString();
      node_to_level_[output_node] = level;
      const auto &real_output_node = GetRealOutputNode(front_output_node.first, front_output_node.second);
      if (real_output_node != nullptr && node_to_level_.find(real_output_node) == node_to_level_.end()) {
        node_to_level_[real_output_node] = level;
      }
    }
  }

  // Parse the levels of kernel graph groups.
  for (const auto &kernel_graph_group_info : kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(kernel_graph_group_info);
    size_t max_level = 0;
    for (const auto &front_input_node : kernel_graph_group_info->front_input_nodes_) {
      const auto &input_node = front_input_node.first;
      MS_EXCEPTION_IF_NULL(input_node);
      auto iter = node_to_level_.find(input_node);
      if (iter == node_to_level_.end()) {
        PrintGraphGroupInfo(kernel_graph_group_infos_);
        MS_LOG_WITH_NODE(EXCEPTION, input_node) << "Failed to get input node:" << input_node->DebugString()
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
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Failed to find input:" << input_node->DebugString()
                                        << " for node:" << node->DebugString() << " in graph output map.";
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

void ControlNodeParser::CreateDeviceTensorForRootGraphParameter() {
  for (const auto &parameter : root_graph_parameters_) {
    MS_EXCEPTION_IF_NULL(parameter);
    const auto &abstract = parameter->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    size_t output_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
    for (size_t i = 0; i < output_num; ++i) {
      KernelWithIndex parameter_with_index(parameter, i);
      if (front_to_backend_parameters_.find(parameter_with_index) == front_to_backend_parameters_.end()) {
        MS_LOG(DEBUG) << "Create device tensor for root graph parameter:" << parameter->DebugString();
        CreateDeviceTensorForFrontNode(parameter_with_index);
        (void)front_to_backend_parameters_[parameter_with_index].emplace(parameter_with_index);
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

KernelWithIndex ControlNodeParser::FetchBackendOutputByKernelGraph(const KernelGraphPtr &graph,
                                                                   const KernelWithIndex &front_node_with_index) {
  MS_EXCEPTION_IF_NULL(graph);
  auto group_info_iter = kernel_graphs_to_group_info_.find(graph);
  if (group_info_iter == kernel_graphs_to_group_info_.end()) {
    MS_LOG(WARNING) << "Failed to get kernel graph group info for graph:" << graph->ToString();
    return {nullptr, 0};
  }
  MS_EXCEPTION_IF_NULL(group_info_iter->second);
  const auto &output_iter = group_info_iter->second->front_output_nodes_.find(front_node_with_index);
  if (output_iter != group_info_iter->second->front_output_nodes_.end()) {
    return output_iter->second;
  }
  const auto &backend_iter = std::find_if(
    group_info_iter->second->front_output_nodes_.begin(), group_info_iter->second->front_output_nodes_.end(),
    [front_node_with_index](const auto &pair) {
      return front_node_with_index == common::AnfAlgo::VisitKernelWithReturnType(pair.first.first, pair.first.second);
    });
  if (backend_iter == group_info_iter->second->front_output_nodes_.end()) {
    return {nullptr, 0};
  }
  return common::AnfAlgo::VisitKernelWithReturnType(backend_iter->second.first, backend_iter->second.second);
}

void ControlNodeParser::PrintParseInfo() {
  for (const auto &group : kernel_graph_group_infos_) {
    MS_EXCEPTION_IF_NULL(group);
    for (const auto &input_pair : group->front_input_nodes_) {
      if (input_pair.first != nullptr) {
        MS_LOG(WARNING) << "Kernel graph group:" << group->group_name_
                        << " input node:" << input_pair.first->fullname_with_scope()
                        << " debug string:" << input_pair.first->DebugString(AnfNode::DebugStringLevel::kLevel2)
                        << " index:" << input_pair.second;
      }
    }
    for (const auto &output_pair : group->front_output_nodes_) {
      if (output_pair.first.first != nullptr && output_pair.second.first != nullptr) {
        MS_LOG(WARNING) << "Kernel graph group:" << group->group_name_
                        << " output node:" << output_pair.first.first->fullname_with_scope()
                        << " debug string:" << output_pair.first.first->DebugString(AnfNode::DebugStringLevel::kLevel2)
                        << " index:" << output_pair.first.second
                        << " backend node:" << output_pair.second.first->fullname_with_scope()
                        << " debug string:" << output_pair.second.first->DebugString(AnfNode::DebugStringLevel::kLevel2)
                        << " index:" << output_pair.second.second;
      }
    }
  }
  for (const auto &f_to_b : front_to_backend_kernels_) {
    if (f_to_b.first.first != nullptr && f_to_b.second.first != nullptr) {
      MS_LOG(WARNING) << "Front to backend map front node:" << f_to_b.first.first->fullname_with_scope()
                      << " debug string:" << f_to_b.first.first->DebugString(AnfNode::DebugStringLevel::kLevel2)
                      << " index:" << f_to_b.first.second
                      << " backend node:" << f_to_b.second.first->fullname_with_scope()
                      << " debug string:" << f_to_b.second.first->DebugString(AnfNode::DebugStringLevel::kLevel2)
                      << " index:" << f_to_b.second.second;
    }
  }
  for (const auto &pair : front_node_to_kernel_graph_) {
    if (pair.first != nullptr && pair.second != nullptr) {
      MS_LOG(WARNING) << "Front node:" << pair.first->fullname_with_scope()
                      << " debug string:" << pair.first->DebugString(AnfNode::DebugStringLevel::kLevel2)
                      << " to kernel graph:" << pair.second->ToString();
    }
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
