/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/device/device_address_utils.h"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include "mindspore/core/ops/sequence_ops.h"
#include "ir/tensor.h"
#include "include/backend/device_address.h"
#include "include/backend/kernel_info.h"
#include "runtime/device/hash_table.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
using device::UserDataPtr;
using tensor::TensorPtr;
namespace runtime {
// Whether device address of anf node is valid and device address type
// is consistent with device type, for example, device address type
// DeviceType::kGPU should be used on GPU device
bool NodeDeviceAddressExist(const DeviceContext *device_context, const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  if (AnfAlgo::OutputAddrExist(node, index)) {
    const auto &address = AnfAlgo::GetOutputAddr(node, index, false);
    MS_EXCEPTION_IF_NULL(address);
    return address->GetDeviceType() == device_context->GetDeviceType();
  }
  return false;
}

void DeviceAddressUtils::CreateDeviceAddressByMapTensorNode(const DeviceContext *device_context, const AnfNodePtr &node,
                                                            size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &abstract_base = common::AnfAlgo::GetNodeAbstractByIndex(node, index);
  if (!abstract_base->isa<abstract::AbstractMapTensor>()) {
    MS_LOG(EXCEPTION) << "Parameter:" << node->DebugString() << " is not a map tensor type.";
  }

  const auto &abstract = abstract_base->cast<abstract::AbstractMapTensorPtr>();
  MS_EXCEPTION_IF_NULL(abstract);

  // Parse attrs for user data by abstract.
  const auto &value_shape = abstract->value_shape();
  MS_EXCEPTION_IF_NULL(value_shape);
  const auto &shape_vector = value_shape->shape();
  const auto &map_tensor_type = abstract->map_tensor_type();
  MS_EXCEPTION_IF_NULL(map_tensor_type);
  MS_EXCEPTION_IF_NULL(map_tensor_type->key_dtype());
  MS_EXCEPTION_IF_NULL(map_tensor_type->value_dtype());

  auto user_data = std::make_shared<UserData>();
  user_data->set(kUserDataType, std::make_shared<UserDataType>(UserDataType::kUserTypeHashTable));
  user_data->set(kHashTableKeyType, std::make_shared<TypeId>(map_tensor_type->key_dtype()->type_id()));
  user_data->set(kHashTableValueType, std::make_shared<TypeId>(map_tensor_type->value_dtype()->type_id()));
  user_data->set(kHashTableShapeVector, std::make_shared<ShapeVector>(shape_vector));
  user_data->set(kHashTableDefaultValue, abstract->default_value());
  user_data->set(kHashTablePermitFilter, abstract->permit_filter_value());
  user_data->set(kHashTableEvictFilter, abstract->evict_filter_value());
  // Create device for map tensor node and the ptr size is 1 byte.
  auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, 1, kOpFormat_DEFAULT, TypeId::kNumberTypeInt8, ShapeVector(), user_data);
  MS_LOG(DEBUG) << "Create device tensor:" << device_address << " type:" << device_address->type_id();
  AnfAlgo::SetOutputAddr(device_address, index, node.get());
}

void DeviceAddressUtils::CreateParameterDeviceAddress(const DeviceContext *device_context,
                                                      const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_inputs = graph->inputs();
  const std::vector<bool> &graph_valid_input = graph->valid_inputs();
  (void)graph_inputs.insert(graph_inputs.end(), graph->child_graph_result().begin(), graph->child_graph_result().end());

  // Anf nodes which need create device address.
  std::vector<AnfNodePtr> nodes_list;
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    AnfNodePtr item = graph_inputs[i];
    MS_EXCEPTION_IF_NULL(item);
    if (i < graph_valid_input.size() && !graph_valid_input[i]) {
      continue;
    }

    const auto &real_device_context = device::FetchRealDeviceContext(item, device_context);
    MS_EXCEPTION_IF_NULL(real_device_context);
    if (common::AnfAlgo::CheckPrimitiveType(item, prim::kPrimMakeTuple)) {
      std::vector<AnfNodePtr> outs = common::AnfAlgo::GetAllOutput(item);
      for (const auto &out : outs) {
        MS_EXCEPTION_IF_NULL(out);
        if (!out->isa<Parameter>() || NodeDeviceAddressExist(real_device_context, out, 0)) {
          continue;
        }
        nodes_list.push_back(out);
      }
    }
    if (!item->isa<Parameter>() || NodeDeviceAddressExist(real_device_context, item, 0)) {
      continue;
    }
    nodes_list.push_back(item);
  }

  // Create device address for anf node in nodes_list
  for (const auto &item : nodes_list) {
    MS_EXCEPTION_IF_NULL(item);
    const auto &real_device_context = device::FetchRealDeviceContext(item, device_context);
    MS_EXCEPTION_IF_NULL(real_device_context);
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      const auto &abstract = common::AnfAlgo::GetNodeAbstractByIndex(item, index);
      if (abstract != nullptr && abstract->isa<abstract::AbstractMapTensor>()) {
        CreateDeviceAddressByMapTensorNode(real_device_context, item, index);
        continue;
      }

      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      if (output_type_id == kTypeUnknown) {
        output_type_id = common::AnfAlgo::GetOutputInferDataType(item, index);
      }

      size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
      auto device_address = real_device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id,
        trans::GetRuntimePaddingShape(item, index));
      MS_EXCEPTION_IF_NULL(device_address);
      // Set the flag of no user parameter.
      if (item->isa<Parameter>()) {
        auto input_param = item->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(input_param);
        // Unused address will not alloc memory, which is easy to cause problems for weight node, so skip weight node.
        if (!common::AnfAlgo::IsParameterWeight(input_param) &&
            !input_param->IsUsedByRealKernelInGraph(graph->graph_id())) {
          MS_LOG(INFO) << "Node:" << item->fullname_with_scope() << " debug name:" << item->DebugString()
                       << " is not used in the graph " << graph->graph_id();
          device_address->UpdateFlag(device::kDeviceAddressFlagNotUsed);
        }
      }
      device_address->SetNodeIndex(item, index);
      device_address->set_from_persistent_mem(item->isa<Parameter>());
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(item)
                    << " addr:" << device_address << " type:" << device_address->type_id();
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
}

void DeviceAddressUtils::CreateDeviceAddressForTensorValue(const DeviceContext *device_context,
                                                           const ValuePtr &node_value, size_t output_idx,
                                                           const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::vector<TensorPtr> tensors;
  if (node_value->isa<ValueSequence>() && node_value->cast<ValueSequencePtr>()->size() != 0 &&
      common::AnfAlgo::IsDynamicSequence(value_node)) {
    (void)tensors.emplace_back(AnfAlgo::SequenceToTensor(node_value));
  } else {
    TensorValueToTensor(node_value, &tensors);
  }
  for (const auto &tensor : tensors) {
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (output_address != nullptr) {
      if (output_address->GetDeviceType() == device_context->GetDeviceType()) {
        // We need to set tensor->device_address to ValueNode even if the tensor is a forward_output tensor
        // in PyNative Bprop graph. ValueNode device_address is necessary for GraphSchedule::Transform.
        AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), output_idx++,
                               value_node.get());
        continue;
      }
      tensor->data_sync();
    }

    size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(value_node, output_idx);
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
    if (output_type_id == kTypeUnknown) {
      output_type_id = common::AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
    std::string output_format = AnfAlgo::GetOutputFormat(value_node, output_idx);

    device::DeviceAddressPtr address = device_context->device_res_manager_->CreateDeviceAddress(
      nullptr, tensor_size, output_format, output_type_id, tensor->shape());
    MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(value_node) << " addr:" << address
                  << " size:" << tensor_size << " format:" << output_format << " type:" << output_type_id
                  << " shape:" << tensor->shape();
    MS_EXCEPTION_IF_NULL(address);
    address->set_from_persistent_mem(true);
    AnfAlgo::SetOutputAddr(address, output_idx++, value_node.get());
  }
}

void DeviceAddressUtils::CreateValueNodeDeviceAddress(const DeviceContext *device_context,
                                                      const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  for (const ValueNodePtr &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (NodeDeviceAddressExist(device_context, value_node, 0)) {
      continue;
    }

    const auto &abstract = value_node->abstract();
    if (abstract != nullptr && abstract->isa<abstract::AbstractMapTensor>()) {
      CreateDeviceAddressByMapTensorNode(device_context, value_node, 0);
      continue;
    }

    const auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueTuple>()) {
      CreateDeviceAddressForTensorValue(device_context, node_value, 0, value_node);
      continue;
    }

    device::DeviceAddressPtr address = nullptr;
    if (node_value->isa<StringImm>()) {
      auto value = GetValue<std::string>(node_value);
      // Allocate one more byte to '/0'
      size_t tensor_size = value.size() + 1;
      address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT,
                                                                         kObjectTypeString, ShapeVector());
    } else if (node_value->isa<Scalar>()) {
      auto scalar_value = node_value->cast<ScalarPtr>();
      MS_EXCEPTION_IF_NULL(scalar_value);
      TypePtr data_type = scalar_value->type();
      MS_EXCEPTION_IF_NULL(data_type);
      TypeId type_id = data_type->type_id();
      address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, GetTypeByte(TypeIdToType(type_id)),
                                                                         kOpFormat_DEFAULT, type_id, ShapeVector());
    }
    if (address != nullptr) {
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(value_node)
                    << " addr:" << address;
      address->set_from_persistent_mem(true);
      AnfAlgo::SetOutputAddr(address, 0, value_node.get());
    } else {
      MS_LOG(INFO) << "No device address for value node:" << value_node->fullname_with_scope()
                   << ", debug name:" << common::AnfAlgo::GetNodeDebugString(value_node);
    }
  }
}

void DeviceAddressUtils::CreateKernelOutputDeviceAddress(const DeviceContext *device_context,
                                                         const KernelGraphPtr &graph, bool is_gradient_out) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);

  if (graph->memory_managed_by_ge()) {
    return;
  }

  bool is_pynative_bprop_graph = graph->has_flag(kFlagIsPynativeBpropGraph);
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output());

  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsBpropCutOpExecInBackend(kernel)) {
      continue;
    }

    bool is_from_persistent_mem =
      (is_gradient_out || (is_pynative_bprop_graph && (find(outputs.begin(), outputs.end(), kernel) != outputs.end())));

    auto output_size = AnfAlgo::GetOutputAddressNum(kernel);
    for (size_t i = 0; i < output_size; ++i) {
      if (AnfAlgo::OutputAddrExist(kernel, i)) {
        continue;
      }

      const auto &real_device_context = device::FetchRealDeviceContext(kernel, device_context);
      MS_EXCEPTION_IF_NULL(real_device_context);
      const auto &abstract = common::AnfAlgo::GetNodeAbstractByIndex(kernel, i);
      if (abstract != nullptr && abstract->isa<abstract::AbstractMapTensor>()) {
        CreateDeviceAddressByMapTensorNode(real_device_context, kernel, i);
        continue;
      }
      auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      auto address_size = AnfAlgo::GetOutputTensorMemSize(kernel, i);
      UserDataPtr user_data = nullptr;
      auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      if (kernel_info->kernel_mod() != nullptr && kernel_info->kernel_mod()->need_user_data()) {
        user_data = std::make_shared<UserData>();
      }
      auto device_address = real_device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, address_size, output_format, output_type, trans::GetRuntimePaddingShape(kernel, i), user_data);
      if (user_data != nullptr) {
        device_address->SetNodeIndex(kernel, i);
      }
      if (is_from_persistent_mem) {
        device_address->set_from_persistent_mem(true);
      }
      if (find(outputs.begin(), outputs.end(), kernel) != outputs.end()) {
        device_address->SetNodeIndex(kernel, i);
      }
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(kernel)
                    << " addr:" << device_address << " type:" << device_address->type_id();
      AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
    }
  }
}

void DeviceAddressUtils::CreateGraphOutputDeviceAddress(const DeviceContext *device_context,
                                                        const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  auto output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output_with_index : output_with_indexs) {
    const auto &output = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (common::AnfAlgo::IsBpropCutOpExecInBackend(output) || HasAbstractMonad(output)) {
      continue;
    }
    auto output_size = AnfAlgo::GetOutputAddressNum(output);
    for (size_t i = 0; i < output_size; ++i) {
      if (AnfAlgo::OutputAddrExist(output, i)) {
        continue;
      }

      const auto &real_device_context = device::FetchRealDeviceContext(output, device_context);
      MS_EXCEPTION_IF_NULL(real_device_context);
      auto output_format = AnfAlgo::GetOutputFormat(output, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(output, i);
      auto address_size = AnfAlgo::GetOutputTensorMemSize(output, i);
      auto device_address = real_device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, address_size, output_format, output_type, trans::GetRuntimePaddingShape(output, i));
      MS_LOG(DEBUG) << "Create addr for node:" << output->DebugString() << " addr:" << device_address
                    << " type:" << device_address->type_id();
      AnfAlgo::SetOutputAddr(device_address, i, output.get());
    }
  }
}

device::DeviceAddressPtr GetInputAddressForRef(const AnfNodePtr &node, const OpCompilerInfoPtr &op_compiler_info) {
  auto graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  for (size_t index = 0; index < graph->inputs().size(); ++index) {
    if (node == graph->inputs()[index]) {
      return op_compiler_info->inputs_[index];
    }
  }
  return nullptr;
}

device::DeviceAddressPtr GetOutputAddressForRef(const AnfNodePtr &node, const OpCompilerInfoPtr &op_compiler_info,
                                                size_t output_index) {
  auto execute_kernel_list = op_compiler_info->execute_kernel_list_;
  auto execute_kernel_list_size = execute_kernel_list.size();
  for (size_t i = 0; i < execute_kernel_list_size; ++i) {
    if (node == execute_kernel_list[i].kernel_) {
      return execute_kernel_list[i].outputs_device_address_[output_index];
    }
  }
  MS_LOG(EXCEPTION) << "Can't find the node in ExecuteKernelList.";
}

void UpdateOutputAddressForRef(const OpCompilerInfoPtr &op_compiler_info,
                               vector<device::DeviceAddressPtr> *device_address_list) {
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  const auto &ref_node_map = graph->GetRefMap();
  for (const auto &iter : ref_node_map) {
    auto &output_pair = iter.first;
    auto &input_pair = iter.second;
    auto &ref_node = output_pair.first;
    auto output_index = output_pair.second;
    auto input_address = GetInputAddressForRef(input_pair.first, op_compiler_info);
    if (input_address == nullptr) {
      continue;
    }
    auto output_address = GetOutputAddressForRef(ref_node, op_compiler_info, output_index);
    MS_EXCEPTION_IF_NULL(output_address);
    for (size_t index = 0; index < op_compiler_info->outputs_.size(); ++index) {
      if (output_address == op_compiler_info->outputs_[index]) {
        (*device_address_list)[index] = input_address;
      }
    }
  }
}

size_t GetTensorDeviceSize(const DeviceContext *device_context, const AnfNodePtr &node, const ShapeVector &shape,
                           std::string format, TypeId dtype, size_t output_index) {
  MS_EXCEPTION_IF_NULL(device_context);
  auto device_shape = shape;
  if (device_context->GetDeviceType() == device::DeviceType::kAscend) {
    if (device_shape.empty() && format != kOpFormat_DEFAULT) {
      device_shape = trans::PaddingShape(device_shape, format, AnfAlgo::GetOutputReshapeType(node, output_index));
      device_shape = trans::TransShapeToDevice(device_shape, format, node, output_index, dtype);
    } else {
      if (trans::IsNeedPadding(format, device_shape)) {
        device_shape =
          trans::PaddingShape(device_shape, format, AnfAlgo::GetOutputReshapeType(node, output_index), node);
      }
      device_shape = trans::TransShapeToDevice(device_shape, format, node, output_index, dtype);
    }
  }
  size_t type_size = GetTypeByte(TypeIdToType(dtype));
  size_t tensor_size = type_size * SizeOf(device_shape);
  return tensor_size;
}

vector<device::DeviceAddressPtr> DeviceAddressUtils::CreateGraphOutputDeviceAddress(
  const OpCompilerInfoPtr &op_compiler_info, const abstract::AbstractBasePtr &out_abstract) {
  MS_LOG(DEBUG) << "CreateGraphOutputDeviceAddress";
  MS_EXCEPTION_IF_NULL(out_abstract);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  auto device_context = op_compiler_info->device_context_;
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &output_nodes = op_compiler_info->graph_output_nodes_;
  auto outputs_address = op_compiler_info->outputs_;
  if (outputs_address.size() != output_nodes.size()) {
    MS_LOG(EXCEPTION) << "outputs_address size:" << outputs_address.size()
                      << ",not equal to output_nodes size:" << output_nodes.size();
  }
  vector<device::DeviceAddressPtr> real_output_address_list(output_nodes.size(), nullptr);
  // ref
  UpdateOutputAddressForRef(op_compiler_info, &real_output_address_list);
  for (size_t i = 0; i < output_nodes.size(); ++i) {
    if (real_output_address_list[i] == nullptr) {
      auto output_node = output_nodes[i].first;
      auto index = output_nodes[i].second;
      auto cache_output_address = op_compiler_info->outputs_[i];

      auto real_abstract = out_abstract;
      if (out_abstract->isa<abstract::AbstractTuple>()) {
        auto abstract_tuple = out_abstract->cast<abstract::AbstractTuplePtr>();
        if (i >= abstract_tuple->elements().size()) {
          MS_LOG(EXCEPTION) << "abstract_tuple size is " << abstract_tuple->elements().size() << " ,but get index is"
                            << i;
        }
        real_abstract = abstract_tuple->elements()[i];
      }
      auto output_shape_ptr = real_abstract->BuildShape();
      MS_EXCEPTION_IF_NULL(output_shape_ptr);
      auto shape_vector = output_shape_ptr->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(shape_vector);
      auto shape = shape_vector->shape();
      auto output_type = cache_output_address->type_id();
      auto output_format = cache_output_address->format();
      auto address_size = GetTensorDeviceSize(device_context, output_node, shape, output_format, output_type, index);
      auto device_address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, address_size,
                                                                                     output_format, output_type, shape);
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(output_node)
                    << " addr:" << device_address;
      real_output_address_list[i] = device_address;
    }
  }
  return real_output_address_list;
}

void DeviceAddressUtils::CreateKernelWorkspaceDeviceAddress(const DeviceContext *device_context,
                                                            const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);

  if (graph->memory_managed_by_ge()) {
    return;
  }

  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsBpropCutOpExecInBackend(kernel)) {
      continue;
    }
    const auto &real_device_context = device::FetchRealDeviceContext(kernel, device_context);
    MS_EXCEPTION_IF_NULL(real_device_context);
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      if (AnfAlgo::WorkspaceAddrExist(kernel, i)) {
        break;
      }
      auto device_address = real_device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, workspace_sizes[i], "", kTypeUnknown, ShapeVector());
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(kernel)
                    << " addr:" << device_address;
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());
    }
  }
}

void DeviceAddressUtils::UpdateDeviceAddressForInplaceNode(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  if (graph->memory_managed_by_ge()) {
    return;
  }

  // Collect the inplace groups.
  std::map<uint32_t, std::vector<CNodePtr>> inplace_groups;
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    if (!common::AnfAlgo::IsInplaceNode(kernel, "inplace_algo")) {
      continue;
    }
    auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel);
    MS_EXCEPTION_IF_NULL(primitive);
    auto inplace_group_attr = primitive->GetAttr("inplace_group");
    MS_EXCEPTION_IF_NULL(inplace_group_attr);
    auto group_id = GetValue<uint32_t>(inplace_group_attr);
    (void)inplace_groups[group_id].emplace_back(kernel);
  }

  const size_t kMinInplaceGroupSize = 2;
  for (const auto &inplace_group : inplace_groups) {
    auto &group_nodes = inplace_group.second;
    if (group_nodes.size() < kMinInplaceGroupSize) {
      continue;
    }
    // Get the device address of the first node in the inplace group.
    auto node_primitive = common::AnfAlgo::GetCNodePrimitive(group_nodes[0]);
    MS_EXCEPTION_IF_NULL(node_primitive);
    auto output_index = GetValue<uint32_t>(node_primitive->GetAttr("inplace_output_index"));
    auto device_address = AnfAlgo::GetMutableOutputAddr(group_nodes[0], output_index, false);
    MS_EXCEPTION_IF_NULL(device_address);

    // Update the device address of other nodes using device address of the first node in the inplace group.
    for (size_t i = 1; i < group_nodes.size(); ++i) {
      auto &group_node = group_nodes[i];
      auto prim = common::AnfAlgo::GetCNodePrimitive(group_node);
      MS_EXCEPTION_IF_NULL(prim);
      auto index = GetValue<uint32_t>(prim->GetAttr("inplace_output_index"));
      AnfAlgo::SetOutputAddr(device_address, index, group_node.get());
      // Update the reference count of device address.
      device_address->IncreaseOriginalRefCount();
      device_address->ResetRefCount();
    }
  }
}

void DeviceAddressUtils::UpdateDeviceAddress(const session::AnfWithOutIndex &cur_pair,
                                             const session::AnfWithOutIndex &origin_pair) {
  MS_EXCEPTION_IF_NULL(cur_pair.first);
  MS_EXCEPTION_IF_NULL(origin_pair.first);
  MS_LOG(INFO) << "Ref node pair: origin kernel is " << origin_pair.first->fullname_with_scope() << ", index is "
               << origin_pair.second << "; cur kernel is " << cur_pair.first->fullname_with_scope() << ", index is "
               << cur_pair.second;
  // If the output of ref node is parameter, need add the monad attr(for example Transdata/Cast node to ref parameter).
  if (!common::AnfAlgo::HasMonadInput(cur_pair.first) && origin_pair.first->isa<Parameter>()) {
    MS_LOG(INFO) << cur_pair.first->fullname_with_scope() << "with index " << cur_pair.second
                 << " ref node to parameter " << origin_pair.first->fullname_with_scope() << " and add the monad attr.";
    common::AnfAlgo::SetNodeAttr(kAttrRefNodeMonadOutputIdx, MakeValue(cur_pair.second), cur_pair.first);
  }

  auto origin_node_output_addr = AnfAlgo::GetMutableOutputAddr(origin_pair.first, origin_pair.second, false);
  MS_EXCEPTION_IF_NULL(origin_node_output_addr);
  auto cur_node_output_addr = AnfAlgo::GetMutableOutputAddr(cur_pair.first, cur_pair.second, false);
  MS_EXCEPTION_IF_NULL(cur_node_output_addr);

  // Update the device address flag.
  origin_node_output_addr->UpdateFlag(device::kDeviceAddressFlagRefNode);

  if (origin_node_output_addr.get() != cur_node_output_addr.get()) {
    // Check the device target whether consistent.
    if (origin_node_output_addr->GetDeviceType() != cur_node_output_addr->GetDeviceType()) {
      std::string error_info =
        "Device target is not consistent: ref origin kernel is " + origin_pair.first->fullname_with_scope() +
        ", index is " + std::to_string(origin_pair.second) + ", device target is " +
        device::GetDeviceNameByType(origin_node_output_addr->GetDeviceType()) + "; cur kernel is " +
        cur_pair.first->fullname_with_scope() + ", index is " + std::to_string(cur_pair.second) +
        ", device target is " + device::GetDeviceNameByType(cur_node_output_addr->GetDeviceType());

      MS_LOG(ERROR) << error_info;
      if (AnfAlgo::IsKernelSelectBackoffOp(origin_pair.first)) {
        const auto &backoff_info = AnfAlgo::GetKernelSelectBackoffInfo(origin_pair.first);
        MS_EXCEPTION(backoff_info.second) << "#umsg#Kernel select failed:#umsg#" << backoff_info.second;
      } else if (AnfAlgo::IsKernelSelectBackoffOp(cur_pair.first)) {
        const auto &backoff_info = AnfAlgo::GetKernelSelectBackoffInfo(cur_pair.first);
        MS_EXCEPTION(backoff_info.second) << "#umsg#Kernel select failed:#umsg#" << backoff_info.second;
      } else {
        MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#" << error_info;
      }
    }
    MS_LOG(INFO) << "Update device address: ref origin kernel is " << origin_pair.first->fullname_with_scope()
                 << ", index is " << origin_pair.second << "; cur kernel is " << cur_pair.first->fullname_with_scope()
                 << ", index is " << cur_pair.second;
    AnfAlgo::SetOutputAddr(origin_node_output_addr, cur_pair.second, cur_pair.first.get());
    // Update the reference count of device address.
    cur_node_output_addr->DecreaseOriginalRefCount();
    cur_node_output_addr->ResetRefCount();
    origin_node_output_addr->IncreaseOriginalRefCount();
    origin_node_output_addr->ResetRefCount();
  } else {
    MS_LOG(DEBUG) << "No need update device address: ref origin kernel is " << origin_pair.first->fullname_with_scope()
                  << ", index is " << origin_pair.second << "; cur kernel is " << cur_pair.first->fullname_with_scope()
                  << ", index is " << cur_pair.second;
  }
}

void DeviceAddressUtils::UpdateDeviceAddressForRefNode(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  if (graph->memory_managed_by_ge()) {
    return;
  }

  AnfAlgo::UpdateGraphValidRefPair(graph);
  for (const auto &ref_pair : graph->GetRefMap()) {
    const auto &out_pair = ref_pair.first;
    const auto &origin_pair = ref_pair.second;
    const auto &recursive_origin_pair = graph->GetRefNodeRecursive(out_pair);
    UpdateDeviceAddress(out_pair, recursive_origin_pair);
    // Update ref map in kernel info which will be used in kernel actor on swap scenario.
    for (size_t input_index = 0; input_index < common::AnfAlgo::GetInputTensorNum(out_pair.first); ++input_index) {
      const auto &prev_node_output = common::AnfAlgo::GetPrevNodeOutput(out_pair.first, input_index, false);
      if (prev_node_output == origin_pair) {
        auto kernel_info = dynamic_cast<device::KernelInfo *>(out_pair.first->kernel_info());
        MS_EXCEPTION_IF_NULL(kernel_info);
        kernel_info->AddRefMap(out_pair.second, input_index);
        break;
      }
    }
  }
}

device::DeviceAddressPtr DeviceAddressUtils::CloneEmptyDeviceAddress(const device::DeviceAddressPtr &old_device_address,
                                                                     const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(old_device_address);
  MS_EXCEPTION_IF_NULL(device_context);
  auto new_device_address = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, old_device_address->GetSize(), old_device_address->format(), old_device_address->type_id(),
    old_device_address->host_shape());
  MS_LOG(DEBUG) << "Create device tensor:" << new_device_address << " type:" << new_device_address->type_id();
  MS_EXCEPTION_IF_NULL(new_device_address);
  new_device_address->set_original_ref_count(old_device_address->original_ref_count());
  new_device_address->ResetRefCount();
  auto node = old_device_address->GetNodeIndex();
  new_device_address->SetNodeIndex(node.first, node.second);
  new_device_address->set_padding_type(old_device_address->padding_type());
  return new_device_address;
}
}  // namespace runtime
}  // namespace mindspore
