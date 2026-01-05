
/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "include/backend/common/device_address_utils.h"

#include <algorithm>
#include <functional>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <numeric>
#include "ops/op_def.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "ir/tensor.h"
#include "ir/tensor_new.h"
#include "ir/dtype/tensor_type.h"
#include "ir/graph_utils.h"
#include "device_address/device_address.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/backend/common/kernel_graph/py_execute_utils.h"
#include "include/utils/anfalgo.h"
#include "include/utils/callback.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/tensor_py.h"
#include "include/backend/common/ms_device_shape_transfer.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "runtime/hardware_abstract/utils.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#ifdef ENABLE_DEBUGGER
#include "device_address/device_type.h"
#endif
#include "include/runtime/pipeline/pipeline.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/core/include/ir/tensor_new.h"
#include "utils/stream_guard.h"
#include "utils/log_adapter.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"

namespace mindspore {
using tensor::TensorPtr;
namespace runtime {
namespace {

KernelTensorPtr CreateKernelTensorForScalarAndString(const DeviceContext *device_context,
                                                     const ValueNodePtr &value_node) {
  KernelTensorPtr kernel_tensor = nullptr;
  const auto &node_value = value_node->value();
  MS_EXCEPTION_IF_NULL(node_value);
  if (node_value->isa<StringImm>()) {
    auto value = GetValue<std::string>(node_value);
    // Allocate one more byte to '/0'
    size_t tensor_size = value.size() + 1;
    if (device_context->device_context_key().device_type_ == device::DeviceType::kAscend) {
      // size of ge::StringHead which defined in Ascend/latest.aarch64-linux/include/types.h
      constexpr size_t GE_STRING_HEAD_SIZE = 16;
      // NOTE: on Ascend, string type need a head of type ge::StringHead
      tensor_size += GE_STRING_HEAD_SIZE;
    }
    kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {value_node, 0}, nullptr, tensor_size, kOpFormat_DEFAULT, kObjectTypeString, ShapeVector(),
      device::GetDeviceNameByType(device_context->device_context_key().device_type_),
      device_context->device_context_key().device_id_);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
  } else if (node_value->isa<Scalar>()) {
    auto scalar_value = node_value->cast<ScalarPtr>();
    MS_EXCEPTION_IF_NULL(scalar_value);
    TypePtr data_type = scalar_value->type();
    MS_EXCEPTION_IF_NULL(data_type);
    TypeId type_id = data_type->type_id();
    kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {value_node, 0}, nullptr, GetTypeByte(TypeIdToType(type_id)), kOpFormat_DEFAULT, type_id, ShapeVector(),
      device::GetDeviceNameByType(device_context->device_context_key().device_type_),
      device_context->device_context_key().device_id_);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
  } else if (node_value->isa<None>()) {
    kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {value_node, 0}, nullptr, 0, kOpFormat_DEFAULT, kTypeNone->type_id(), ShapeVector(),
      device::GetDeviceNameByType(device_context->device_context_key().device_type_),
      device_context->device_context_key().device_id_);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
  }
  AnfAlgo::SetOutputKernelTensor(kernel_tensor, 0, value_node.get());

  return kernel_tensor;
}

Format GetFormatByTensorShape(const DeviceContext *device_context, const ShapeVector &tensor_shape) {
  return Format::DEFAULT_FORMAT;
}

const DeviceContext *GetDeviceContextForOffloadedParameter(const DeviceContext *origin_device_context,
                                                           const AnfNodePtr &node) {
  if (origin_device_context == nullptr) {
    return origin_device_context;
  }
  auto device_str = AnfAlgo::GetParameterDeviceStr(node);
  if (device_str.empty()) {
    return origin_device_context;
  }
  if (device_str == kToCpu) {
    auto hete_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device::GetDeviceTypeByName(device_str), origin_device_context->device_context_key().device_id_});
    MS_EXCEPTION_IF_NULL(hete_device_context);
    MS_LOG(INFO) << "Use " << device_str << " DeviceContext for offloaded parameter: " << node->DebugString();
    return hete_device_context;
  } else if (device_str == kToRemote) {
    return origin_device_context;
  } else {
    MS_LOG(EXCEPTION) << "Device of parameter only support \"CPU\" but got " << device_str;
  }
}

bool IsRemoteParameterNode(const mindspore::AnfNodePtr &node) {
  static const bool skip = ((common::GetEnv("MS_DEV_HYPER_OFFLOAD") != "1"));
  if (skip) {
    return false;
  }
  const auto &device_str = AnfAlgo::GetParameterDeviceStr(node);
  return device_str == kToRemote;
}

bool IsRemoteParameterTensor(const TensorPtr &tensor) {
  if ((common::GetEnv("MS_DEV_HYPER_OFFLOAD") != "1")) {
    return false;
  }
  constexpr auto kParameterDeviceUserDataName = "parameter_device";
  const auto &user_data = tensor->user_data<tensor::TensorPyUserData>(kParameterDeviceUserDataName);
  if (user_data == nullptr) {
    return false;
  }
  py::gil_scoped_acquire acquire;
  if (!py::isinstance<py::str>(user_data->obj)) {
    return false;
  }
  const auto &device_str = py::cast<std::string>(user_data->obj);
  return device_str == kToRemote;
}
}  // namespace

bool DeviceAddressUtils::NodeDeviceAddressExist(const DeviceContext *device_context, const AnfNodePtr &node,
                                                size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  if (AnfAlgo::OutputAddrExist(node, index)) {
    const auto address = AnfAlgo::GetMutableOutputAddr(node, index, false);
    MS_EXCEPTION_IF_NULL(address);
    return address->GetDeviceType() == device_context->GetDeviceType();
  }
  return false;
}

void DeviceAddressUtils::CopyNoneTensorDataToDevice(const device::DeviceContext *device_context,
                                                    const KernelTensorPtr &kernel_tensor, const ShapeVector &shape) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  // Break copy data to device address if has the device_address has flag ignore.
  if (TEST_FLAG(kernel_tensor->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
    MS_LOG(DEBUG) << "Address " << device_address << " has flag ignore device address, so skip copy tensor to device";
    return;
  }

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kConstantValue,
                                                 device_address->GetSize(), device_address.get());
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  if ((device_address->GetPtr() == nullptr)) {
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    } else {
      static std::string name = "Alloc memory";
      kernel_tensor->IncreaseNewRefCount(name);
    }
  }

  // Copy data from host to device.
  auto data_size = kernel_tensor->size();
  if (data_size == 0) {
    MS_LOG(INFO) << "Constant size is zero.";
    return;
  }
  const void *node_value = kernel_tensor->GetValuePtr();
  MS_EXCEPTION_IF_NULL(node_value);
  if (kernel_tensor->dtype_id() == TypeId::kObjectTypeString && kernel_tensor->IsConstValue()) {
    auto value = GetValue<std::string>(kernel_tensor->GetValueTrack());
    size_t tensor_size = value.size();
    ShapeVector tensor_shape{SizeToLong(tensor_size)};
    auto string_tensor =
      tensor::from_buffer(TypeId::kObjectTypeString, tensor_shape, const_cast<void *>(node_value), tensor_size);
    const auto &host_device_address = (dynamic_cast<device::DeviceAddress *>(string_tensor->device_address().get()));
    MS_EXCEPTION_IF_NULL(host_device_address);
    host_device_address->SetSize(tensor_size + 1);
    MS_LOG(DEBUG) << "Sync string to device size:" << tensor_size
                  << " device address:" << host_device_address->ToString()
                  << " dst device address:" << device_address->ToString();
    if (!device_context->device_res_manager_->SyncAllStreams() ||
        !SyncCopy(kernel_tensor.get(), string_tensor.get(), kDefaultStreamIndex)) {
      MS_LOG(ERROR) << "Failed sync string to device size:" << tensor_size
                    << " device address:" << host_device_address->ToString()
                    << " dst device address:" << device_address->ToString();
    }
    return;
  }
  if (!device_context->device_res_manager_->SyncAllStreams() ||
      !device_context->device_res_manager_->Copy(device_address->GetMutablePtr(), node_value, data_size,
                                                 device::CopyType::kH2D, device_address->stream_id())) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed";
  }
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
    auto real_device_context = device::FetchRealDeviceContext(item, device_context);
    auto origin_device_context = real_device_context;
    real_device_context = GetDeviceContextForOffloadedParameter(real_device_context, item);
    bool is_remote = IsRemoteParameterNode(item);
    MS_EXCEPTION_IF_NULL(real_device_context);
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      if (output_type_id == kTypeUnknown) {
        output_type_id = common::AnfAlgo::GetOutputInferDataType(item, index);
      }

      size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
      const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
        {item, index}, nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id,
        AnfAlgo::GetRuntimePaddingShape(item, index),
        device::GetDeviceNameByType(real_device_context->device_context_key().device_type_),
        real_device_context->device_context_key().device_id_, nullptr, 0, is_remote);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      AnfAlgo::SetOutputKernelTensor(kernel_tensor, index, item.get());
      kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(item));
      auto device_address = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_address);
      MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString() << " for item:" << item->DebugString()
                    << " in graph:" << graph->ToString();
      // Set the flag of no user parameter.
      if (item->isa<Parameter>()) {
        auto input_param = item->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(input_param);
        // Unused address will not alloc memory, which is easy to cause problems for weight node, so skip weight node.
        if (!common::AnfAlgo::IsParameterWeight(input_param) &&
            !input_param->IsUsedByRealKernelInGraph(graph->graph_id())) {
          MS_LOG(INFO) << "Node:" << item->fullname_with_scope() << " debug name:" << item->DebugString()
                       << " is not used in the graph " << graph->graph_id();
          kernel_tensor->UpdateFlag(device::kDeviceAddressFlagNotUsed);
        }
      }
      if (origin_device_context != real_device_context) {
        if (device_address->GetDeviceType() == device::DeviceType::kCPU &&
            origin_device_context->device_res_manager_->pin_mem_allocator() != nullptr) {
          device_address->set_allocator(origin_device_context->device_res_manager_->pin_mem_allocator());
          MS_LOG(DEBUG) << "Use PinMemoryAllocator for offloaded parameter. Parameter: " << item->fullname_with_scope();
        }
      }
      device_address->SetNodeIndex(item, index);
      device_address->set_from_persistent_mem(item->isa<Parameter>());
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(item)
                    << " addr:" << device_address->ToString();
    }
  }
}

std::vector<KernelTensorPtr> DeviceAddressUtils::CreateKernelTensorForTensorValue(const DeviceContext *device_context,
                                                                                  const ValuePtr &node_value,
                                                                                  size_t output_idx,
                                                                                  const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  std::vector<KernelTensorPtr> kernel_tensor_list;
  if (node_value->isa<tensor::Tensor>()) {
    auto tensor = node_value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto output_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (output_address != nullptr) {
      if (output_address->GetDeviceType() == device_context->GetDeviceType()) {
        // We need to set tensor->device_address to ValueNode even if the tensor is a forward_output tensor
        // in PyNative Bprop graph. ValueNode device_address is necessary for GraphSchedule::Transform.
        AnfAlgo::SetOutputAddr(tensor->device_address(), output_idx, value_node);
        auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(value_node, output_idx, false);
        MS_EXCEPTION_IF_NULL(kernel_tensor);
        kernel_tensor->set_format(tensor->format());
        kernel_tensor->SetShapeVector(tensor->shape());
        (void)kernel_tensor_list.emplace_back(kernel_tensor);
        return kernel_tensor_list;
      }
      auto cpu_tensor = tensor->cpu();
      value_node->set_value(cpu_tensor);
    }
  }

  size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(value_node, output_idx);
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
  if (output_type_id == kTypeUnknown) {
    output_type_id = common::AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    if (output_type_id == kTypeUnknown && value_node->value() != nullptr && value_node->value()->isa<ValueTuple>()) {
      const auto &value_tuple = value_node->value()->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(value_tuple);
      if (value_tuple->size() == 0) {
        MS_LOG(DEBUG) << "Set int64 type for empty value tuple node:" << value_node->DebugString();
        output_type_id = TypeId::kNumberTypeInt64;
      }
    }
  }
  std::string output_format = AnfAlgo::GetOutputFormat(value_node, output_idx);

  const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {value_node, output_idx}, nullptr, tensor_size, output_format, output_type_id, {},
    device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  kernel_tensor->SetShapeVector(kernel_tensor->GetShapeVector());
  kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
  device::DeviceAddressPtr address = kernel_tensor->device_address();
  MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(value_node) << " addr:" << address
                << " size:" << tensor_size << " format:" << output_format << " type:" << output_type_id
                << " shape:" << kernel_tensor->GetShapeVector();
  MS_EXCEPTION_IF_NULL(address);
  address->set_from_persistent_mem(true);
  AnfAlgo::SetOutputKernelTensor(kernel_tensor, output_idx, value_node.get());
  (void)kernel_tensor_list.emplace_back(kernel_tensor);
  return kernel_tensor_list;
}

mindspore::HashSet<mindspore::AnfNodePtr> FetchValueNodesNeedDevicePtr(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  mindspore::HashSet<mindspore::AnfNodePtr> nodes;
  auto topo_nodes = TopoSort(graph->get_return());
  for (auto const &n : topo_nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto node = n->cast<CNodePtr>();
    auto op_name = common::AnfAlgo::GetCNodeName(node);
    auto input_num = common::AnfAlgo::GetInputTensorNum(node);
    mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
    if (op_def == nullptr) {
      MS_LOG(DEBUG) << op_name << " is not found in OpDef.";
      for (size_t i = 0; i < input_num; i++) {
        auto input = common::AnfAlgo::GetInputNode(node, i);
        (void)nodes.insert(input);
      }
      continue;
    }
    auto args = op_def->args_;
    if (input_num != args.size()) {
      int input_with_init_args = std::count_if(args.begin(), args.end(), [](auto arg) { return arg.as_init_arg_; });
      size_t total = input_num - IntToSize(input_with_init_args);
      for (size_t i = 0; i < total; i++) {
        (void)nodes.insert(common::AnfAlgo::GetInputNode(node, i));
      }
      MS_LOG(DEBUG) << "Node " << op_name << ", has " << input_num << " inputs, but has " << args.size()
                    << " inputs in op_def, it means allsame input, input with init args number: "
                    << input_with_init_args;
      continue;
    }
    for (size_t i = 0; i < input_num; i++) {
      if (args[i].as_init_arg_ == 0) {
        auto input = common::AnfAlgo::GetInputNode(node, i);
        (void)nodes.insert(input);
      }
    }
  }
  return nodes;
}

device::DeviceAddressPtr CreateDeviceAddressForTypeValue(const DeviceContext *device_context,
                                                         const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {value_node, 0}, nullptr, 0, kOpFormat_DEFAULT, kMetaTypeTypeType, {},
    device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
  device::DeviceAddressPtr address = kernel_tensor->device_address();
  MS_LOG(DEBUG) << "Create addr for node:" << value_node->DebugString() << " addr:" << address;
  MS_EXCEPTION_IF_NULL(address);
  address->set_from_persistent_mem(true);
  AnfAlgo::SetOutputKernelTensor(kernel_tensor, 0, value_node.get());
  return address;
}

void DeviceAddressUtils::CreateValueNodeDeviceAddress(const DeviceContext *device_context,
                                                      const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
#ifdef ENABLE_DEBUGGER
  constexpr char kInputNeedDump[] = "InputNeedDump";
  static auto input_need_dump_callback = callback::CommonCallback::GetInstance().GetCallback<bool>(kInputNeedDump);
  bool enable_debug = false;
  if (input_need_dump_callback) {
    enable_debug = input_need_dump_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get InputNeedDump, data dump function may not work.";
  }
#endif
  // store node without init args, means need device addr
  auto value_nodes_without_init_args = FetchValueNodesNeedDevicePtr(graph);
  for (const ValueNodePtr &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (NodeDeviceAddressExist(device_context, value_node, 0)) {
      continue;
    }

    const auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueSequence>()) {
      auto kernel_tensor_list = CreateKernelTensorForTensorValue(device_context, node_value, 0, value_node);
      // Deal with tensor and tuple
      if (value_nodes_without_init_args.find(value_node) == value_nodes_without_init_args.end()) {
        for (const auto &kernel_tensor : kernel_tensor_list) {
#ifdef ENABLE_DEBUGGER
          if (enable_debug) {
            continue;
          }
#endif
          MS_EXCEPTION_IF_NULL(kernel_tensor);
          kernel_tensor->UpdateFlag(device::kDeviceAddressFlagIgnoreDevicePtr);
          MS_LOG(DEBUG) << "Find node " << value_node->DebugString() << " has init args";
        }
      }
      continue;
    } else if (node_value->isa<Type>()) {
      CreateDeviceAddressForTypeValue(device_context, value_node);
      continue;
    }

    KernelTensorPtr kernel_tensor = CreateKernelTensorForScalarAndString(device_context, value_node);
    // Deal with string and scalar; Address will be nullptr if the input is a type.
    if (kernel_tensor && (value_nodes_without_init_args.find(value_node) == value_nodes_without_init_args.end())) {
      kernel_tensor->UpdateFlag(device::kDeviceAddressFlagIgnoreDevicePtr);
      MS_LOG(DEBUG) << "Find node " << value_node->DebugString() << " has init args";
#ifdef ENABLE_DEBUGGER
      if (enable_debug) {
        kernel_tensor->ClearFlag(device::kDeviceAddressFlagIgnoreDevicePtr);
      }
#endif
    }
    if (kernel_tensor != nullptr) {
      auto address = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(address);
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(value_node)
                    << " addr:" << address;
      address->set_from_persistent_mem(true);
      AnfAlgo::SetOutputKernelTensor(kernel_tensor, 0, value_node.get());
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
  MS_LOG(DEBUG) << "Start create kernel output device address for graph:" << graph->ToString();
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output());

  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsBpropCutOpExecInBackend(kernel)) {
      continue;
    }

    bool is_from_persistent_mem = is_gradient_out;

    auto output_size = AnfAlgo::GetOutputAddressNum(kernel);
    const bool is_move_to = IsPrimitiveCNode(kernel, prim::kPrimMoveTo);
    const bool is_copy_to_host = IsPrimitiveCNode(kernel, prim::kPrimCopyToHost);
    std::string move_to;
    if (is_move_to) {
      move_to = common::AnfAlgo::GetMoveToDstStr(kernel);
    }
    for (size_t i = 0; i < output_size; ++i) {
      if (AnfAlgo::OutputAddrExist(kernel, i)) {
        continue;
      }

      auto real_device_context = device::FetchRealDeviceContext(kernel, device_context);
      auto origin_device_context = real_device_context;
      if (real_device_context != nullptr && is_move_to) {
        if (move_to == kToCpu) {
          real_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
            {device::GetDeviceTypeByName(move_to), real_device_context->device_context_key().device_id_});
          MS_LOG(INFO) << "Use " << move_to << " DeviceContext for MoveTo node: " << kernel->DebugString();
        } else if (move_to != kToNpu) {
          MS_LOG(EXCEPTION) << R"(Destination for MoveTo is supposed to be "CPU" or "Ascend", but got )" << move_to;
        }
      }
      if (real_device_context != nullptr && is_copy_to_host) {
        real_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
          {device::GetDeviceTypeByName(kToCpu), real_device_context->device_context_key().device_id_});
        MS_LOG(INFO) << "Use " << kToCpu << " DeviceContext for CopyToHost node: " << kernel->DebugString();
      }
      MS_EXCEPTION_IF_NULL(real_device_context);
      auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      auto address_size = AnfAlgo::GetOutputTensorMemSize(kernel, i);

      UserDataPtr user_data = nullptr;
      auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      if (kernel_info->kernel_mod() != nullptr && kernel_info->kernel_mod()->need_user_data()) {
        user_data = std::make_shared<UserData>();
        user_data->set(kSyncUserDataHandler,
                       std::make_shared<kernel::KernelTensor::SyncUserDataHandler>(pyexecute::UserDataToRawMemory));
        user_data->set(kGetValueByUserDataHandler,
                       std::make_shared<ValuePtr (*)(const UserDataPtr &)>(pyexecute::GetValueFromUserData));
        graph->set_has_kernel_need_user_data(true);
      }
      const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
        {kernel, i}, nullptr, address_size, output_format, output_type, AnfAlgo::GetRuntimePaddingShape(kernel, i),
        device::GetDeviceNameByType(real_device_context->device_context_key().device_type_),
        real_device_context->device_context_key().device_id_, user_data);
      auto stream_id = AnfAlgo::GetStreamId(kernel);
      kernel_tensor->set_stream_id(stream_id);
      AnfAlgo::SetAllocStreamId(kernel_tensor, stream_id, kernel);
      MS_LOG(DEBUG) << "Kernel tensor created without set stream id, but set after device address created.";
      auto device_address = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_address);
      if (origin_device_context != real_device_context) {
        if (device_address->GetDeviceType() == device::DeviceType::kCPU &&
            origin_device_context->device_res_manager_->pin_mem_allocator() != nullptr) {
          device_address->set_allocator(origin_device_context->device_res_manager_->pin_mem_allocator());
          MS_LOG(DEBUG) << "Use PinMemoryAllocator for MoveTo cpu output. Kernel: " << kernel->fullname_with_scope();
        }
      }
      SetSymmetricMemoryAllocator(kernel, device_address, real_device_context);

      device_address->SetNodeIndex(kernel, i);
      if (is_from_persistent_mem) {
        device_address->set_from_persistent_mem(true);
      }
      MS_LOG(DEBUG) << "Create addr for node:" << kernel->fullname_with_scope() << " index:" << i
                    << ", kernel tensor: " << kernel_tensor->ToString() << " addr size:" << address_size
                    << " real size:" << device_address->GetSize();
      device_address->set_stream_id(AnfAlgo::GetStreamId(kernel));
      AnfAlgo::SetOutputKernelTensor(kernel_tensor, i, kernel.get());
    }
  }
  MS_LOG(DEBUG) << "End create kernel output device address for graph:" << graph->ToString();
}

void DeviceAddressUtils::SetSymmetricMemoryAllocator(const CNodePtr &kernel,
                                                     const device::DeviceAddressPtr &device_address,
                                                     const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(device_context);

  if (kernel->fullname_with_scope().find(kCreateSymmetricMemoryOpName) != std::string::npos) {
    device_address->set_allocator(device_context->device_res_manager_->symmetric_memory_allocator());
    MS_LOG(DEBUG) << "Use SymmetricMemoryAllocator for CreateSymmetricMemory: " << kernel->fullname_with_scope();
  }
}

void DeviceAddressUtils::CreateGraphOutputDeviceAddress(const DeviceContext *device_context,
                                                        const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  auto output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output_with_index : output_with_indexs) {
    const auto &output =
      common::AnfAlgo::VisitKernelWithReturnType(output_with_index.first, output_with_index.second).first;
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
      MS_EXCEPTION_IF_NULL(real_device_context->device_res_manager_);
      auto output_format = AnfAlgo::GetOutputFormat(output, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(output, i);
      auto address_size = AnfAlgo::GetOutputTensorMemSize(output, i);
      const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
        {output, i}, nullptr, address_size, output_format, output_type, AnfAlgo::GetRuntimePaddingShape(output, i),
        device::GetDeviceNameByType(real_device_context->device_context_key().device_type_),
        real_device_context->device_context_key().device_id_);
      kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(output));
      MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString() << " for node:" << output->DebugString();
      AnfAlgo::SetOutputKernelTensor(kernel_tensor, i, output.get());
    }
  }
}

size_t DeviceAddressUtils::GetTensorDeviceSize(const DeviceContext *device_context, const AnfNodePtr &node,
                                               const ShapeVector &shape, const string &format, TypeId dtype,
                                               size_t output_index) {
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
      auto kernel_tensor =
        AnfAlgo::CreateKernelTensor(nullptr, workspace_sizes[i], Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
                                    device::GetDeviceNameByType(real_device_context->device_context_key().device_type_),
                                    real_device_context->device_context_key().device_id_);
      auto stream_id = AnfAlgo::GetStreamId(kernel);
      kernel_tensor->set_stream_id(stream_id);
      AnfAlgo::SetAllocStreamId(kernel_tensor, stream_id, kernel);
      auto device_address = kernel_tensor->device_address();
      MS_LOG(DEBUG) << "Create addr for node:" << kernel->fullname_with_scope()
                    << " kernel tensor:" << kernel_tensor->ToString();
      AnfAlgo::SetWorkspaceKernelTensor(kernel_tensor, i, kernel.get());
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

  constexpr size_t kMinInplaceGroupSize = 2;
  for (const auto &inplace_group : inplace_groups) {
    auto &group_nodes = inplace_group.second;
    if (group_nodes.size() < kMinInplaceGroupSize) {
      continue;
    }
    // Get the device address of the first node in the inplace group.
    auto node_primitive = common::AnfAlgo::GetCNodePrimitive(group_nodes[0]);
    MS_EXCEPTION_IF_NULL(node_primitive);
    auto output_index = GetValue<uint32_t>(node_primitive->GetAttr("inplace_output_index"));
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(group_nodes[0], output_index, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);

    // Update the device address of other nodes using device address of the first node in the inplace group.
    for (size_t i = 1; i < group_nodes.size(); ++i) {
      auto &group_node = group_nodes[i];
      auto prim = common::AnfAlgo::GetCNodePrimitive(group_node);
      MS_EXCEPTION_IF_NULL(prim);
      auto index = GetValue<uint32_t>(prim->GetAttr("inplace_output_index"));
      auto group_node_kernel_tensor = AnfAlgo::GetOutputKernelTensor(group_node, index, false);
      MS_EXCEPTION_IF_NULL(group_node_kernel_tensor);
      // Update the reference count of device address.
      group_node_kernel_tensor->set_pointer_ref_count(kernel_tensor.get());
    }
  }
}

void DeviceAddressUtils::UpdateDeviceAddressMonadFlag(const session::AnfWithOutIndex &cur_pair,
                                                      const session::AnfWithOutIndex &origin_pair) {
  MS_EXCEPTION_IF_NULL(cur_pair.first);
  MS_EXCEPTION_IF_NULL(origin_pair.first);
  // If the output of ref node is parameter, need add the monad attr(for example Transdata/Cast node to ref
  // parameter).
  if (!common::AnfAlgo::HasMonadInput(cur_pair.first) && origin_pair.first->isa<Parameter>()) {
    MS_LOG(INFO) << cur_pair.first->fullname_with_scope() << "with index " << cur_pair.second
                 << " ref node to parameter " << origin_pair.first->fullname_with_scope() << " and add the monad attr.";
    common::AnfAlgo::SetNodeAttr(kAttrRefNodeMonadOutputIdx, MakeValue(cur_pair.second), cur_pair.first);
  }
}

void DeviceAddressUtils::UpdateDeviceAddress(const session::AnfWithOutIndex &cur_pair,
                                             const session::AnfWithOutIndex &origin_pair) {
  MS_EXCEPTION_IF_NULL(cur_pair.first);
  MS_EXCEPTION_IF_NULL(origin_pair.first);
  MS_LOG(INFO) << "Ref node pair: origin kernel is " << origin_pair.first->fullname_with_scope() << ", index is "
               << origin_pair.second << "; cur kernel is " << cur_pair.first->fullname_with_scope() << ", index is "
               << cur_pair.second;
  // If the output of ref node is parameter, need add the monad attr(for example Transdata/Cast node to ref
  // parameter).
  UpdateDeviceAddressMonadFlag(cur_pair, origin_pair);
  auto origin_node_output_kt = AnfAlgo::GetOutputKernelTensor(origin_pair.first, origin_pair.second, false);
  MS_EXCEPTION_IF_NULL(origin_node_output_kt);
  auto origin_node_output_addr = origin_node_output_kt->device_address();
  MS_EXCEPTION_IF_NULL(origin_node_output_addr);
  auto cur_node_output_kt = AnfAlgo::GetOutputKernelTensor(cur_pair.first, cur_pair.second, false);
  MS_EXCEPTION_IF_NULL(cur_node_output_kt);
  auto cur_node_output_addr = cur_node_output_kt->device_address();
  MS_EXCEPTION_IF_NULL(cur_node_output_addr);
  auto origin_stream_id = origin_node_output_addr->stream_id();
  auto cur_stream_id = cur_node_output_addr->stream_id();
  if (origin_stream_id != cur_stream_id) {
    MS_LOG(DEBUG) << "Origin node output addr : " << origin_node_output_addr << " stream id : " << origin_stream_id
                  << " is not equal to cur node output addr stream id : " << cur_stream_id << ".";
  }

  // Update the device address flag.
  origin_node_output_kt->UpdateFlag(device::kDeviceAddressFlagRefNode);

  if (origin_node_output_addr->device_pointer() != cur_node_output_addr->device_pointer()) {
    // Check the device target whether consistent.
    if (origin_node_output_addr->GetDeviceType() != cur_node_output_addr->GetDeviceType()) {
      MS_LOG(INFO) << "Device target is not consistent: ref origin device address " << origin_node_output_addr
                   << "kernel is " << origin_pair.first->fullname_with_scope() << ", index is "
                   << std::to_string(origin_pair.second) << ", device target is "
                   << device::GetDeviceNameByType(origin_node_output_addr->GetDeviceType()) << "; cur device address "
                   << cur_node_output_addr << "kernel is " << cur_pair.first->fullname_with_scope() << ", index is "
                   << std::to_string(cur_pair.second) << ", device target is "
                   << device::GetDeviceNameByType(cur_node_output_addr->GetDeviceType());
      return;
    }
    MS_LOG(INFO) << "Update device address: ref origin device address:" << origin_node_output_addr << "kernel is "
                 << origin_pair.first->fullname_with_scope() << ", index is " << origin_pair.second
                 << "; cur device address " << cur_node_output_addr << " kernel is "
                 << cur_pair.first->fullname_with_scope() << ", index is " << cur_pair.second;
    cur_node_output_kt->set_pointer_ref_count(origin_node_output_kt.get());
    origin_node_output_kt->UpdateFlag(device::kDeviceAddressFlagRefNode);
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
  for (const auto &pair : graph->GetRefMap()) {
    MS_LOG(DEBUG) << "Ref node pair for node:" << pair.first.first->DebugString() << " index:" << pair.first.second
                  << " and node:" << pair.second.first->DebugString() << " index:" << pair.second.second;
  }
  for (const auto &ref_pair : graph->GetRefMap()) {
    const auto &out_pair = ref_pair.first;
    const auto &origin_pair = ref_pair.second;
    const auto &recursive_origin_pair = graph->GetRefNodeRecursive(out_pair);
    UpdateDeviceAddressMonadFlag(out_pair, recursive_origin_pair);
    //  Update ref map in kernel info which will be used in kernel actor on swap scenario.
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

void DeviceAddressUtils::UpdateDeviceAddressForRefNodeForSingleOp(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  if (graph->memory_managed_by_ge()) {
    return;
  }

  AnfAlgo::UpdateGraphValidRefPair(graph);
  for (const auto &pair : graph->GetRefMap()) {
    MS_LOG(DEBUG) << "Ref node pair for node:" << pair.first.first->DebugString() << " index:" << pair.first.second
                  << " and node:" << pair.second.first->DebugString() << " index:" << pair.second.second;
  }
  for (const auto &ref_pair : graph->GetRefMap()) {
    const auto &out_pair = ref_pair.first;
    const auto &origin_pair = ref_pair.second;
    const auto &recursive_origin_pair = graph->GetRefNodeRecursive(out_pair);
    UpdateDeviceAddress(out_pair, recursive_origin_pair);
    //  Update ref map in kernel info which will be used in kernel actor on swap scenario.
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

KernelTensorPtr DeviceAddressUtils::CloneEmptyKernelTensor(const KernelTensorPtr &old_kernel_tensor,
                                                           const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(old_kernel_tensor);
  MS_EXCEPTION_IF_NULL(device_context);

  auto device_address = old_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  auto new_device_address =
    std::make_shared<device::DeviceAddress>(device_address->device_pointer()->ptr(), device_address->size(),
                                            device_context->GetDeviceType(), device_address->stream_id());
  auto new_kernel_tensor = old_kernel_tensor->CloneKernelTensor();
  new_kernel_tensor->set_user_data(old_kernel_tensor->user_data());
  new_kernel_tensor->set_need_sync_user_data(old_kernel_tensor->need_sync_user_data());
  new_kernel_tensor->set_device_address(new_device_address);

  MS_EXCEPTION_IF_NULL(new_kernel_tensor);
  auto &old_device_address = old_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(old_device_address);

  new_kernel_tensor->set_device_ptr(nullptr);
  MS_LOG(DEBUG) << "Create kernel tensor: " << new_kernel_tensor->ToString() << " by:" << old_kernel_tensor->ToString();
  auto node = old_device_address->GetNodeIndex();
  new_device_address->SetNodeIndex(node.first, node.second);
  new_device_address->set_padding_type(old_device_address->padding_type());
  return new_kernel_tensor;
}

void CheckAutoH2D(const DeviceContext *device_context, const tensor::TensorPtr &tensor) {
  if (!EnableDispatchWithCheck()) {
    return;
  }
  if (tensor->source_type() != ops::OP_DTYPE::DT_BEGIN) {
    MS_LOG(DEBUG) << "Input tensor source_type is " << tensor->source_type();
  }
  auto addr = tensor->device_address();
  auto device_address = std::static_pointer_cast<device::DeviceAddress>(addr);
  if (device_address->GetDeviceType() != device_context->GetDeviceType()) {
    MS_LOG(EXCEPTION) << "[Dispatch]The tensor device address type is " << device_address->GetDeviceType()
                      << ". Need to call Tensor.to first";
  }
}

void DeviceAddressUtils::LazyCopy(const tensor::TensorPtr &tensor, size_t stream_id) {
  const auto &dst = tensor->device_address();
  const auto &src = tensor->implicit_copy_address();
  if (src == nullptr) {
    MS_LOG(DEBUG) << "No need to do implicit copy for " << tensor->ToString();
    return;
  }
  MS_EXCEPTION_IF_NULL(dst);
  MS_LOG(DEBUG) << "Lazy copy for dst " << dst->ToString() << " src " << src->ToString() << " on stream " << stream_id;
  DeviceAddressExtPtr src_ext =
    std::make_shared<DeviceAddressExt>(tensor->implicit_copy_format(), tensor->data_type(), tensor->shape());
  DeviceAddressExtPtr dst_ext =
    std::make_shared<DeviceAddressExt>(tensor->format(), tensor->data_type(), tensor->shape());
  if (src->GetDeviceType() != device::DeviceType::kCPU && dst->GetDeviceType() == device::DeviceType::kCPU) {
    if (!SyncCopy(dst, src, stream_id, src_ext, dst_ext)) {
      MS_LOG(EXCEPTION) << "Lazy Sync copy failed. dst " << dst->ToString() << " src " << src->ToString()
                        << " on stream " << stream_id;
    }
  } else {
    if (!AsyncCopy(dst, src, stream_id, true, src_ext, dst_ext)) {
      MS_LOG(EXCEPTION) << "Lazy Async copy failed. dst " << dst->ToString() << " src " << src->ToString()
                        << " on stream " << stream_id;
    }
  }
  tensor->set_implicit_copy_address(nullptr);
  MS_LOG(DEBUG) << "Copy success, and delete implicit address of tensor " << tensor->ToString();
}

void DeviceAddressUtils::CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                                                  const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(device_context);
  if (tensor == nullptr) {
    return;
  }

  auto addr = tensor->device_address();
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "The " << tensor->ToString() << " is uninitialized. "
                      << "Maybe the Tensor is create by initializer. "
                      << "You need to call Tensor.init_data before using this Tensor. "
                      << "For more detail with 'Tensor', Please refer to "
                      << "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html";
  }

  CheckAutoH2D(device_context, tensor);

  auto tensor_address = std::static_pointer_cast<device::DeviceAddress>(addr);
  if (tensor_address->GetDeviceType() == device_context->GetDeviceType()) {
    MS_LOG(DEBUG) << "Already have device address of tensor " << tensor->id();
    return;
  }

  // Not type_cast from python scalar or tuple.
  if (tensor->source_type() == ops::OP_DTYPE::DT_BEGIN) {
    runtime::Pipeline::Get().WaitForward();
  }

  MS_LOG(DEBUG) << "Input tensor device type is " << tensor_address->GetDeviceType()
                << " but current device context is " << device_context->GetDeviceType();

  auto tensor_size = LongToSize(tensor->DataNBytes());
  const auto &format = GetFormatByTensorShape(device_context, tensor->shape());
  auto device_address =
    std::make_shared<device::DeviceAddress>(nullptr, tensor_size, device_context->GetDeviceType(), stream_id);
  bool is_remote = IsRemoteParameterTensor(tensor);
  device_address->set_remote(is_remote);
  MS_EXCEPTION_IF_NULL(device_address);
  device_address->set_from_persistent_mem(tensor->is_parameter());

  // keep origin device_address and execute in another thread.
  tensor->set_implicit_copy_address(addr);
  tensor->set_implicit_copy_format(tensor->format());
  tensor->set_device_address(device_address);
  tensor->set_format(format);
  MS_LOG(DEBUG) << "Create input tensor device address " << device_address << " for " << index
                << "th input, Shape: " << tensor->shape() << ", Type: " << TypeIdToType(tensor->data_type())->ToString()
                << ", Size:" << tensor_size;
}

void DeviceAddressUtils::CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                                                  const ValueTuplePtr &value_tuple) {
  MS_EXCEPTION_IF_NULL(value_tuple);
  const auto &values = value_tuple->value();
  auto size = values.size();
  std::vector<tensor::TensorPtr> tensors;
  for (size_t i = 0; i < size; ++i) {
    const auto &value = values[i];
    if (value != nullptr && value->isa<tensor::Tensor>()) {
      tensors.push_back(GetValue<tensor::TensorPtr>(value));
    }
  }
  CreateInputTensorAddress(device_context, stream_id, index, tensors);
}

void DeviceAddressUtils::CreateInputTensorAddress(const DeviceContext *device_context, size_t stream_id, size_t index,
                                                  const std::optional<tensor::TensorPtr> &val) {
  if (!val.has_value()) {
    return;
  }
  CreateInputTensorAddress(device_context, stream_id, index, val.value());
}

KernelTensorPtr DeviceAddressUtils::CreateInputKernelTensor(const DeviceContext *device_context, size_t stream_id,
                                                            const abstract::AbstractBasePtr &abs, size_t index,
                                                            const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(tensor);
  BaseShapePtr shape;
  TypePtr type;
  if (abs != nullptr) {
    shape = abs->GetShape();
    type = abs->GetType();
  } else {
    shape = std::make_shared<abstract::Shape>(tensor->shape());
    type = tensor->Dtype();
  }

  auto addr = tensor->device_address();
  if (addr->GetDeviceType() == device_context->GetDeviceType()) {
    auto device_address = std::static_pointer_cast<device::DeviceAddress>(addr);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() != nullptr) {
      auto kernel_tensor = std::make_shared<kernel::KernelTensor>(shape, type, nullptr);
      kernel_tensor->set_device_address(device_address);
      kernel_tensor->SetShapeVector(tensor->shape());
      kernel_tensor->set_format(tensor->format());
      MS_LOG(DEBUG) << "Input tensor already have address " << device_address.get() << " and device Ptr "
                    << device_address->GetPtr() << ", kernel tensor info: " << kernel_tensor->ToString();
      return kernel_tensor;
    }
  }

  const auto &tensor_size = tensor->DataNBytes();
  const auto &format = GetFormatByTensorShape(device_context, tensor->shape());
  auto kernel_tensor = AnfAlgo::CreateKernelTensor(
    shape, type, nullptr, nullptr, tensor_size, kernel::GetFormatFromEnumToStr(format), tensor->data_type(),
    tensor->shape(), device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(stream_id);
  device::DeviceAddressPtr device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  device_address->set_from_persistent_mem(tensor->is_parameter());
  tensor->set_device_address(device_address);
  tensor->set_format(format);

  auto mem_type =
    tensor->is_parameter() ? memory::mem_pool::MemType::kWeight : memory::mem_pool::MemType::kConstantValue;
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", mem_type, device_address->GetSize(),
                                                 device_address.get());
  if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
    MS_LOG(EXCEPTION) << "Allocate memory failed";
  } else {
    static std::string name = "Alloc memory";
    kernel_tensor->IncreaseNewRefCount(name);
  }
  if (!AsyncCopy(kernel_tensor.get(), tensor.get(), device_address->stream_id())) {
    MS_LOG(EXCEPTION) << "Copy host data to device failed";
  }
  MS_LOG(DEBUG) << "Create input tensor device address " << device_address << " for " << index
                << "th input, Shape: " << shape->ToString()
                << ", Type: " << TypeIdToType(tensor->data_type())->ToString() << ", host shape: " << tensor->shape()
                << ", dev ptr " << device_address->GetPtr();
  return kernel_tensor;
}

KernelTensorPtr DeviceAddressUtils::CreateInputKernelTensor(const DeviceContext *device_context, size_t stream_id,
                                                            const abstract::AbstractBasePtr &abs, size_t index,
                                                            const ScalarPtr &scalar_value) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(scalar_value);
  const auto type = scalar_value->type();
  MS_EXCEPTION_IF_NULL(type);
  auto kernel_tensor = AnfAlgo::CreateKernelTensor(
    abstract::kNoShape, type, scalar_value, nullptr, GetTypeByte(TypeIdToType(type->type_id())), kOpFormat_DEFAULT,
    type->type_id(), ShapeVector(), device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(stream_id);
  auto device_address = kernel_tensor->device_address();
  device_address->set_from_persistent_mem(true);

  if (device_address->GetPtr() == nullptr) {
    CopyNoneTensorDataToDevice(device_context, kernel_tensor);
  }
  MS_LOG(DEBUG) << "Create input scalar device address " << device_address << " for " << index
                << "th input, Shape: " << abstract::kNoShape->ToString() << ", Type: " << type->ToString()
                << ", Value: " << (scalar_value ? scalar_value->ToString() : "nullptr") << ", dev ptr "
                << device_address->GetPtr() << ", kernel tensor: " << kernel_tensor->ToString();
  return kernel_tensor;
}

KernelTensorPtr DeviceAddressUtils::CreateInputKernelTensor(const DeviceContext *device_context, size_t stream_id,
                                                            const abstract::AbstractBasePtr &abs, size_t index,
                                                            const std::optional<tensor::TensorPtr> &val) {
  if (!val.has_value()) {
    return nullptr;
  }
  return CreateInputKernelTensor(device_context, stream_id, abs, index, val.value());
}

KernelTensorPtr DeviceAddressUtils::CreateInputKernelTensor(const DeviceContext *device_context, size_t stream_id,
                                                            const abstract::AbstractBasePtr &abs, size_t index,
                                                            const StringImmPtr &string_imm) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(string_imm);
  const auto &type = string_imm->type();
  MS_EXCEPTION_IF_NULL(type);
  const auto &tensor_value = GetValue<std::string>(string_imm);
  // Allocate one more byte to '/0'
  size_t size = tensor_value.size() + 1;
  auto kernel_tensor = AnfAlgo::CreateKernelTensor(
    abstract::kNoShape, type, string_imm, nullptr, size, kOpFormat_DEFAULT, kObjectTypeString, ShapeVector(),
    device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(stream_id);
  auto device_address = kernel_tensor->device_address();
  device_address->set_from_persistent_mem(true);

  if (device_address->GetPtr() == nullptr) {
    CopyNoneTensorDataToDevice(device_context, kernel_tensor);
  }
  MS_LOG(DEBUG) << "Create input string device address " << device_address << " for " << index
                << "th input, Shape: " << abstract::kNoShape->ToString() << ", Type: " << type->ToString()
                << ", Value: " << (string_imm ? string_imm->ToString() : "nullptr") << ", dev ptr "
                << device_address->GetPtr() << ", kernel tensor: " << kernel_tensor->ToString();
  return kernel_tensor;
}

KernelTensorPtr DeviceAddressUtils::CreateInputKernelTensor(const DeviceContext *device_context, size_t stream_id,
                                                            const abstract::AbstractBasePtr &abs, size_t index,
                                                            const TypePtr &type_ptr) {
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &type = type_ptr->type();
  MS_EXCEPTION_IF_NULL(type);
  auto kernel_tensor = AnfAlgo::CreateKernelTensor(
    abstract::kNoShape, type, nullptr, nullptr, GetTypeByte(TypeIdToType(type->type_id())), kOpFormat_DEFAULT,
    type_ptr->type_id(), ShapeVector(), device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(stream_id);
  auto device_address = kernel_tensor->device_address();
  device_address->set_from_persistent_mem(true);

  if (device_address->GetPtr() == nullptr) {
    CopyNoneTensorDataToDevice(device_context, kernel_tensor);
  }
  MS_LOG(DEBUG) << "Create input " << type_ptr->ToString() << " device address for " << index
                << "th input, Shape: " << abstract::kNoShape->ToString() << ", Type: " << type->ToString()
                << ", Value: nullptr, device address:" << device_address
                << ", kernel tensor: " << kernel_tensor->ToString();
  return kernel_tensor;
}

void DeviceAddressUtils::CreateOutputTensorAddress(const DeviceContext *device_context, size_t stream_id,
                                                   const std::vector<tensor::TensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(device_context);
  const auto device_type = DeviceManagerConf::GetInstance()->device_type();

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &tensor = outputs[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto existing_addr = tensor->device_address();
    if (existing_addr != nullptr && existing_addr->GetDeviceType() == device_type) {
      MS_LOG(DEBUG) << "Output tensor " << tensor->ToString() << " already has device address "
                    << existing_addr->ToString();
      continue;
    }
    const auto &tensor_shape = tensor->shape();
    auto tensor_size = LongToSize(tensor->DataNBytes());
    const auto &format = GetFormatByTensorShape(device_context, tensor_shape);
    auto data_type = tensor->data_type();
    auto device_address =
      std::make_shared<device::DeviceAddress>(nullptr, tensor_size, device_context->GetDeviceType(), stream_id);
    MS_EXCEPTION_IF_NULL(device_address);
    tensor->set_device_address(device_address);
    tensor->set_format(format);
    MS_LOG(DEBUG) << "Create output tensor device address " << device_address << " for " << i
                  << "th output, Shape: " << tensor_shape << ", Type: " << TypeIdToType(data_type)->ToString()
                  << ", Size:" << tensor_size;
  }
}

void DeviceAddressUtils::CreateOutputTensorAddress(const DeviceContext *device_context, size_t stream_id,
                                                   const tensor::TensorPtr &output_tensor, size_t size) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(output_tensor);
  const auto &tensor_shape = output_tensor->shape();
  auto data_type = output_tensor->data_type();
  const auto &format = GetFormatByTensorShape(device_context, tensor_shape);
  auto device_address =
    std::make_shared<device::DeviceAddress>(nullptr, size, device_context->GetDeviceType(), stream_id);
  MS_EXCEPTION_IF_NULL(device_address);
  output_tensor->set_device_address(device_address);
  output_tensor->set_format(format);
  MS_LOG(DEBUG) << "Create output tensor device address " << device_address
                << "the output, Shape: " << static_cast<int64_t>(size / GetTypeByte(TypeIdToType(data_type)))
                << ", Type: " << TypeIdToType(data_type)->ToString() << ", Size:" << size;
}

device::DeviceAddressPtr DeviceAddressUtils::CreateDeviceAddress(const DeviceContext *device_context,
                                                                 const tensor::TensorPtr &tensor,
                                                                 const ShapeVector &real_shape,
                                                                 const size_t &stream_id) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(tensor);
  auto tensor_size = GetTypeByte(TypeIdToType(tensor->data_type())) * SizeOf(real_shape);
  const auto &device_format = GetFormatByTensorShape(device_context, tensor->shape());
  auto kernel_tensor =
    AnfAlgo::CreateKernelTensor(nullptr, tensor_size, device_format, tensor->data_type(), real_shape,
                                device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(stream_id);
  device::DeviceAddressPtr device_address = kernel_tensor->device_address();
  MS_LOG(DEBUG) << "Create tensor device address " << kernel_tensor->ToString() << "Shape: " << tensor->shape()
                << ", Type: " << TypeIdToType(tensor->data_type())->ToString();
  return device_address;
}

KernelTensorPtr DeviceAddressUtils::CreateKernelTensor(const DeviceContext *device_context,
                                                       const tensor::TensorPtr &tensor, const ShapeVector &real_shape,
                                                       const size_t &stream_id) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(tensor);
  auto tensor_size = GetTypeByte(TypeIdToType(tensor->data_type())) * SizeOf(real_shape);
  const auto &device_format = GetFormatByTensorShape(device_context, tensor->shape());
  auto kernel_tensor =
    AnfAlgo::CreateKernelTensor(nullptr, tensor_size, device_format, tensor->data_type(), real_shape,
                                device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(stream_id);
  MS_LOG(DEBUG) << "Create kernel tensor " << kernel_tensor->ToString() << "Shape: " << tensor->shape()
                << ", Type: " << TypeIdToType(tensor->data_type())->ToString() << ", kernel tensor: " << kernel_tensor;
  return kernel_tensor;
}

void DeviceAddressUtils::MallocForOutputs(const DeviceContext *device_context,
                                          const std::vector<tensor::TensorPtr> &outputs) {
  MallocForOutputs(device_context, CurrentStream::id(), outputs);
}

void DeviceAddressUtils::MallocForOutputs(const DeviceContext *device_context, size_t stream_id,
                                          const std::vector<tensor::TensorPtr> &outputs) {
  for (const auto &output : outputs) {
    auto device_address = std::static_pointer_cast<device::DeviceAddress>(output->device_address());
    if (device_address->GetPtr() != nullptr) {
      // ref output
      continue;
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                   device_address->GetSize(), device_address.get());
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get(), stream_id)) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }
  }
}

device::DeviceAddressPtr DeviceAddressUtils::CreateWorkspaceAddressWithoutKernelTensor(
  const DeviceContext *device_context, size_t stream_id, const size_t &workspace_size, bool no_exception) {
  MS_EXCEPTION_IF_NULL(device_context);
  auto device_address =
    std::make_shared<device::DeviceAddress>(nullptr, workspace_size, device_context->GetDeviceType(), stream_id);
  MS_EXCEPTION_IF_NULL(device_address);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "WorkspaceAddress", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kWorkSpace,
                                                 device_address->GetSize(), device_address.get());
  if (device_address->GetPtr() == nullptr &&
      !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
    if (!no_exception) {
      MS_LOG(EXCEPTION) << "Allocate dynamic workspace memory failed";
    }
  }
  MS_LOG(DEBUG) << "Create workspace device address:" << device_address;
  return device_address;
}

KernelTensorPtr DeviceAddressUtils::CreateWorkspaceKernelTensor(const DeviceContext *device_context, size_t stream_id,
                                                                const size_t &workspace_size) {
  MS_EXCEPTION_IF_NULL(device_context);

  auto kernel_tensor =
    AnfAlgo::CreateKernelTensor(nullptr, workspace_size, Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
                                device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(stream_id);

  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Graph", "WorkspaceAddress", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "Graph", device::tracker::MemType::kWorkSpace,
                                                 device_address->GetSize(), device_address.get());
  if (device_address->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate dynamic workspace memory failed";
    } else {
      static std::string name = "Alloc memory";
      kernel_tensor->IncreaseNewRefCount(name);
    }
  }
  MS_LOG(DEBUG) << "Create workspace kernel tensor:" << kernel_tensor->ToString();
  return kernel_tensor;
}

tensor::TensorPtr DeviceAddressUtils::TensorContiguous(const tensor::TensorPtr &tensor) { return nullptr; }

void DeviceAddressUtils::ConvertContiguousTensorSync(const tensor::TensorPtr &tensor, size_t stream_id) {
  if (tensor == nullptr || tensor->storage_info() == nullptr) {
    return;
  }

  MS_LOG(DEBUG) << "Tensor storage_info is not nullptr, need to contiguous, id:" << tensor->id();
  const auto &new_device_address = ConvertContiguousDeviceAddress(nullptr, tensor, stream_id);
  MS_EXCEPTION_IF_NULL(new_device_address);
  tensor->set_device_address(new_device_address);
  tensor->set_storage_info(nullptr);
}

device::DeviceAddressPtr DeviceAddressUtils::ConvertContiguousDeviceAddress(const DeviceContext *input_device_context,
                                                                            const tensor::TensorPtr &input_tensor,
                                                                            size_t stream_id) {
  const auto &old_device_address = input_tensor->device_address();
  MS_EXCEPTION_IF_NULL(old_device_address);
  const DeviceContext *device_context = input_device_context;
  if (device_context == nullptr) {
    auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device::GetDeviceTypeByName(device_name), device_id});
  }

  MS_EXCEPTION_IF_NULL(device_context);
  if (stream_id == SIZE_MAX) {
    stream_id = device_context->device_res_manager_->GetCurrentStreamId();
  }

  GilReleaseWithCheck release_gil;
  const auto &old_storage_info = input_tensor->storage_info();
  if (old_storage_info == nullptr) {
    return old_device_address;
  }

  auto address_size = GetTypeByte(TypeIdToType(input_tensor->data_type())) * SizeOf(old_storage_info->shape);
  auto new_device_address =
    std::make_shared<device::DeviceAddress>(nullptr, address_size, device_context->GetDeviceType(), stream_id);

  auto output_tensor =
    std::make_shared<tensor::Tensor>(input_tensor->data_type(), old_storage_info->shape, new_device_address);
  MS_LOG(DEBUG) << "Create tensor:" << output_tensor->ToString();

  // ExecuteKernelTask sync, need to wait until all tasks in queue are complete.
  runtime::Pipeline::Get().WaitForward();
  if (!device_context->GetKernelExecutor()->ExecuteKernelTask(runtime::KernelTaskType::kCONTIGUOUS_TASK, {input_tensor},
                                                              {output_tensor}, stream_id)) {
    MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type:" << runtime::KernelTaskType::kCONTIGUOUS_TASK;
  }
  runtime::Pipeline::Get().WaitForward();

  return new_device_address;
}

void DeviceAddressUtils::GetCrossStreamAddressInfoFromInput(
  size_t op_stream_id, std::vector<std::pair<uint32_t, void *>> *cross_stream_addresses,
  const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->device_address() == nullptr) {
    return;
  }

  auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  if (op_stream_id != device_address->stream_id()) {
    // Device address is cross stream.
    MS_EXCEPTION_IF_NULL(cross_stream_addresses);
    (void)cross_stream_addresses->emplace_back(device_address->stream_id(), device_address->GetMutablePtr());
  }
}

void DeviceAddressUtils::GetCrossStreamAddressInfoFromInput(
  size_t op_stream_id, std::vector<std::pair<uint32_t, void *>> *cross_stream_addresses,
  const mindspore::kernel::KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (op_stream_id != tensor->stream_id()) {
    MS_EXCEPTION_IF_NULL(cross_stream_addresses);
    (void)cross_stream_addresses->emplace_back(tensor->stream_id(), tensor->device_ptr());
  }
}

void DeviceAddressUtils::GetCrossStreamAddressInfoFromInput(
  size_t op_stream_id, std::vector<std::pair<uint32_t, void *>> *cross_stream_addresses,
  const device::DeviceAddressPtr &device_address) {
  MS_EXCEPTION_IF_NULL(device_address);
  if (op_stream_id != device_address->stream_id()) {
    MS_EXCEPTION_IF_NULL(cross_stream_addresses);
    (void)cross_stream_addresses->emplace_back(device_address->stream_id(), device_address->GetMutablePtr());
  }
}
}  // namespace runtime
}  // namespace mindspore
