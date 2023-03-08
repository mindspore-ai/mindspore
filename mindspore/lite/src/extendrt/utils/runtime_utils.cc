/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <functional>
#include <vector>
#include <string>
#include <memory>

#include "extendrt/utils/runtime_utils.h"
#include "extendrt/utils/kernel_graph_utils.h"

#include "src/extendrt/infer_device_address.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace {
constexpr auto kNameCustomAscend = "CustomAscend";
const size_t tensor_max_size_utils = 0x1000000;
}  // namespace

void *RuntimeUtils::GetAddressPtr(device::DeviceAddressPtr address_ptr) {
  MS_EXCEPTION_IF_NULL(address_ptr);
  return address_ptr->ptr_;
}

void RuntimeUtils::SetAddressPtr(device::DeviceAddressPtr address_ptr, void *ptr) {
  MS_EXCEPTION_IF_NULL(address_ptr);
  address_ptr->ptr_ = ptr;
}

void RuntimeUtils::AllocAddressPtr(device::DeviceAddressPtr address_ptr) {
  MS_EXCEPTION_IF_NULL(address_ptr);
  if (address_ptr->ptr_ == nullptr) {
    address_ptr->ptr_ = malloc(address_ptr->size_);
  }
}

std::vector<AnfNodePtr> RuntimeUtils::GetGraphDataInputs(const KernelGraphPtr &kernel_graph) {
  std::vector<AnfNodePtr> data_inputs;
  auto inputs = kernel_graph->inputs();
  for (auto &input : inputs) {
    if (input->isa<Parameter>()) {
      auto parameter = input->cast<ParameterPtr>();
      if (parameter != nullptr && !parameter->has_default()) {
        data_inputs.emplace_back(input);
      }
    }
  }
  return data_inputs;
}

void RuntimeUtils::CopyInputTensorsToKernelGraph(const std::vector<tensor::Tensor> &inputs,
                                                 KernelGraphPtr kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto graph_inputs = GetGraphDataInputs(kernel_graph);
  if (graph_inputs.size() != inputs.size()) {
    MS_LOG(ERROR) << "Graph inputs size[" << graph_inputs.size() << "] is not equal to User input size[ "
                  << inputs.size() << "].";
    return;
  }
  for (size_t i = 0; i < graph_inputs.size(); i++) {
    auto &input = inputs[i];
    auto graph_input = graph_inputs[i];
    auto graph_input_addr = AnfAlgo::GetMutableOutputAddr(graph_input, 0);
    if (graph_input_addr == nullptr || graph_input_addr->ptr_ == nullptr) {
      MS_LOG(ERROR) << "Input " << i << " of input " << graph_input->DebugString() << " output addr ptr is nullptr.";
      return;
    }
    if (graph_input_addr->size_ < input.Size()) {
      MS_LOG(ERROR) << "Graph input " << i << " size[" << graph_input_addr->size_ << "] is less then user input size[ "
                    << input.Size() << "]";
      return;
    }
    memcpy(graph_input_addr->ptr_, input.data_c(), input.Size());
  }
}

void RuntimeUtils::CopyOutputTensorsFromKernelGraph(std::vector<tensor::Tensor> *outputs, KernelGraphPtr kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  outputs->clear();
  auto graph_outputs = KernelGraphUtils::GetKernelGraphOutputs(kernel_graph);
  for (auto graph_output : graph_outputs) {
    auto real_output_with_index = common::AnfAlgo::VisitKernelWithReturnType(graph_output, 0);
    auto real_output = real_output_with_index.first;
    MS_EXCEPTION_IF_NULL(real_output);
    MS_LOG(DEBUG) << " Real output info: " << real_output->DebugString();
    auto graph_output_address = AnfAlgo::GetMutableOutputAddr(real_output, real_output_with_index.second);
    auto data = graph_output_address->ptr_;
    auto data_size = graph_output_address->size_;
    auto type_id = graph_output_address->type_id_;
    auto uint_shape = AnfAlgo::GetOutputDeviceShape(real_output, real_output_with_index.second);
    std::vector<int64_t> shape;
    for (auto us : uint_shape) {
      auto s = static_cast<int64_t>(us);
      shape.push_back(s);
    }
    outputs->emplace_back(mindspore::tensor::Tensor(type_id, shape, data, data_size));
  }
}

kernel::AddressPtr RuntimeUtils::GetAddressFromDevice(device::DeviceAddressPtr device_address) {
  MS_EXCEPTION_IF_NULL(device_address);
  kernel::AddressPtr kernel_address = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(kernel_address);
  if (device_address->ptr_ == nullptr) {
    device_address->ptr_ = malloc(device_address->size_);
  }
  MS_EXCEPTION_IF_NULL(device_address->ptr_);
  kernel_address->addr = device_address->ptr_;
  kernel_address->size = device_address->size_;
  return kernel_address;
}

void RuntimeUtils::AssignKernelGraphAddress(KernelGraphPtr kernel_graph) {
  AssignValueNodeAddress(kernel_graph);
  AssignInputNodeAddress(kernel_graph);
  AssignKernelOutputAddress(kernel_graph);
}

void RuntimeUtils::AssignValueNodeAddress(KernelGraphPtr kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &item_node : kernel_graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(item_node);
    if (item_node->isa<ValueNode>()) {
      auto value_node = item_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto node_value = value_node->value();
      MS_EXCEPTION_IF_NULL(node_value);
      if (!node_value->isa<tensor::Tensor>()) {
        continue;
      }
      auto tensor = node_value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor->device_address() != nullptr) {
        AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), 0,
                               item_node.get());
        continue;
      }
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item_node, 0);
      if (output_type_id == kTypeUnknown) {
        output_type_id = common::AnfAlgo::GetOutputInferDataType(item_node, 0);
      }
      size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
      ShapeVector data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(), type_size, std::multiplies<size_t>());
      mindspore::device::DeviceAddressPtr address = nullptr;
      address = CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, output_type_id);
      address->set_from_persistent_mem(tensor->is_parameter());
      MS_EXCEPTION_IF_NULL(address);
      if (tensor->data_type() == output_type_id) {
        address->ptr_ = tensor->data_c();
      } else {
        if (tensor_size == 0 || tensor_size >= tensor_max_size_utils) {
          MS_LOG(WARNING) << "tensor is too big with size " << tensor_max_size_utils;
          continue;
        }
        RuntimeUtils::SetAddressPtr(address, malloc(tensor_size));
        if (!address->SyncHostToDevice(data_shape, LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                       tensor->data_c())) {
          MS_LOG(EXCEPTION) << "Value node sync host to device failed!";
        }
      }
      address->ref_count_ = 1;
      AnfAlgo::SetOutputAddr(address, 0, item_node.get());
    }
  }
}

void RuntimeUtils::AssignInputNodeAddress(KernelGraphPtr kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &item : kernel_graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(item);
    if (item->isa<Parameter>()) {
      auto output_num = AnfUtils::GetOutputTensorNum(item);
      for (size_t index = 0; index < output_num; index++) {
        TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
        if (output_type_id == kTypeUnknown) {
          output_type_id = common::AnfAlgo::GetOutputInferDataType(item, index);
        }
        auto fmt_shape = AnfAlgo::GetOutputDeviceShape(item, index);
        size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
        size_t tensor_size;
        if (std::any_of(fmt_shape.begin(), fmt_shape.end(), [](int64_t shape) { return shape < 0; })) {
          tensor_size = type_size;
        } else {
          tensor_size = fmt_shape.empty()
                          ? type_size
                          : std::accumulate(fmt_shape.begin(), fmt_shape.end(), type_size, std::multiplies<size_t>());
        }
        auto format = AnfAlgo::GetOutputFormat(item, index);
        auto address = CreateDeviceAddress(malloc(tensor_size), tensor_size, format, output_type_id);
        address->set_from_persistent_mem(true);
        AnfAlgo::SetOutputAddr(address, index, item.get());
      }
    }
  }
}

void RuntimeUtils::AssignKernelOutputAddress(KernelGraphPtr kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto kernels = kernel_graph->execution_order();
  for (auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      AnfAlgo::SetOutputAddr(CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type), i,
                             kernel.get());
    }
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      AnfAlgo::SetWorkspaceAddr(CreateDeviceAddress(nullptr, workspace_sizes[i], kOpFormat_DEFAULT, kNumberTypeFloat32),
                                i, kernel.get());
    }
  }
}

device::DeviceAddressPtr RuntimeUtils::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                           TypeId type_id) {
  return std::make_shared<InferDeviceAddress>(device_ptr, device_size, format, type_id);
}

void RuntimeUtils::UpdateKernelNodeOutputInfo(const AnfNodePtr &kernel_node,
                                              const std::vector<kernel::AddressPtr> &output_addrs) {
  std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == kNameCustomAscend) {
    size_t output_num = AnfUtils::GetOutputTensorNum(kernel_node);
    if (output_addrs.size() != output_num) {
      MS_LOG(ERROR) << "Output addr size[" << output_addrs.size() << "] is not equal to node outputs size["
                    << output_num << "]";
      return;
    }
    // update output addr
    bool is_update_shape = false;
    for (size_t i = 0; i < output_num; ++i) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(kernel_node, i);
      auto addr_ptr = device_address->GetMutablePtr();
      if (addr_ptr != nullptr && output_addrs[i]->addr != addr_ptr) {
        free(addr_ptr);
        device_address->set_ptr(output_addrs[i]->addr);
        device_address->SetSize(output_addrs[i]->size);
        is_update_shape = true;
      }
    }
    if (!is_update_shape) {
      MS_LOG(DEBUG) << "There is no need to update output shape.";
      return;
    }
    // update output shape
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto kernel_tensors = kernel_mod->RetrieveOutputShape();
    if (kernel_tensors.empty()) {
      MS_LOG(ERROR) << "The output shape size of custom ascend is empty.";
      return;
    }
    auto abstract = kernel_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (utils::isa<abstract::AbstractTuplePtr>(abstract)) {
      auto abstract_tuple = abstract->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(abstract_tuple);
      if (abstract_tuple->elements().size() != kernel_tensors.size()) {
        MS_LOG(ERROR) << "Abstract size[" << abstract_tuple->elements().size() << "] is not equal to output shape size["
                      << kernel_tensors.size() << "]";
        return;
      }
      for (size_t i = 0; i < abstract_tuple->elements().size(); ++i) {
        auto tmp_abstract = abstract_tuple->elements()[i];
        MS_EXCEPTION_IF_NULL(tmp_abstract);
        tmp_abstract->set_shape(std::make_shared<abstract::Shape>(kernel_tensors[i]->GetShapeVector()));
      }
    } else {
      abstract->set_shape(std::make_shared<abstract::Shape>(kernel_tensors[0]->GetShapeVector()));
    }
  }
}
}  // namespace mindspore
