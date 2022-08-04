/**
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

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "src/extendrt/single_op_session.h"
#include "src/extendrt/infer_device_address.h"

#include "plugin/factory/ms_factory.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"
#include "src/extendrt/utils/kernel_build_utils.h"

namespace mindspore {
const size_t tensor_max_size = 0x1000000;

Status SingleOpInferSession::Init(const std::shared_ptr<Context> context) {
  MS_LOG(INFO) << "SingleOpInferSession::Init";
  session_basic_ = std::make_shared<session::SessionBasic>();
  return kSuccess;
}

Status SingleOpInferSession::CompileGraph(FuncGraphPtr graph) {
  MS_LOG(INFO) << "SingleOpInferSession::CompileGraph";
  std::vector<KernelGraphPtr> all_out_graph;
  kernel_graph_ = session_basic_->ConstructKernelGraph(graph, &all_out_graph, mindspore::device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(kernel_graph_);

  auto &nodes = kernel_graph_->nodes();
  for (const auto &node : nodes) {
    std::string node_name = common::AnfAlgo::GetCNodeName(node);
    MS_LOG(INFO) << "SingleOpInferSession::Nodes " << node_name;
  }

  auto &kernel_nodes = kernel_graph_->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    mindspore::infer::SetKernelInfo(kernel_node);
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    std::shared_ptr<kernel::CpuKernelMod> cpu_kernel_mod =
      kernel::Factory<kernel::CpuKernelMod>::Instance().Create(kernel_name);
    MS_LOG(INFO) << "SingleOpInferSession::Kernels " << kernel_name;
    auto args = kernel::AbstractArgsFromCNode(kernel_node);
    auto ret = cpu_kernel_mod->Init(args.op, args.inputs, args.outputs);
    MS_LOG(INFO) << "SingleOpInferSession::Kernels ret " << ret;

    std::vector<size_t> input_size_list;
    std::vector<size_t> output_size_list;
    input_size_list.clear();
    output_size_list.clear();
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      TypeId type_id = AnfAlgo::GetInputDeviceDataType(kernel_node, input_index);
      size_t type_size = GetTypeByte(TypeIdToType(type_id));
      auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, input_index);
      size_t tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      tensor_size = std::max(tensor_size, type_size);
      (void)input_size_list.emplace_back(tensor_size);
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      TypeId type_id = AnfAlgo::GetOutputDeviceDataType(kernel_node, output_index);
      size_t type_size = GetTypeByte(TypeIdToType(type_id));
      auto shape = AnfAlgo::GetOutputDeviceShape(kernel_node, output_index);
      size_t tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      tensor_size = std::max(tensor_size, type_size);
      (void)output_size_list.emplace_back(tensor_size);
    }
    cpu_kernel_mod->SetInputSizeList(input_size_list);
    cpu_kernel_mod->SetOutputSizeList(output_size_list);

    AnfAlgo::SetKernelMod(cpu_kernel_mod, kernel_node.get());
  }

  this->AssignKernelGraphAddress(kernel_graph_);

  session_basic_->GetModelInputsInfo(kernel_graph_->graph_id(), &inputs_, &input_names_);
  session_basic_->GetModelOutputsInfo(kernel_graph_->graph_id(), &outputs_, &output_names_);

  return kSuccess;
}

Status SingleOpInferSession::RunGraph() { return kSuccess; }
Status SingleOpInferSession::RunGraph(const std::vector<tensor::TensorPtr> &inputs,
                                      std::vector<tensor::TensorPtr> *outputs) {
  MS_LOG(INFO) << "SingleOpInferSession::RunGraph with input and outputs";
  MS_EXCEPTION_IF_NULL(kernel_graph_);

  CopyInputs(inputs);

  auto &kernel_nodes = kernel_graph_->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_LOG(INFO) << "SingleOpInferSession::RunGraph " << kernel_name;
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    std::vector<kernel::AddressPtr> kernel_inputs;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_num; ++i) {
      auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel_node, i).get();
      MS_EXCEPTION_IF_NULL(device_address);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      if (device_address->ptr_ == nullptr) {
        device_address->ptr_ = malloc(device_address->size_);
      }
      MS_EXCEPTION_IF_NULL(device_address->ptr_);
      input->addr = device_address->ptr_;
      input->size = device_address->size_;
      kernel_inputs.push_back(input);
    }
    std::vector<kernel::AddressPtr> kernel_outputs;
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t i = 0; i < output_num; ++i) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(kernel_node, i).get();
      MS_EXCEPTION_IF_NULL(device_address);
      kernel::AddressPtr output = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(output);
      if (device_address->ptr_ == nullptr) {
        device_address->ptr_ = malloc(device_address->size_);
      }
      MS_EXCEPTION_IF_NULL(device_address->ptr_);
      output->addr = device_address->ptr_;
      output->size = device_address->size_;
      kernel_outputs.push_back(output);
    }
    std::vector<kernel::AddressPtr> kernel_workspaces;
    bool ret = true;
    try {
      ret = kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, 0);
    } catch (std::exception &e) {
      MS_LOG(EXCEPTION) << e.what();
    }
    if (!ret) {
      MS_LOG(EXCEPTION) << "Launch kernel failed.";
    }
  }

  CopyOutputs(outputs);

  return kSuccess;
}
Status SingleOpInferSession::Resize(const std::vector<tensor::TensorPtr> &inputs,
                                    const std::vector<std::vector<int64_t>> &dims) {
  return kSuccess;
}
std::vector<tensor::TensorPtr> SingleOpInferSession::GetOutputs() { return outputs_; }
std::vector<tensor::TensorPtr> SingleOpInferSession::GetInputs() { return inputs_; }
std::vector<std::string> SingleOpInferSession::GetOutputNames() { return output_names_; }
std::vector<std::string> SingleOpInferSession::GetInputNames() { return input_names_; }
tensor::TensorPtr SingleOpInferSession::GetOutputByTensorName(const std::string &tensorName) { return nullptr; }
tensor::TensorPtr SingleOpInferSession::GetInputByTensorName(const std::string &name) { return nullptr; }

void SingleOpInferSession::AssignKernelGraphAddress(KernelGraphPtr kernel_graph) {
  this->AssignValueNodeAddress(kernel_graph);
  this->AssignInputNodeAddress(kernel_graph);
  this->AssignKernelOutputAddress(kernel_graph);
}

void SingleOpInferSession::AssignValueNodeAddress(KernelGraphPtr kernel_graph) {
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
        if (tensor_size == 0 || tensor_size >= tensor_max_size) {
          MS_LOG(WARNING) << "tensor is too big with size " << tensor_max_size;
          continue;
        }
        address->ptr_ = malloc(tensor_size);
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

void SingleOpInferSession::AssignInputNodeAddress(KernelGraphPtr kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &item : kernel_graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(item);
    if (item->isa<Parameter>()) {
      auto output_num = common::AnfAlgo::GetOutputTensorNum(item);
      for (size_t index = 0; index < output_num; index++) {
        TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
        if (output_type_id == kTypeUnknown) {
          output_type_id = common::AnfAlgo::GetOutputInferDataType(item, index);
        }
        auto fmt_shape = AnfAlgo::GetOutputDeviceShape(item, index);
        size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
        size_t tensor_size =
          fmt_shape.empty() ? type_size
                            : std::accumulate(fmt_shape.begin(), fmt_shape.end(), type_size, std::multiplies<size_t>());
        auto format = AnfAlgo::GetOutputFormat(item, index);
        auto address = CreateDeviceAddress(malloc(tensor_size), tensor_size, format, output_type_id);
        address->set_from_persistent_mem(true);
        AnfAlgo::SetOutputAddr(address, index, item.get());
      }
    }
  }
}

void SingleOpInferSession::AssignKernelOutputAddress(KernelGraphPtr kernel_graph) {
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

device::DeviceAddressPtr SingleOpInferSession::CreateDeviceAddress(void *device_ptr, size_t device_size,
                                                                   const string &format, TypeId type_id) const {
  return std::make_shared<InferDeviceAddress>(device_ptr, device_size, format, type_id);
}

void SingleOpInferSession::CopyInputs(const std::vector<tensor::TensorPtr> inputs) {
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  auto graph_inputs = kernel_graph_->inputs();
  for (size_t i = 0; i < graph_inputs.size(); i++) {
    auto input = inputs[i];
    auto graph_input = graph_inputs[i];
    auto graph_input_addr = AnfAlgo::GetMutableOutputAddr(graph_input, 0).get();
    memcpy(graph_input_addr->ptr_, input->data_c(), graph_input_addr->size_);
  }
}

void SingleOpInferSession::CopyOutputs(std::vector<tensor::TensorPtr> *outputs) {
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  auto graph_outputs = kernel_graph_->outputs();

  for (auto graph_output : graph_outputs) {
    auto graph_output_address = AnfAlgo::GetMutableOutputAddr(graph_output, 0).get();
    auto data = graph_output_address->ptr_;
    auto data_size = graph_output_address->size_;
    auto type_id = graph_output_address->type_id_;
    auto uint_shape = AnfAlgo::GetOutputDeviceShape(graph_output, 0);
    std::vector<int64_t> shape;
    for (auto us : uint_shape) {
      auto s = static_cast<int64_t>(us);
      shape.push_back(s);
    }
    auto tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(type_id, shape, data, data_size);
    outputs->push_back(tensor_ptr);
  }
}
}  // namespace mindspore
