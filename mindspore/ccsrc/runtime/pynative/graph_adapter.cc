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

#include "runtime/pynative/graph_adapter.h"

#include <string>
#include <memory>
#include <vector>
#include "ir/tensor.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "runtime/graph_scheduler/device_tensor_store.h"

namespace mindspore::pynative {
namespace {
constexpr auto kAttrBpropValueNodeRefCount = "bprop_value_node_ref_count";
constexpr auto kAttrValueNodeForwardOuputFlags = "value_node_forward_output_flags";

tensor::TensorPtr GetTensorFromValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    return nullptr;
  }
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  // ValueTuple is already expanded into tensors in backend.
  if (!value->isa<tensor::Tensor>()) {
    MS_LOG(DEBUG) << "Only need to process forward output tensor. value:" << value->ToString();
    return nullptr;
  }

  auto tensor = value->cast<tensor::TensorPtr>();
  return tensor;
}
}  // namespace

void GraphAdapter::ClearForwardOutputValueNodeDeviceAddress(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor->is_forward_output()) {
        AnfAlgo::SetOutputAddr(nullptr, 0, value_node.get());
      }
    }
  }
}

// The device address of graph value node need to release
// if the value node is output of forward_graph in PyNative mode.
void GraphAdapter::GenerateRefCountForBpropValueNode(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  HashMap<std::string, size_t> tensor_counts;
  auto execution_nodes = graph->execution_order();
  for (auto &node : execution_nodes) {
    std::vector<session::KernelWithIndex> real_inputs;
    common::AnfAlgo::GetRealInputs(node, &real_inputs);
    for (auto &real_input : real_inputs) {
      auto forward_output_tensor = GetTensorFromValueNode(real_input.first);
      if (forward_output_tensor == nullptr || !forward_output_tensor->is_forward_output()) {
        continue;
      }
      tensor_counts[forward_output_tensor->id()] += 1;
    }
  }

  std::vector<size_t> value_node_ref_count;
  std::vector<bool> value_node_forward_output_flags;
  for (auto &value_node : graph->graph_value_nodes()) {
    auto tensor = GetTensorFromValueNode(value_node);
    if (tensor == nullptr || !tensor->is_forward_output()) {
      value_node_ref_count.emplace_back(SIZE_MAX);
      value_node_forward_output_flags.emplace_back(false);
      continue;
    }
    auto iter = tensor_counts.find(tensor->id());
    if (iter == tensor_counts.end()) {
      // The tensor is in bp graph but not used.
      // e.g. %1-MakeTuple(T1, T2) -> TupleGetItem(%1, 0). T2 is not used.
      MS_LOG(DEBUG) << "Tensor " << tensor->ToString() << " is not found in value node";
      value_node_ref_count.emplace_back(SIZE_MAX);
      value_node_forward_output_flags.emplace_back(false);
      continue;
    }

    value_node_ref_count.emplace_back(iter->second);
    value_node_forward_output_flags.emplace_back(true);
  }
  graph->set_attr(kAttrBpropValueNodeRefCount, MakeValue(value_node_ref_count));
  graph->set_attr(kAttrValueNodeForwardOuputFlags, MakeValue(value_node_forward_output_flags));
}

void GraphAdapter::UpdateForwardOutputInBpropGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Update start";
  auto value_node_ref_counts = GetValue<std::vector<size_t>>(graph->get_attr(kAttrBpropValueNodeRefCount));
  auto value_node_forward_output_flags = GetValue<std::vector<bool>>(graph->get_attr(kAttrValueNodeForwardOuputFlags));
  size_t value_node_size = graph->graph_value_nodes().size();
  if (value_node_ref_counts.size() != value_node_size || value_node_forward_output_flags.size() != value_node_size) {
    MS_LOG(EXCEPTION) << "value_node_ref_count.size " << value_node_ref_counts.size()
                      << " value_node_forward_output_flags.size " << value_node_forward_output_flags.size()
                      << " not equal to " << value_node_size;
  }

  size_t value_node_index = 0;
  HashMap<device::DeviceAddressPtr, size_t> address_ref_count;
  // Update ValueNode device address
  for (auto &value_node : graph->graph_value_nodes()) {
    auto is_forward_output = value_node_forward_output_flags[value_node_index];
    if (!is_forward_output) {
      value_node_index++;
      continue;
    }
    size_t value_node_ref_count = value_node_ref_counts[value_node_index++];
    auto tensor = GetTensorFromValueNode(value_node);
    MS_EXCEPTION_IF_NULL(tensor);
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (device_address == nullptr) {
      MS_LOG(WARNING) << "Forward output " << tensor->ToString() << " device address is null";
      continue;
    }

    if (device_address->GetDeviceType() != device::DeviceType::kCPU) {
      address_ref_count[device_address] += value_node_ref_count;
      device_address->set_from_tensor(tensor);
    }

    auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
    runtime::DeviceTensorStore::GetInstance().Insert(front_node.get(), device_address);
  }

  for (auto &[address, ref_count] : address_ref_count) {
    address->set_original_ref_count(ref_count);
    address->ResetRefCount();
    MS_LOG(DEBUG) << "device_address " << address.get() << " ref_count " << address->ref_count();
  }
  MS_LOG(DEBUG) << "Update end";
}

bool GraphAdapter::ReplaceBpropGraphParameter(const KernelGraphPtr &graph,
                                              const std::vector<tensor::TensorPtr> &input_tensors) {
  size_t index = 0;
  bool changed = false;
  for (const auto &input_node : graph->input_nodes()) {
    auto params = common::AnfAlgo::GetAllOutput(input_node);
    for (const auto &param : params) {
      if (index >= input_tensors.size()) {
        MS_LOG(EXCEPTION) << "Parameter size out of range. Parameter index: " << index
                          << ", input size: " << input_tensors.size();
      }
      const auto &input_tensor = input_tensors[index++];
      MS_EXCEPTION_IF_NULL(input_tensor);
      const auto &tensor_address = input_tensor->device_address();
      auto address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor_address);
      if (address != nullptr) {
        auto tensor_format = address->format();
        auto param_format = AnfAlgo::GetOutputFormat(param, 0);
        if (tensor_format != param_format) {
          // Update parameter format
          auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
          MS_EXCEPTION_IF_NULL(kernel_build_info_builder);
          kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{address->format()});
          kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{address->type_id()});
          kernel_build_info_builder->SetOutputsReshapeType({input_tensor->padding_type()});
          AnfAlgo::SetOutputAddr(address, 0, param.get());
          AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), param.get());

          // Update abstract
          auto type_of_tensor = input_tensor->Dtype();
          auto shape_of_tensor = input_tensor->shape();
          auto abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, shape_of_tensor);
          param->set_abstract(abstract);
          changed = true;
        }
      }
    }
  }
  return changed;
}
}  // namespace mindspore::pynative
