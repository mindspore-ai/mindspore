/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License"){}
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

#include "runtime/framework/graph_compiler.h"
#include <numeric>
#include <map>
#include "runtime/framework/graph_scheduler.h"
#include "runtime/device/device_address.h"
#include "common/trans.h"
#include "utils/convert_utils.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
namespace {
// Whether device address of anf node is valid and device address type
// is consistent with device type, for example, device address type
// DeviceAddressType::kGPU should be used on GPU device
bool NodeDeviceAddressExist(const DeviceContext *device_context, const AnfNodePtr &kernel, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(device_context);
  if (AnfAlgo::OutputAddrExist(kernel, index)) {
    const auto &address = AnfAlgo::GetOutputAddr(kernel, index);
    MS_EXCEPTION_IF_NULL(address);
    return address->DeviceType() == device_context->GetDeviceAddressType();
  }
  return false;
}

void CreateParameterDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_inputs = graph->inputs();
  const std::vector<bool> &graph_valid_input = graph->valid_inputs();
  graph_inputs.insert(graph_inputs.end(), graph->child_graph_result().begin(), graph->child_graph_result().end());

  // Anf nodes which need create device address.
  std::vector<AnfNodePtr> nodes_list;
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    AnfNodePtr item = graph_inputs[i];
    MS_EXCEPTION_IF_NULL(item);
    if (i < graph_valid_input.size() && !graph_valid_input[i]) {
      continue;
    }

    if (AnfAlgo::CheckPrimitiveType(item, prim::kPrimMakeTuple)) {
      std::vector<AnfNodePtr> outs = AnfAlgo::GetAllOutput(item);
      for (const auto &out : outs) {
        MS_EXCEPTION_IF_NULL(out);
        if (!out->isa<Parameter>() || NodeDeviceAddressExist(device_context, out, 0)) {
          continue;
        }
        nodes_list.push_back(out);
      }
    }
    if (!item->isa<Parameter>() || NodeDeviceAddressExist(device_context, item, 0)) {
      continue;
    }
    nodes_list.push_back(item);
  }

  // Create device address for anf node in nodes_list
  for (const auto &item : nodes_list) {
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      // if graph output is a weight and doesn't link to any cnode, it's data type will be unknown
      if (output_type_id == kTypeUnknown) {
        MS_LOG(WARNING) << "It is not suggested to use a lonely weight parameter as the output of graph";
        continue;
      }

      size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
      auto device_address = device_context->CreateDeviceAddress(nullptr, tensor_size,
                                                                AnfAlgo::GetOutputFormat(item, index), output_type_id);
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
}

void CreateDeviceAddressForTensorValue(const DeviceContext *device_context, const ValuePtr &node_value,
                                       size_t output_idx, const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);

  for (const auto &tensor : tensors) {
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (output_address != nullptr && output_address->DeviceType() == device_context->GetDeviceAddressType()) {
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), output_idx++,
                             value_node.get());
      continue;
    }

    size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(value_node, output_idx);
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
    if (output_type_id == kTypeUnknown) {
      output_type_id = AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
    std::string output_format = AnfAlgo::GetOutputFormat(value_node, output_idx);

    device::DeviceAddressPtr address =
      device_context->CreateDeviceAddress(nullptr, tensor_size, output_format, output_type_id);
    MS_EXCEPTION_IF_NULL(address);
    AnfAlgo::SetOutputAddr(address, output_idx, value_node.get());
  }
}

void CreateValueNodeDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  for (const ValueNodePtr &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (NodeDeviceAddressExist(device_context, value_node, 0)) {
      continue;
    }

    const auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueTuple>()) {
      CreateDeviceAddressForTensorValue(device_context, node_value, 0, value_node);
    } else if (node_value->isa<StringImm>()) {
      auto value = GetValue<std::string>(node_value);
      size_t tensor_size = value.size();
      auto address = device_context->CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, kNumberTypeUInt8);
      MS_EXCEPTION_IF_NULL(address);

      AnfAlgo::SetOutputAddr(address, 0, value_node.get());
    }
  }
}

void CreateKernelOutputDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      if (AnfAlgo::OutputAddrExist(kernel, i)) {
        continue;
      }

      std::string output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      auto device_address = device_context->CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type);
      AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
    }
  }
}

void CreateKernelWorkspaceDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      auto device_address = device_context->CreateDeviceAddress(nullptr, workspace_sizes[i], "", kTypeUnknown);
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());
    }
  }
}
}  // namespace

void GraphCompiler::set_device_context(DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  device_context_ = device_context;

  // The member variable 'session_' will be removed after removing session module.
  if (session_ == nullptr) {
    session_ = std::make_shared<session::SessionBasic>();
    const device::DeviceContextKey &device_context_key = device_context->device_context_key();
    session_->InitExecutor(device_context_key.device_name_, device_context_key.device_id_);
  }
}

GraphId GraphCompiler::CompileGraph(const AnfNodePtrList &nodes, const AnfNodePtrList &outputs) {
  MS_EXCEPTION_IF_NULL(session_);
  // Generate kernel graph.
  KernelGraphPtr graph = session_->ConstructKernelGraph(nodes, outputs);
  MS_EXCEPTION_IF_NULL(graph);
  return CompileGraphImpl(graph);
}

GraphId GraphCompiler::CompileGraphImpl(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context_);
  // Optimization pass which is irrelevant to device type or format.
  device_context_->OptimizeGraphWithoutDeviceInfo(graph);

  device_context_->SetOperatorInfo(graph->execution_order());

  // Optimization pass which is relevant to device type or format.
  device_context_->OptimizeGraphWithDeviceInfo(graph);

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  device_context_->CreateKernel(graph->execution_order());

  // Create device address for all anf nodes of graph.
  CreateDeviceAddress(graph);

  // Transform graph to actor DAG, contains build and link.
  const auto &actor_set = GraphScheduler::GetInstance().Transform(graph, device_context_);
  GraphScheduler::GetInstance().Schedule(actor_set);

  return graph->graph_id();
}

GraphId GraphCompiler::CompileGraph(session::OpRunInfo *op_run_info, const GraphInfo &graph_info,
                                    std::vector<tensor::TensorPtr> *input_tensors,
                                    const std::vector<int64_t> &tensors_mask) {
  // Check if the graph cache exists.
  auto iter = run_op_graphs_.find(graph_info);
  if (iter != run_op_graphs_.end()) {
    const auto &graph = iter->second;
    MS_EXCEPTION_IF_NULL(graph);
    return graph->graph_id();
  }
  // Generate kernel graph.
  MS_EXCEPTION_IF_NULL(session_);
  KernelGraphPtr graph = session_->ConstructSingleOpGraph(*op_run_info, *input_tensors, tensors_mask);
  MS_EXCEPTION_IF_NULL(graph);

  MS_EXCEPTION_IF_NULL(device_context_);
  device_context_->SetOperatorInfo(graph->execution_order());

  device_context_->OptimizeSingleOpGraph(graph);
  MS_EXCEPTION_IF_NULL(session_);
  session_->RunOpHideNopNode(graph);
  session_->RunOpRemoveNopNode(graph);

  // Generate 'KernelMod' for kernel in graph.
  device_context_->CreateKernel(graph->execution_order());

  // Create device address for all anf nodes of graph.
  CreateDeviceAddress(graph);
  // Transform graph to actor DAG, contains build and link.
  GraphScheduler::GetInstance().Transform(graph, device_context_, input_tensors, GraphExecutionStrategy::kStep);
  run_op_graphs_[graph_info] = graph;
  return graph->graph_id();
}

KernelGraphPtr GraphCompiler::Fetch(GraphId graph_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  return session_->GetGraph(graph_id);
}

KernelGraphPtr GraphCompiler::Fetch(const GraphInfo &graph_info) const {
  auto iter = run_op_graphs_.find(graph_info);
  if (iter == run_op_graphs_.end()) {
    MS_LOG(ERROR) << "Can't find graph for: " << graph_info;
    return nullptr;
  }
  return iter->second;
}

void GraphCompiler::CreateDeviceAddress(const KernelGraphPtr &graph) const {
  CreateParameterDeviceAddress(device_context_, graph);
  CreateValueNodeDeviceAddress(device_context_, graph);
  CreateKernelOutputDeviceAddress(device_context_, graph);
  CreateKernelWorkspaceDeviceAddress(device_context_, graph);
}
}  // namespace runtime
}  // namespace mindspore
