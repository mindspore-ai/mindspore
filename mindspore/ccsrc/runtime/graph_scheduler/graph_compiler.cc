/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/graph_compiler.h"
#include <numeric>
#include <map>
#include <utility>
#include <algorithm>
#include <functional>
#include "runtime/graph_scheduler/graph_scheduler.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/device/device_address.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/pynative/op_runtime_info.h"
#include "include/common/utils/convert_utils.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "kernel/common_utils.h"
#include "profiler/device/profiling.h"
#include "backend/common/optimizer/helper.h"
#include "base/base_ref_utils.h"
#include "include/common/debug/dump_proto.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/anf_ir_dump.h"
#endif
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#ifdef WITH_BACKEND
#include "ps/ps_context.h"
#include "runtime/graph_scheduler/embedding_cache_scheduler.h"
#endif

namespace mindspore {
namespace runtime {
namespace {
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

void CreateParameterDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
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

    if (common::AnfAlgo::CheckPrimitiveType(item, prim::kPrimMakeTuple)) {
      std::vector<AnfNodePtr> outs = common::AnfAlgo::GetAllOutput(item);
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
    auto output_size = common::AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      if (output_type_id == kTypeUnknown) {
        output_type_id = common::AnfAlgo::GetOutputInferDataType(item, index);
      }

      size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
      auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id,
        trans::GetRuntimePaddingShape(item, index));
      device_address->set_from_persistent_mem(item->isa<Parameter>());
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(item)
                    << " addr:" << device_address;
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
  std::vector<TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);

  for (const auto &tensor : tensors) {
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (output_address != nullptr && output_address->GetDeviceType() == device_context->GetDeviceType()) {
      // We need to set tensor->device_address to ValueNode even if the tensor is a forward_output tensor
      // in PyNative Bprop graph. ValueNode device_address is necessary for GraphSchedule::Transform.
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), output_idx++,
                             value_node.get());
      continue;
    }

    size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(value_node, output_idx);
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
    if (output_type_id == kTypeUnknown) {
      output_type_id = common::AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
    std::string output_format = AnfAlgo::GetOutputFormat(value_node, output_idx);

    device::DeviceAddressPtr address = device_context->device_res_manager_->CreateDeviceAddress(
      nullptr, tensor_size, output_format, output_type_id, trans::GetRuntimePaddingShape(value_node, output_idx));
    MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(value_node) << " addr:" << address;
    MS_EXCEPTION_IF_NULL(address);
    address->set_from_persistent_mem(true);
    AnfAlgo::SetOutputAddr(address, output_idx++, value_node.get());
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
      auto address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT,
                                                                              kNumberTypeUInt8, ShapeVector());
      MS_EXCEPTION_IF_NULL(address);
      address->set_from_persistent_mem(true);
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(value_node)
                    << " addr:" << address;

      AnfAlgo::SetOutputAddr(address, 0, value_node.get());
    }
  }
}

void CreateKernelOutputDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph,
                                     bool is_gradient_out) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);

  bool is_pynative_bprop_graph = graph->has_flag(kFlagIsPynativeBpropGraph);
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output());

  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsControlOpExecInBackend(kernel)) {
      continue;
    }

    bool is_from_persistent_mem =
      (is_gradient_out || (is_pynative_bprop_graph && (find(outputs.begin(), outputs.end(), kernel) != outputs.end())));

    auto output_size = AnfAlgo::GetOutputAddressNum(kernel);
    for (size_t i = 0; i < output_size; ++i) {
      if (AnfAlgo::OutputAddrExist(kernel, i)) {
        continue;
      }
      auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      auto address_size = AnfAlgo::GetOutputTensorMemSize(kernel, i);
      auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, address_size, output_format, output_type, trans::GetRuntimePaddingShape(kernel, i));
      if (is_from_persistent_mem) {
        device_address->set_from_persistent_mem(true);
      }
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(kernel)
                    << " addr:" << device_address;
      AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
    }
  }
}

void CreateKernelWorkspaceDeviceAddress(const DeviceContext *device_context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsControlOpExecInBackend(kernel)) {
      continue;
    }
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      if (AnfAlgo::WorkspaceAddrExist(kernel, i)) {
        break;
      }
      auto device_address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, workspace_sizes[i], "",
                                                                                     kTypeUnknown, ShapeVector());
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(kernel)
                    << " addr:" << device_address;
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());
    }
  }
}

void UpdateDeviceAddressForInplaceNode(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
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

void UpdateDeviceAddress(const session::AnfWithOutIndex &cur_pair, const session::AnfWithOutIndex &origin_pair) {
  MS_EXCEPTION_IF_NULL(cur_pair.first);
  MS_EXCEPTION_IF_NULL(origin_pair.first);

  auto origin_node_output_addr = AnfAlgo::GetMutableOutputAddr(origin_pair.first, origin_pair.second, false);
  MS_EXCEPTION_IF_NULL(origin_node_output_addr);
  auto cur_node_output_addr = AnfAlgo::GetMutableOutputAddr(cur_pair.first, cur_pair.second, false);
  MS_EXCEPTION_IF_NULL(cur_node_output_addr);

  if (origin_node_output_addr.get() != cur_node_output_addr.get()) {
    MS_LOG(INFO) << "Update device address: ref origin kernel is " << origin_pair.first->fullname_with_scope()
                 << ", index is " << origin_pair.second << ", cur kernel is " << cur_pair.first->fullname_with_scope()
                 << ", index is " << cur_pair.second;
    AnfAlgo::SetOutputAddr(origin_node_output_addr, cur_pair.second, cur_pair.first.get());
    // Update the reference count of device address.
    cur_node_output_addr->DecreaseOriginalRefCount();
    cur_node_output_addr->ResetRefCount();
    origin_node_output_addr->IncreaseOriginalRefCount();
    origin_node_output_addr->ResetRefCount();
  } else {
    MS_LOG(INFO) << "No need update device address: ref origin kernel is " << origin_pair.first->fullname_with_scope()
                 << ", index is " << origin_pair.second << ", cur kernel is " << cur_pair.first->fullname_with_scope()
                 << ", index is " << cur_pair.second;
  }
}

void UpdateDeviceAddressForRefNode(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &kernels = graph->execution_order();
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto output_num = common::AnfAlgo::GetOutputTensorNum(kernel);
    if (output_num == 0) {
      MS_LOG(DEBUG) << "This kernel has no output size.";
      continue;
    }
    for (size_t i = 0; i < output_num; ++i) {
      session::AnfWithOutIndex out_pair(kernel, i);
      if (graph->IsInRefOutputMap(out_pair)) {
        auto origin_pair = graph->GetRefCorrespondOutput(out_pair);
        UpdateDeviceAddress(out_pair, origin_pair);
      }
    }
  }
}

void SetSummaryNodesRefCount(const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->summary_node_exist()) {
    return;
  }

  const std::map<std::string, std::pair<AnfNodePtr, int>> &summary_nodes = graph->summary_nodes();
  if (summary_nodes.empty()) {
    return;
  }

  for (const auto &item : summary_nodes) {
    const AnfNodePtr &node = item.second.first;
    size_t index = IntToSize(item.second.second);
    auto device_address = AnfAlgo::GetMutableOutputAddr(node, index, false);
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->set_original_ref_count(SIZE_MAX);
    device_address->ResetRefCount();
  }
}

void SetGraphInputNodeActualAbstract(const session::BackendOpRunInfoPtr &op_run_info, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!op_run_info->base_op_run_info.has_dynamic_output && !op_run_info->base_op_run_info.has_dynamic_input) {
    return;
  }
  const auto &tensor_mask = op_run_info->base_op_run_info.input_mask;
  const auto &input_tensors = op_run_info->base_op_run_info.input_tensor;
  auto &graph_inputs = graph->inputs();
  for (size_t i = 0, j = 0; i < op_run_info->base_op_run_info.input_tensor.size() && j < graph_inputs.size(); ++i) {
    if (tensor_mask[i] == kValueNodeTensorMask) {
      continue;
    }
    if (input_tensors[i]->base_shape_ptr() != nullptr) {
      const auto &shape_of_tensor = input_tensors[i]->shape();
      auto actual_abstract = std::make_shared<abstract::AbstractTensor>(input_tensors[i]->Dtype(), shape_of_tensor);
      graph_inputs[j]->set_user_data(kActualAbstract, actual_abstract);
    }
    ++j;
  }
}
}  // namespace

GraphCompilerInfo::~GraphCompilerInfo() {
  GraphScheduler::GetInstance().Clear(name_, graphs_, origin_parameters_order_, control_node_parser_);
}

namespace {
// Fetch the real input of the nop node recursively.
AnfNodePtr FetchRealNodeByNopNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if ((!node->isa<CNode>()) || (!common::AnfAlgo::IsNopNode(node))) {
    return node;
  }

  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  const auto &inputs = cnode->inputs();
  if (inputs.size() <= 1) {
    MS_LOG(EXCEPTION) << "Invalid cnode:" << cnode->DebugString();
  }
  return FetchRealNodeByNopNode(inputs[1]);
}

// Recursively delete the nodes in the eliminate nodes list in the graph, check node records
// the nodes that have been checked during the recursive process.
void EliminateNodesFromGraph(CNode *node, const std::set<AnfNodePtr> &eliminate_nodes,
                             std::set<CNode *> *checked_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(checked_nodes);

  if (checked_nodes->find(node) != checked_nodes->end()) {
    return;
  }

  (void)checked_nodes->emplace(node);
  const auto &inputs = node->inputs();
  std::vector<AnfNodePtr> new_inputs;
  for (auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>()) {
      (void)new_inputs.emplace_back(input);
      continue;
    }

    if (eliminate_nodes.find(input) == eliminate_nodes.end()) {
      (void)new_inputs.emplace_back(input);
    } else {
      // If input is an eliminate node, replace it by its real input.
      const auto &real_input = FetchRealNodeByNopNode(input);
      MS_EXCEPTION_IF_NULL(real_input);

      // Since the output of previous node will be cached, the cache needs to be updated after eliminating the nopnode.
      auto kernel_info = node->kernel_info();
      if (kernel_info) {
        auto runtime_cache = kernel_info->runtime_cache();
        if (runtime_cache.runtime_cache().is_valid()) {
          runtime_cache.runtime_cache().update_prev_node_output(
            new_inputs.size() - 1, common::AnfAlgo::VisitKernelWithReturnType(real_input, 0));
        }
      }
      (void)new_inputs.emplace_back(real_input);
    }
    const auto &cnode = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    EliminateNodesFromGraph(cnode.get(), eliminate_nodes, checked_nodes);
  }
  node->set_inputs(new_inputs);
}

// Check whether a cnode has a monad input.
bool HasMonadInput(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (HasAbstractMonad(input)) {
      return true;
    }
  }
  return false;
}

// Collect all nopnodes which are input of kernel that not support multi-thread execute.
std::set<CNodePtr> FetchNopNodeNotSupportEliminate(const KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::set<CNodePtr> invalid_nopnodes;

  // In the implementation of some cpu operators, shape information is obtained through the input of the kernel
  // in launchkernel stage. The nopnode input of these operators cannot be eliminated.
  const std::set<std::string> kCPUOpNoEliminateList = {kConcatOpName};

  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    // If target is not cpu, the total cnode in graph can skip.
    auto target = GetCNodeTarget(cnode);
    if (target != kCPUDevice) {
      break;
    }

    // kernel not support multi-thread execute will be inited in launch kernel, so its input cannot be eliminated.
    if (IsOneOfNotSupportMultiThreadExec(common::AnfAlgo::GetCNodeName(cnode)) ||
        (kCPUOpNoEliminateList.find(common::AnfAlgo::GetCNodeName(cnode)) != kCPUOpNoEliminateList.end())) {
      const auto &inputs = cnode->inputs();
      for (const auto &input : inputs) {
        MS_EXCEPTION_IF_NULL(input);
        const auto &input_with_index = common::AnfAlgo::VisitKernelWithReturnType(input, 0);
        if ((input_with_index.first != nullptr) && (input_with_index.first->isa<CNode>()) &&
            common::AnfAlgo::IsNopNode(input_with_index.first)) {
          // Collect all of the nopnode inputs.
          (void)invalid_nopnodes.emplace(input->cast<CNodePtr>());
          MS_LOG(INFO) << "Add invalid nopnode:" << input->DebugString()
                       << " for node not support mulit-thread execute list.";
        }
      }
    }
  }
  return invalid_nopnodes;
}

void OptimizeNopNode(KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<CNodePtr> new_execution_order;
  std::vector<CNodePtr> nop_nodes_need_set_ref;
  std::set<AnfNodePtr> nop_nodes_need_eliminated;

  // Skip the graph mode or dynamic shape.
  if (graph->is_graph_run_mode()) {
    return;
  }

  // Invalid nopnode is those cannot be eliminated in some scene.
  const auto &invalid_nopnodes = FetchNopNodeNotSupportEliminate(graph);
  const auto &output_node = graph->output();
  MS_EXCEPTION_IF_NULL(output_node);
  const auto &graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(output_node);
  // Collect all the nopnodes that can be eliminated.
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    if ((!common::AnfAlgo::IsNopNode(cnode)) || graph->IsInRefOutputMap({cnode, 0}) ||
        graph->IsRefOutputMapValue({cnode, 0}) ||
        (std::find(graph_outputs.begin(), graph_outputs.end(), KernelWithIndex(cnode, 0)) != graph_outputs.end())) {
      (void)new_execution_order.emplace_back(cnode);
      continue;
    }
    // The nopnode which satisfies the following conditions cannot be eliminated and set to ref node:
    // 1.dynamic shape 2.side effect 3. must not be eliminated.
    if (graph->is_dynamic_shape() || HasMonadInput(cnode) || (invalid_nopnodes.find(cnode) != invalid_nopnodes.end())) {
      (void)new_execution_order.emplace_back(cnode);
      (void)nop_nodes_need_set_ref.emplace_back(cnode);
    } else {
      MS_LOG(DEBUG) << "Eliminate node:" << cnode->DebugString();
      (void)nop_nodes_need_eliminated.emplace(cnode);
    }
  }

  // Add the ref node pairs.
  for (auto &ref_node : nop_nodes_need_set_ref) {
    MS_EXCEPTION_IF_NULL(ref_node);
    auto input_node = common::AnfAlgo::GetInputNode(ref_node, 0);
    MS_EXCEPTION_IF_NULL(input_node);
    auto origin_pair = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
    MS_EXCEPTION_IF_NULL(origin_pair.first);
    // The device address of parameter as input may be not the running used in the heterogeneous or control flow
    // scenarios, and not set the ref node.
    if (origin_pair.first->isa<Parameter>()) {
      continue;
    }
    MS_LOG(INFO) << "The reference relation of nopnode " << ref_node->fullname_with_scope() << ", index: " << 0
                 << " to input " << origin_pair.first->fullname_with_scope() << ", index: " << origin_pair.second;
    graph->AddRefCorrespondPairs(std::make_pair(ref_node, 0), origin_pair);
  }

  std::set<CNode *> checked_nodes;
  MS_EXCEPTION_IF_NULL(graph->return_node());
  EliminateNodesFromGraph(graph->return_node().get(), nop_nodes_need_eliminated, &checked_nodes);
  graph->set_execution_order(new_execution_order);
}
}  // namespace

GraphId GraphCompiler::CompileGraph(const GraphSegmentPtr &segment, const AnfNodePtrList &outputs,
                                    const DeviceContext *device_context, device::RunMode run_mode,
                                    bool run_in_pynative) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(segment);
  MS_LOG(INFO) << "Status record: start compile graph.";
  auto nodes = segment->nodes_;
  auto device_terget = device_context->GetDeviceType();
  // Generate kernel graph.
  KernelGraphPtr graph = session_->ConstructKernelGraph(nodes, outputs, device_terget);
  MS_EXCEPTION_IF_NULL(graph);
  opt::EliminateIllegalDataTypePass(graph);
  SetGraphDependency(graph, segment);

  // Unify the MindIR, must be before of the graph optimization.
  auto deprecated_kernel_executor =
    dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
  if (deprecated_kernel_executor != nullptr) {
    deprecated_kernel_executor->UnifyMindIR(graph);
  } else {
    opt::CommonUnifyMindIR(graph);
  }

  // The graph common optimization.
  graph->UpdateGraphAquireGilAttr();
  opt::BackendCommonOptimization(graph);
  graph->SetInputNodes();
  auto manager = MakeManager({graph});
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  session_->SetInputNodeUsage(graph, manager);
  graph->SetOptimizerFlag();

  if (run_mode == device::RunMode::kUnknown) {
    graph->set_run_mode(device_context->GetRunMode(graph));
  } else {
    graph->set_run_mode(run_mode);
  }

  GraphId graph_id = 0;
  if (run_in_pynative) {
    MS_EXCEPTION_IF_NULL(session_);
    // Graph kernel does not support pynative mode now, print a warning here.
    graphkernel::GraphKernelFlags::GetInstance().CheckSupport();
    graph_id = graph->graph_id();
  } else {
    graph_id = CompileGraphImpl(graph, device_context, run_in_pynative);
  }
  session_->InitAllBucket(graph, device_context);

  graph->set_front_outputs(outputs);

  session_->DumpGraphs({graph});

  // The graph is not compiled yet in PyNative Mode.
  // Need to cache output latter when the graph is compiled.
  if (!run_in_pynative) {
    // Cache the backend graph output nodes to front nodes with output index.
    auto backend_node = graph->output();
    MS_EXCEPTION_IF_NULL(backend_node);
    graph->CacheGraphOutputToFrontNodeWithIndex({backend_node}, outputs);
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target == kGPUDevice) {
    graph->set_root_graph_id(graph_id);
  }
  AnfAlgo::UpdateGraphValidRefPair(graph);

  for (auto &node : graph->execution_order()) {
    if (common::AnfAlgo::IsControlOpExecInBackend(node)) {
      graph->set_flag(kFlagsIsCutGraph, true);
    }
  }

  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
}

GraphId GraphCompiler::CompileWholeGraphForGraphRunMode(const FuncGraphPtr &func_graph,
                                                        const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Status record: start compile graph.";
  // Generate kernel graph.
  std::vector<KernelGraphPtr> all_graphs;
  auto device_target = device_context->GetDeviceType();
  KernelGraphPtr root_graph = session_->ConstructKernelGraph(func_graph, &all_graphs, device_target);
  MS_EXCEPTION_IF_NULL(root_graph);
  for (const auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    graph->set_root_graph_id(root_graph->graph_id());
  }

  // todo: waiting for GraphExecutor
  if (MsContext::GetInstance()->backend_policy() == "ge") {
    auto manager = MakeManager();
    for (const auto &graph : all_graphs) {
      MS_EXCEPTION_IF_NULL(graph);
      manager->AddFuncGraph(graph);
      graph->set_manager(manager);
    }
    MS_EXCEPTION_IF_NULL(device_context->graph_executor_);
    if (!device_context->graph_executor_->CompileGraph(root_graph, {})) {
      MS_LOG(EXCEPTION) << "Compile graph failed: " << root_graph->graph_id();
    }
    root_graph->CacheGraphOutputToFrontNodeWithIndex({root_graph->output()}, {func_graph->output()});
    return root_graph->graph_id();
  }

  // set executing sink true in graph mode
  root_graph->set_run_mode(device::RunMode::kGraphMode);
  root_graph->set_is_loop_count_sink(true);
#ifdef WITH_BACKEND
  // Embedding cache need global step of compute graph, can not enable loop sink, move loop control to loop count actor.
  if (ps::PSContext::instance()->cache_enable()) {
    root_graph->set_is_loop_count_sink(false);
  }
#endif

  // Unify the MindIR, must be before of the graph optimization.
  auto deprecated_kernel_executor =
    dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
  if (deprecated_kernel_executor != nullptr) {
    deprecated_kernel_executor->UnifyMindIR(root_graph);
  }

  // The graph common optimization.
  opt::BackendCommonOptimization(root_graph);
  root_graph->SetInputNodes();

  auto graph_id = CompileGraphImpl(root_graph, device_context);

  // Set summary nodes for all graphs.
  session_->SetSummaryNodesForAllGraphs(root_graph.get(), all_graphs);

  // dump all graphs.
  // for ascend mindRT.
  session_->DumpGraphs(all_graphs);

  // Cache the backend graph output nodes to front nodes with output index.
  auto output = func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  auto backend_node = root_graph->output();
  MS_EXCEPTION_IF_NULL(backend_node);
  root_graph->CacheGraphOutputToFrontNodeWithIndex({backend_node}, {output});
  AnfAlgo::UpdateGraphValidRefPair(root_graph);

  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
}

GraphId GraphCompiler::CompileGraphImpl(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                        bool run_in_pynative) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(session_);

#ifdef ENABLE_DUMP_IR
  bool save_graphs = ms_context->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  // Dump .pb graph before graph optimization.
  if (save_graphs) {
    DumpIRProto(graph, "before_opt_" + std::to_string(graph->graph_id()));
  }
#endif

  // Execute optimization pass.
  device_context->kernel_executor_->OptimizeGraph(graph);

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  device_context->kernel_executor_->CreateKernel(graph->execution_order());

  // Read the output and input ref map and set to the kernel graph.
  AddOutInRefToGraph(graph);

  // Optimize the nop node.
  if (!run_in_pynative) {
    OptimizeNopNode(graph.get());
#ifdef ENABLE_DUMP_IR
    if (save_graphs) {
      DumpIR("hwopt_comm_after_eliminate_nopnode_" + graph->ToString(), graph);
    }
#endif
  }

#ifndef ENABLE_SECURITY
  session_->SetSummaryNodes(graph.get());
  // Update needed dump kernels for mindRT.
  DumpJsonParser::GetInstance().UpdateNeedDumpKernels(*graph.get());
#endif

#ifdef WITH_BACKEND
  // Set device address for embedding cache parameter, only enable when enable embedding cache mode.
  EmbeddingCacheScheduler::GetInstance().SetEmbedCachedParamAddress(device_context, graph);
#endif

  // dynamic shape pass of graphmode
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->is_dynamic_shape()) {
    opt::DynamicShapeConvertPass(graph);
  }
  auto profiler_manage_inst = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manage_inst);
  if (kernel_graph->is_dynamic_shape()) {
    profiler_manage_inst->SetNetDynamicShapeStatus();
  }

  // Adjust kernel graph before run graph.
  device_context->kernel_executor_->PreprocessBeforeRun(graph);

  // Create device address for all anf nodes of graph.
  CreateDeviceAddress(graph, device_context);

  SetSummaryNodesRefCount(graph.get());
#ifdef ENABLE_DUMP_IR
  // Dump .pb graph after graph optimization.
  if (save_graphs) {
    DumpIRProto(graph, "after_opt_" + std::to_string(graph->graph_id()));
  }
#endif

#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  // Dump graph for GPU mindRT if dump is enabled.
  debugger->DumpInGraphCompiler(graph);
  if (debugger && debugger->DebuggerBackendEnabled()) {
    // Load graphs for GPU and Ascend mindRT.
    debugger->LoadGraphs(graph);
  }
#endif

  graph->EnableRuntimeCache();
  return graph->graph_id();
}

GraphId GraphCompiler::CompileGraph(const session::BackendOpRunInfoPtr &op_run_info, bool *single_op_cache_hit,
                                    const DeviceContext *device_context) {
  // Check if the graph cache exists.
  auto iter = run_op_graphs_.find(op_run_info->base_op_run_info.graph_info);
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (iter != run_op_graphs_.end() && op_executor.BuildQueueEmpty()) {
    const auto &graph = iter->second;
    MS_EXCEPTION_IF_NULL(graph);
    SetGraphInputNodeActualAbstract(op_run_info, graph);
    *single_op_cache_hit = true;
    return graph->graph_id();
  }
  *single_op_cache_hit = false;
  // Generate kernel graph.
  MS_EXCEPTION_IF_NULL(session_);
  KernelGraphPtr graph = session_->ConstructSingleOpGraph(
    op_run_info, op_run_info->base_op_run_info.input_tensor, op_run_info->base_op_run_info.input_mask,
    device_context->GetDeviceType() == device::DeviceType::kAscend);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);

  graph->set_run_mode(device::RunMode::kKernelMode);
  graph->set_is_from_single_op(true);
  // session_ is SessionBasic, AscendUnifyMindIR has not been executed.
  auto deprecated_kernel_executor =
    dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
  if (deprecated_kernel_executor != nullptr) {
    deprecated_kernel_executor->UnifyMindIR(graph);
  } else {
    opt::CommonUnifyMindIR(graph);
  }

  // Select kernel and optimize
  device_context->kernel_executor_->OptimizeGraph(graph);

  UpdateRefInfoBeforeCreateKernel(op_run_info, graph);

  // Set dynamic shape actual abstract
  SetGraphInputNodeActualAbstract(op_run_info, graph);

  // Create device address for all anf nodes of graph.
  CreateDeviceAddressWithoutWorkspace(graph, device_context, op_run_info->is_gradient_out);

  run_op_graphs_[op_run_info->base_op_run_info.graph_info] = graph;

  auto output_nodes = graph->outputs();
  auto &outputs_with_index = run_op_graph_output_nodes_[graph->graph_id()];
  for (auto &node : output_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    (void)outputs_with_index.emplace_back(common::AnfAlgo::VisitKernelWithReturnType(node, 0, false));
  }

  AnfAlgo::UpdateGraphValidRefPair(graph);
  return graph->graph_id();
}

void GraphCompiler::UpdateRefInfoBeforeCreateKernel(const session::BackendOpRunInfoPtr &op_run_info,
                                                    const KernelGraphPtr &graph) const {
  // Building Graph and Create Kernel is async, under pynative mode.Ref info is bind with kernel.
  // So need to get ref info to generate output addr, before create kernel.
  if (op_run_info->base_op_run_info.device_target != kCPUDevice &&
      op_run_info->base_op_run_info.device_target != kGPUDevice) {
    // just ascend ref mode is diff with cpu and gpu
    return;
  }

  AddOutInRefToGraph(graph);
}

void GraphCompiler::BuildSingleOpGraphs(const std::vector<KernelGraphPtr> &graphs,
                                        const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(device_context);
  std::vector<CNodePtr> node_to_build;
  for (const auto &graph : graphs) {
    const auto &nodes = graph->execution_order();
    (void)std::copy(nodes.begin(), nodes.end(), std::back_inserter(node_to_build));
  }
  // Kernel build
  device_context->kernel_executor_->CreateKernel(node_to_build);

  for (const auto &graph : graphs) {
    device_context->kernel_executor_->PreprocessBeforeRun(graph);
    CreateKernelWorkspaceDeviceAddress(device_context, graph);
    // Need to execute after PreprocessBeforeRunSingleOpGraph
    runtime::OpRuntimeInfo::CacheGraphOpRuntimeInfo(graph);
  }
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

void GraphCompiler::AddOutInRefToGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    auto kernel_info = dynamic_cast<device::KernelInfo *>(cnode->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    for (const auto &ref : kernel_info->out_in_ref_map()) {
      size_t output_index = ref.first;
      size_t input_index = ref.second;
      auto final_pair = std::make_pair(cnode, output_index);
      auto origin_pair = common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(cnode, input_index), 0);
      MS_LOG(INFO) << "The reference relation output " << final_pair.first->fullname_with_scope()
                   << ", output index: " << final_pair.second << " to input "
                   << origin_pair.first->fullname_with_scope() << ", output index: " << origin_pair.second;
      // Add to graph only if the input is not a monad.
      if (!HasAbstractUMonad(origin_pair.first) && !HasAbstractIOMonad(origin_pair.first)) {
        graph->AddRefCorrespondPairs(final_pair, origin_pair);
      }
    }
  }
}

void GraphCompiler::CreateDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) const {
  MS_LOG(INFO) << "Status record: start create device address. graph id: " << graph->graph_id();
  CreateParameterDeviceAddress(device_context, graph);
  CreateValueNodeDeviceAddress(device_context, graph);
  CreateKernelOutputDeviceAddress(device_context, graph, false);
  CreateKernelWorkspaceDeviceAddress(device_context, graph);
  UpdateDeviceAddressForInplaceNode(graph);
  UpdateDeviceAddressForRefNode(graph);

  MS_LOG(INFO) << "Status record: end create device address. graph id: " << graph->graph_id();
}

void GraphCompiler::CreateDeviceAddressWithoutWorkspace(const KernelGraphPtr &graph,
                                                        const DeviceContext *device_context,
                                                        bool is_gradient_out) const {
  CreateParameterDeviceAddress(device_context, graph);
  CreateValueNodeDeviceAddress(device_context, graph);
  CreateKernelOutputDeviceAddress(device_context, graph, is_gradient_out);
  UpdateDeviceAddressForInplaceNode(graph);
  UpdateDeviceAddressForRefNode(graph);
}

void GraphCompiler::GetParamAndOutputIndex(
  const KernelGraphPtr &graph, const std::vector<TensorPtr> &inputs, VectorRef *const outputs,
  std::map<AnfNodePtr, size_t> *parameter_index,
  std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->GetParameterIndex(graph.get(), inputs, parameter_index);
  session_->CreateOutputPlaceholder(graph, inputs, outputs, output_indexes);
}

void GraphCompiler::GetSingleOpInputTensors(const CNodePtr &kernel,
                                            const std::map<KernelWithIndex, TensorPtr> &op_output,
                                            const std::map<AnfNodePtr, size_t> &parameter_index,
                                            const std::vector<TensorPtr> &graph_inputs,
                                            InputTensorInfo *const input_tensor_info) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->GetOpInputTensors(kernel, op_output, parameter_index, graph_inputs, input_tensor_info);
}

TensorPtr GraphCompiler::GetSingleOpInputTensorByIndex(const CNodePtr &kernel,
                                                       const std::map<KernelWithIndex, TensorPtr> &op_output,
                                                       const std::map<AnfNodePtr, size_t> &parameter_index,
                                                       const std::vector<TensorPtr> &graph_inputs,
                                                       InputTensorInfo *const input_tensor_info, size_t input_index) {
  MS_EXCEPTION_IF_NULL(session_);
  return session_->GetOpInputTensorByIndex(kernel, op_output, parameter_index, graph_inputs, input_tensor_info,
                                           input_index);
}

void GraphCompiler::GetSingleOpRunInfoAndGraphInfo(const CNodePtr &kernel, const InputTensorInfo &tensor_info,
                                                   session::BackendOpRunInfoPtr *op_run_info, GraphInfo *graph_info,
                                                   GraphOutputInfo *const graph_output_info) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(graph_info);
  session_->GetSingleOpGraphInfo(kernel, tensor_info, graph_info);
  *op_run_info = session_->GetSingleOpRunInfo(kernel, *graph_info, tensor_info, graph_output_info);
}

void GraphCompiler::CalculateRefCount(const KernelGraphPtr &graph, std::map<KernelWithIndex, size_t> *ref_count) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->GetRefCount(graph.get(), ref_count);
}

void GraphCompiler::CalculateForwardOpOutputCount(const KernelGraphPtr &graph,
                                                  const std::vector<tensor::TensorPtr> &inputs,
                                                  std::map<std::string, size_t> *forward_op_output_tensor_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  forward_op_output_tensor_id->clear();
  session_->GetForwardOpOutputRefCount(graph.get(), inputs, forward_op_output_tensor_id);
}

void GraphCompiler::UpdateRefCount(const std::set<KernelWithIndex> &input_kernels_with_index,
                                   std::map<KernelWithIndex, size_t> *ref_count,
                                   std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->HandleOpInputs(input_kernels_with_index, ref_count, op_output_map);
}

void GraphCompiler::UpdateForwardOpOutputRefCount(const std::vector<tensor::TensorPtr> &input_tensor,
                                                  std::map<std::string, size_t> *forward_op_output_tensor_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(forward_op_output_tensor_id);
  session_->ReleaseForwardOpOutput(input_tensor, forward_op_output_tensor_id);
}

void GraphCompiler::RecoverGraphOutput(const AnfNodePtr &kernel, const VectorRef &op_outputs,
                                       const std::map<KernelWithIndex, size_t> &ref_count,
                                       std::map<KernelWithIndex, TensorPtr> *op_output_map,
                                       GraphOutputInfo *const graph_output_info) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->HandleOpOutputs(kernel, op_outputs, ref_count, op_output_map, graph_output_info);
}

void GraphCompiler::DoAllReduceOnGrads(const std::string &actor_info, const std::vector<tensor::TensorPtr> &outputs,
                                       const device::DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->DoAllReduceOnGrads(actor_info, outputs, device_context);
}

void GraphCompiler::AddGradAddrToBucket(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &grad_tensor) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->AddGradAddrToBucket(graph_id, grad_tensor);
}

void GraphCompiler::ClearAllBucket(const GraphId &graph_id) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->ClearAllBucket(graph_id);
}

const std::vector<KernelWithIndex> &GraphCompiler::GetGraphOutputNodes(GraphId graph_id) const {
  const auto &iter = run_op_graph_output_nodes_.find(graph_id);
  if (iter == run_op_graph_output_nodes_.end()) {
    MS_LOG(EXCEPTION) << "Can not find output nodes for graph id: " << graph_id;
  }
  return iter->second;
}

void GraphCompiler::RegisterSummaryCallBackFunc(const CallBackFunc &callback) const {
  MS_EXCEPTION_IF_NULL(session_);
#ifndef ENABLE_SECURITY
  session_->RegisterSummaryCallBackFunc(callback);
#endif
}

void GraphCompiler::Summary(const std::vector<KernelGraphPtr> &graphs) const {
  MS_EXCEPTION_IF_NULL(session_);
  for (const auto &graph : graphs) {
#ifndef ENABLE_SECURITY
    session_->Summary(graph.get());
#endif
  }
}

void GraphCompiler::EraseSingleOpCache(const GraphInfo &graph_info, const GraphId &graph_id) {
  (void)run_op_graphs_.erase(graph_info);
  (void)run_op_graph_output_nodes_.erase(graph_id);
}

void GraphCompiler::SetGraphDependency(const KernelGraphPtr &graph, const GraphSegmentPtr &segment) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(segment);
  segment->graph_id_ = graph->graph_id();
  for (auto &pre_segment : segment->pre_segments_) {
    MS_EXCEPTION_IF_NULL(pre_segment);
    auto pre_graph = Fetch(pre_segment->graph_id_);
    MS_EXCEPTION_IF_NULL(pre_graph);
    pre_graph->AddPostGraph(graph);
    graph->AddPreGraph(pre_graph);
    MS_LOG(INFO) << "Link graph " << pre_segment->graph_id_ << " to " << graph->graph_id();
  }
}
}  // namespace runtime
}  // namespace mindspore
