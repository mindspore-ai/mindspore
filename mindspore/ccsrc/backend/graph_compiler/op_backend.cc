/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/graph_compiler/op_backend.h"

#include <string>
#include <vector>
#include <algorithm>
#include "ops/structure_op_name.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_runner.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/pipeline/pipeline.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/backend/mem_reuse/mem_tracker.h"

namespace mindspore::compile {
namespace {
#if !defined(__APPLE__)
bool EnablePyNativeSyncRunning() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
}
#endif

bool DisableRunOpAsync(const OpCompilerInfoPtr &op_compiler_info, const session::BackendOpRunInfoPtr &op_run_info) {
#if defined(__APPLE__)
  return true;
#else
  return op_run_info->base_op_run_info.has_dynamic_output ||  // Infer output is dynamic.
         op_compiler_info->need_refresh_abstract_ ||          // Graph output is dynamic after IR Pass. (e.g. Dropout)
         op_compiler_info->need_erase_ ||                     // Random op cache need to be erased.
         runtime::OpExecutor::NeedSync() ||                   // Cannot find a wait point before compile graph.
         EnablePyNativeSyncRunning();                         // context.set_context(pynative_synchronize=True)
#endif
}

void WaitBackendQueue() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                     runtime::kDefaultOpName);
  GilReleaseWithCheck gil_release;
  runtime::Pipeline::Get().backend_stage()->Wait();
}

}  // namespace

void OpBackend::Run(const BackendOpRunInfoPtr &op_run_info, const std::string &device_name, uint32_t device_id,
                    VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (op_run_info->base_op_run_info.use_dynamic_shape_process) {
    RunInnerDynamic(op_run_info, device_name, device_id, outputs);
  } else {
    RunInner(op_run_info, device_name, device_id, outputs);
  }
}

void OpBackend::RunInner(const BackendOpRunInfoPtr &op_run_info, const std::string &device_name, uint32_t device_id,
                         VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "RunOp name " << op_run_info->base_op_run_info.op_name << " device " << device_name << " id "
                << device_id << " with static shape";

  bool single_op_cache_hit = true;
  auto op_compiler_info =
    pynative::OpCompiler::GetInstance().Compile(op_run_info, &single_op_cache_hit, device_name, device_id);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  op_compiler_info->WaitReady();
  RunOpImpl(single_op_cache_hit, op_compiler_info, op_run_info, outputs);
}

void OpBackend::RunInnerDynamic(const BackendOpRunInfoPtr &op_run_info, const std::string &device_name,
                                uint32_t device_id, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "RunOp name " << op_run_info->base_op_run_info.op_name << " device " << device_name << " id "
                << device_id << " with dynamic shape";

  // Single op graph compile
  bool single_op_cache_hit = true;
  auto op_compiler_info =
    pynative::OpCompiler::GetInstance().Compile(op_run_info, &single_op_cache_hit, device_name, device_id);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  op_compiler_info->WaitReady();
  RunOpImplDynamic(single_op_cache_hit, op_compiler_info, op_run_info, outputs);
}

void OpBackend::RunOpImpl(bool single_op_cache_hit, const OpCompilerInfoPtr &op_compiler_info,
                          const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  // Fetch outputs.
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  const auto &output_nodes = op_compiler_info->graph_output_nodes_;
  MS_EXCEPTION_IF_NULL(outputs);

  auto device_context = op_compiler_info->device_context_;
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (!DisableRunOpAsync(op_compiler_info, op_run_info)) {
    MS_LOG(DEBUG) << "Async exec enabled, op: " << op_run_info->base_op_run_info.op_name;
    DispatchOpTask(single_op_cache_hit, outputs, op_compiler_info, op_run_info);
    return;
  }

  MS_LOG(DEBUG) << "Async exec disabled, op: " << op_run_info->base_op_run_info.op_name;
  if (!op_executor.RunQueueEmpty()) {
    WaitBackendQueue();
  }
  if (!single_op_cache_hit) {
    pynative::OpCompiler::GetInstance().KernelBuild(op_compiler_info, device_context, false);
  }
  const auto &tensors_without_value_mask = runtime::OpRunner::GetTensorWithoutValueMask(op_run_info);
  runtime::OpRunner::UpdateDeviceAddress(graph, tensors_without_value_mask, device_context, true);

  runtime::OpRunner::RunSingleOpGraph(op_run_info, op_compiler_info, tensors_without_value_mask);

  if (!op_run_info->is_infer) {
    post_run_.ReleaseForwardOpOutput(op_run_info->base_op_run_info.expanded_input_values);
  }
  post_run_.UpdateOutput(output_nodes, outputs);

  post_run_.ClearGraphDeviceAddress(graph, device_context, op_run_info->is_gradient_out);
  post_run_.ClearInputDeviceAddress(graph, device_context);
  post_run_.ClearOpInputOutput(op_compiler_info);

  if (op_run_info->base_op_run_info.has_dynamic_output || op_compiler_info->need_refresh_abstract_) {
    post_run_.UpdateOutputAbstract(*outputs, op_run_info);
  }
  if (op_compiler_info->need_erase_) {
    pynative::OpCompiler::GetInstance().ClearOpCache(op_compiler_info->graph_info_);
  }
}

void OpBackend::RunOpImplDynamic(bool single_op_cache_hit, const OpCompilerInfoPtr &op_compiler_info,
                                 const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  MS_LOG(DEBUG) << "RunOpImplDynamic " << op_run_info->base_op_run_info.op_name;
  // Fetch outputs.
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(outputs);

  auto device_context = op_compiler_info->device_context_;
  if (!single_op_cache_hit) {
    pynative::OpCompiler::GetInstance().KernelBuild(op_compiler_info, device_context, true);
  }
  if (!DisableRunOpAsync(op_compiler_info, op_run_info)) {
    MS_LOG(DEBUG) << "Async exec enabled, op: " << op_run_info->base_op_run_info.op_name;
    auto input_tensors = runtime::OpRunner::GetTensorWithoutValueMask(op_run_info);
    runtime::DynamicOpRunner::UpdateInputDeviceAddress(op_compiler_info, input_tensors, false);
    auto device_address_list = runtime::DeviceAddressUtils::CreateGraphOutputDeviceAddress(
      op_compiler_info, op_run_info->base_op_run_info.abstract, op_run_info->base_op_run_info.stream_id);
    // Create output tensor
    post_run_.UpdateOutputDynamic(op_run_info, op_compiler_info, device_address_list, outputs);
    DispatchOpTaskDynamic(outputs, op_compiler_info, op_run_info, device_address_list);
    return;
  }
  MS_LOG(DEBUG) << "Async exec disabled, op: " << op_run_info->base_op_run_info.op_name;
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (!op_executor.RunQueueEmpty()) {
    WaitBackendQueue();
  }
  auto input_tensors = runtime::OpRunner::GetTensorWithoutValueMask(op_run_info);
  runtime::DynamicOpRunner::UpdateInputDeviceAddress(op_compiler_info, input_tensors, true);
  runtime::DynamicOpRunner::RunSingleOpGraph(op_run_info, op_compiler_info, input_tensors);

  if (!op_run_info->is_infer) {
    post_run_.ReleaseForwardOpOutput(op_run_info->base_op_run_info.expanded_input_values);
  }

  const auto &device_address_list = GetOutputDeviceAddress(op_compiler_info);
  // Create output tensor
  post_run_.UpdateOutputDynamic(op_run_info, op_compiler_info, device_address_list, outputs);
  post_run_.UpdateOutputAbstract(*outputs, op_run_info);
  post_run_.ClearOpInputOutput(op_compiler_info);
  if (op_compiler_info->need_erase_) {
    pynative::OpCompiler::GetInstance().ClearOpCache(op_compiler_info->graph_info_);
  }
}

void OpBackend::DispatchOpTask(bool single_op_cache_hit, VectorRef *outputs, const OpCompilerInfoPtr &op_compiler_info,
                               const session::BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);

  runtime::OpRunner::UpdateDeviceAddress(graph, runtime::OpRunner::GetTensorWithoutValueMask(op_run_info),
                                         op_compiler_info->device_context_, false);
  // Create output tensor
  post_run_.UpdateOutput(op_compiler_info->graph_output_nodes_, outputs);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  auto run_op_context =
    std::make_shared<runtime::OpTaskContext>(graph->graph_id(), graph, op_run_info, op_compiler_info, infer_flag);

  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (!single_op_cache_hit) {
    pynative::OpCompiler::GetInstance().KernelBuild(op_compiler_info, op_compiler_info->device_context_, false);
  }

  auto run_task = std::make_shared<runtime::DeviceOpRunTask>(
    run_op_context, [this](const std::shared_ptr<runtime::OpTaskContext> &ctx) { OpRunCallback(ctx); });
  runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(run_task->task_id());
  op_executor.PushOpRunTask(run_task);
}

void OpBackend::OpRunCallback(const std::shared_ptr<runtime::OpTaskContext> &context) {
  MS_LOG(DEBUG) << "OpRunCallback start";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, context->is_pynative_infer());
  MS_EXCEPTION_IF_NULL(context);
  runtime::OpRunner::RunSingleOpGraph(context->op_run_info(), context->op_compiler_info(),
                                      runtime::OpRunner::GetTensorWithoutValueMask(context->op_run_info()));

  MS_EXCEPTION_IF_NULL(context->op_run_info());
  if (!context->op_run_info()->is_infer) {
    post_run_.ReleaseForwardOpOutput(context->op_run_info()->base_op_run_info.expanded_input_values);
  }

  post_run_.ClearGraphDeviceAddress(context->graph(), context->device_context(),
                                    context->op_run_info()->is_gradient_out);
  post_run_.ClearInputDeviceAddress(context->graph(), context->device_context());
  post_run_.ClearOpInputOutput(context->op_compiler_info());

  // Reset PyNative infer flag.
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, infer_flag);
  MS_LOG(DEBUG) << "OpRunCallback end";
}

void OpBackend::DispatchOpTaskDynamic(VectorRef *outputs, const OpCompilerInfoPtr &op_compiler_info,
                                      const session::BackendOpRunInfoPtr &op_run_info,
                                      const vector<device::DeviceAddressPtr> &device_address_list) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  auto run_op_context =
    std::make_shared<runtime::OpTaskContext>(graph->graph_id(), graph, op_run_info, op_compiler_info, infer_flag);

  auto &op_executor = runtime::OpExecutor::GetInstance();
  auto task = std::make_shared<runtime::DeviceOpRunTask>(
    run_op_context, [this](const std::shared_ptr<runtime::OpTaskContext> &ctx) { OpRunCallbackDynamic(ctx); });
  runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(task->task_id());
  op_executor.PushOpRunTask(task);
}

void OpBackend::OpRunCallbackDynamic(const std::shared_ptr<runtime::OpTaskContext> &context) {
  MS_LOG(DEBUG) << "OpRunCallback start";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, context->is_pynative_infer());

  MS_EXCEPTION_IF_NULL(context);
  runtime::DynamicOpRunner::RunSingleOpGraph(context->op_run_info(), context->op_compiler_info(),
                                             runtime::OpRunner::GetTensorWithoutValueMask(context->op_run_info()));

  MS_EXCEPTION_IF_NULL(context->op_run_info());
  if (!context->op_run_info()->is_infer) {
    post_run_.ReleaseForwardOpOutput(context->op_run_info()->base_op_run_info.expanded_input_values);
  }

  post_run_.ClearOpInputOutput(context->op_compiler_info());
  // Reset PyNative infer flag.
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, infer_flag);
  MS_LOG(DEBUG) << "OpRunCallback end";
}

device::DeviceAddressPtrList OpBackend::GetOutputDeviceAddress(const OpCompilerInfoPtr &op_compiler_info) const {
  const auto &output_edges = op_compiler_info->simple_graph_->outputs_;
  device::DeviceAddressPtrList output_address;
  output_address.reserve(output_edges.size());
  std::transform(output_edges.begin(), output_edges.end(), std::back_inserter(output_address),
                 [](const pynative::EdgePtr &edge) { return edge->address_; });
  return output_address;
}

void OpBackend::RunViewKernelTask(const pynative::BaseOpRunInfo &base_op_run_info,
                                  const runtime::KernelTaskType &task_type, bool enable_async) const {
  view_backend_.RunViewKernelTask(base_op_run_info, task_type, enable_async);
}

void OpBackend::RunAllocMemTask(DeviceContext *device_context, const tensor::BaseTensorPtr &tensor, bool enable_async,
                                bool is_cpu_address_exist) const {
  view_backend_.RunAllocMemTask(device_context, tensor, enable_async, is_cpu_address_exist);
}

void PostRunOp::UpdateOutput(const std::vector<session::KernelWithIndex> &output_nodes, VectorRef *outputs) const {
  MS_EXCEPTION_IF_NULL(outputs);

  for (auto &item_with_index : output_nodes) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (AnfAlgo::GetOutputTensorNum(item_with_index.first) == 0) {
      continue;
    }
    auto output_tensor = CreateOutputTensor(item_with_index.first, item_with_index.second);
    MS_EXCEPTION_IF_NULL(output_tensor);
    output_tensor->set_need_pipeline_sync(true);
    outputs->emplace_back(output_tensor);
  }
}

tensor::BaseTensorPtr PostRunOp::CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index) const {
  MS_EXCEPTION_IF_NULL(output_node);
  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(output_node, output_index, false);
  MS_EXCEPTION_IF_NULL(device_tensor);

  const auto &user_data = device_tensor->user_data();
  bool is_map_tensor_output = user_data && user_data->get<UserDataType>(kUserDataType) &&
                              *(user_data->get<UserDataType>(kUserDataType)) == UserDataType::kUserTypeHashTable;
  if (is_map_tensor_output) {
    return AnfAlgo::CreateMapTensor(output_node, output_index);
  }

  device_tensor->SetNodeIndex(output_node, output_index);
  device_tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));
  runtime::DeviceAddressUtils::UpdateDeviceAddressHostInfoByNode(device_tensor, output_node, output_index);

  const auto &kernel_tensor = device_tensor->kernel_tensor();
  MS_EXCEPTION_IF_NULL(kernel_tensor);

  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto tensor = std::make_shared<tensor::BaseTensor>(kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector());

  // Put device tensor into host tensor.
  tensor->set_device_address(device_tensor);
  tensor->set_sync_status(kNeedSyncDeviceToHost);

  // MindRT is disabled in the multi graphs scenario
  // Delete tensor->data_sync() when MindRT is enabled in all scenes.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      !runtime::OpExecutor::GetInstance().async_for_graph()) {
    // If execution mode is Graph Mode in MsContext, the tensor will be the input of graph which will execute in Graph
    // Mode, if the graph contain no CNode after optimization, the tensor need sync to host.
    tensor->data_sync(false);
  }

  return tensor;
}

void PostRunOp::ReleaseForwardOpOutput(const std::vector<ValuePtr> &input_values) {
  for (const auto &value : input_values) {
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    if (tensor == nullptr) {
      continue;
    }

    if (!tensor->is_forward_output()) {
      continue;
    }
    auto it = forward_tensor_ref_count_.find(tensor->id());
    if (it != forward_tensor_ref_count_.end()) {
      if (--(it->second) == 0) {
        MS_LOG(DEBUG) << "Release DeviceAddress on tensor " << tensor->ToString() << " id " << tensor->id()
                      << " forward_output " << tensor->is_forward_output();
        tensor->set_device_address(nullptr);
        forward_tensor_ref_count_.erase(it);
      }
    }
  }
}

void PostRunOp::ClearGraphDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                        bool is_gradient_out) const {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &node : graph->execution_order()) {
    auto output_address_num = AnfAlgo::GetOutputAddressNum(node);
    // Clear old output device address of kernel
    for (size_t i = 0; i < output_address_num; ++i) {
      if (!AnfAlgo::OutputAddrExist(node, i, false)) {
        continue;
      }
      const auto &device_address = AnfAlgo::GetMutableOutputAddr(node, i, false);
      if (device_address == nullptr) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(device_context);
      auto new_device_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(device_address, device_context);
      if (is_gradient_out) {
        new_device_address->set_from_persistent_mem(true);
      }
      AnfAlgo::SetOutputAddr(new_device_address, i, node.get());
    }

    // Clear old workspace device address of kernel
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_lists.size(); ++i) {
      if (!AnfAlgo::WorkspaceAddrExist(node, i)) {
        continue;
      }
      const auto &device_address = AnfAlgo::GetMutableWorkspaceAddr(node, i);
      auto new_device_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(device_address, device_context);
      AnfAlgo::SetWorkspaceAddr(new_device_address, i, node.get());
    }
  }
}

void PostRunOp::ClearInputDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  for (const auto &node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<Parameter>()) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(node, 0, false);
      if (device_address == nullptr) {
        continue;
      }
      auto new_device_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(device_address, device_context);
      AnfAlgo::SetOutputAddr(new_device_address, 0, node.get());
    }
  }
}

void PostRunOp::ClearOpInputOutput(const OpCompilerInfoPtr &op_compiler_info) const {
  const auto &all_edges = op_compiler_info->simple_graph_->all_edges_;
  for (const auto &edge : all_edges) {
    if (edge->type_ != pynative::EdgeType::kValueNodeEdge) {
      // Just set edge address to null rather than clone empty address.
      // Clone empty address in next RunOp if needed.
      edge->address_ = nullptr;
    }
  }
}

void PostRunOp::UpdateOutputAbstract(const VectorRef &outputs, const session::BackendOpRunInfoPtr &op_run_info) const {
  auto output_size = outputs.size();
  if (output_size == 1 && op_run_info->base_op_run_info.op_name != kGetNextOpName) {
    auto output_tensor = utils::cast<tensor::BaseTensorPtr>(outputs[0]);
    MS_EXCEPTION_IF_NULL(output_tensor);
    op_run_info->base_op_run_info.abstract = output_tensor->ToAbstract();
    MS_LOG(DEBUG) << "Update output abstract of " << op_run_info->base_op_run_info.op_name << " to "
                  << op_run_info->base_op_run_info.abstract->ToString();
    return;
  }
  AbstractBasePtrList elements;
  for (size_t i = 0; i < output_size; ++i) {
    auto output_tensor = utils::cast<tensor::BaseTensorPtr>(outputs[i]);
    MS_EXCEPTION_IF_NULL(output_tensor);
    (void)elements.emplace_back(output_tensor->ToAbstract());
  }
  op_run_info->base_op_run_info.abstract = std::make_shared<abstract::AbstractTuple>(elements);
  MS_LOG(DEBUG) << "Update output abstract of " << op_run_info->base_op_run_info.op_name << " to "
                << op_run_info->base_op_run_info.abstract->ToString();
}

void PostRunOp::UpdateOutputDynamic(const session::BackendOpRunInfoPtr &op_run_info,
                                    const OpCompilerInfoPtr &op_compiler_info,
                                    const vector<device::DeviceAddressPtr> &device_address_list,
                                    VectorRef *outputs) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "No promise, just create tensor and address, op " << op_run_info->base_op_run_info.op_name;
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  auto output_nodes = op_compiler_info->graph_output_nodes_;
  auto outputs_size = output_nodes.size();
  if (op_compiler_info->graph_outputs_tensor_num_.size() != outputs_size) {
    MS_LOG(EXCEPTION) << "The size of graph_outputs_tensor_num_:" << op_compiler_info->graph_outputs_tensor_num_.size()
                      << " is not equal to outputs_size:" << outputs_size;
  }

  if (device_address_list.size() != outputs_size) {
    MS_LOG(EXCEPTION) << "The size of device_address_list:" << device_address_list.size()
                      << " is not equal to outputs_size:" << outputs_size;
  }

  for (size_t i = 0; i < outputs_size; ++i) {
    auto item_with_index = output_nodes[i];
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (op_compiler_info->graph_outputs_tensor_num_[i] == 0) {
      continue;
    }
    auto output_address = device_address_list[i];
    MS_EXCEPTION_IF_NULL(output_address);
    auto output_tensor =
      CreateOutputTensorDynamicImpl(op_compiler_info, item_with_index.first, item_with_index.second, output_address, i);
    MS_EXCEPTION_IF_NULL(output_tensor);
    output_tensor->set_need_pipeline_sync(true);
    outputs->emplace_back(output_tensor);
  }
}

tensor::BaseTensorPtr PostRunOp::CreateOutputTensorDynamicImpl(const OpCompilerInfoPtr &op_compiler_info,
                                                               const AnfNodePtr &output_node, size_t output_index,
                                                               const std::shared_ptr<device::DeviceAddress> &address,
                                                               size_t idx_in_graph_outputs) const {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(op_compiler_info);

  const auto &user_data = address->user_data();
  bool is_map_tensor_output = user_data && user_data->get<UserDataType>(kUserDataType) &&
                              *(user_data->get<UserDataType>(kUserDataType)) == UserDataType::kUserTypeHashTable;
  if (is_map_tensor_output) {
    return AnfAlgo::CreateMapTensor(address);
  }

  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto tensor = std::make_shared<tensor::BaseTensor>(address->type_id(), address->host_shape());

  // Put device tensor into host tensor.
  address->SetNodeIndex(output_node, output_index);
  address->set_padding_type(op_compiler_info->graph_outputs_padding_type_[idx_in_graph_outputs]);
  tensor->set_device_address(address);

  // MindRT is disabled in the multi graphs scenario
  // Delete tensor->data_sync() when MindRT is enabled in all scenes.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      !runtime::OpExecutor::GetInstance().async_for_graph()) {
    // If execution mode is Graph Mode in MsContext, the tensor will be the input of graph which will execute in Graph
    // Mode, if the graph contain no CNode after optimization, the tensor need sync to host.
    tensor->data_sync(false);
  }
  return tensor;
}

void ViewBackend::RunViewKernelTask(const pynative::BaseOpRunInfo &base_op_run_info,
                                    const runtime::KernelTaskType &task_type, bool enable_async) const {
  device::DeviceAddressPtrList input_addr_list;
  device::DeviceAddressPtrList output_addr_list;

  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {base_op_run_info.device_target, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);

  for (size_t idx = 0; idx < base_op_run_info.expanded_input_values.size(); idx++) {
    auto input_tensor = base_op_run_info.expanded_input_values[idx]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(input_tensor);
    if (input_tensor->device_address() == nullptr) {
      if (idx == 0) {
        MS_LOG(EXCEPTION) << "First tensor can not be nullptr, op name:" << base_op_run_info.op_name;
      }
      auto address_size = GetTypeByte(TypeIdToType(input_tensor->data_type())) * SizeOf(input_tensor->shape());

      auto kernel_tensor = std::make_shared<kernel::KernelTensor>(
        nullptr, address_size, Format::DEFAULT_FORMAT, input_tensor->data_type(), input_tensor->shape(),
        device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
      kernel_tensor->SetType(std::make_shared<TensorType>(input_tensor->Dtype()));
      kernel_tensor->SetShape(std::make_shared<abstract::TensorShape>(input_tensor->shape()));
      kernel_tensor->set_stream_id(base_op_run_info.stream_id);
      auto input_addr = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);

      input_tensor->set_device_address(input_addr);
      RunAllocMemTask(device_context, input_tensor, enable_async, false);
      (void)input_addr_list.emplace_back(input_addr);
    } else {
      auto input_addr = std::static_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
      MS_EXCEPTION_IF_NULL(input_addr);
      if (input_addr->GetDeviceType() == device::DeviceType::kCPU) {
        RunAllocMemTask(device_context, input_tensor, enable_async, true);
      }

      (void)input_addr_list.emplace_back(input_addr);
    }
  }

  std::transform(base_op_run_info.output_tensors.begin(), base_op_run_info.output_tensors.end(),
                 std::back_inserter(output_addr_list), [](const auto &tensor) {
                   return std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
                 });

  if (enable_async) {
    RunViewKernelTaskAsyncImpl(task_type, device_context, input_addr_list, output_addr_list,
                               base_op_run_info.stream_id);
  } else {
    WaitBackendQueue();
    runtime::OpRunner::LaunchKernelTask(task_type, device_context, input_addr_list, output_addr_list,
                                        base_op_run_info.stream_id);
  }
}

void ViewBackend::RunAllocMemTask(DeviceContext *device_context, const tensor::BaseTensorPtr &tensor, bool enable_async,
                                  bool is_cpu_address_exist) const {
  if (!enable_async) {
    WaitBackendQueue();
    return AllocateMemForTensor(tensor, device_context, is_cpu_address_exist);
  }
  auto alloc_mem_func = [this, device_context, tensor, is_cpu_address_exist]() {
    AllocateMemForTensor(tensor, device_context, is_cpu_address_exist);
  };
  runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
    std::make_shared<runtime::PassthroughDeviceTask>(alloc_mem_func));
}

void ViewBackend::RunViewKernelTaskAsyncImpl(const runtime::KernelTaskType &task_type, DeviceContext *device_context,
                                             const device::DeviceAddressPtrList &input_addr_list,
                                             const device::DeviceAddressPtrList &output_addr_list,
                                             const size_t &stream_id) const {
  static auto kernel_task_func = [stream_id, task_type, &input_addr_list, &output_addr_list, device_context]() {
    runtime::OpRunner::LaunchKernelTask(task_type, device_context, input_addr_list, output_addr_list, stream_id);
  };

  runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
    std::make_shared<runtime::PassthroughDeviceTask>(kernel_task_func));
}

void ViewBackend::AllocateMemForTensor(const tensor::BaseTensorPtr &tensor, DeviceContext *device_context,
                                       bool is_cpu_address_exist) const {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context);

  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  device_address->set_is_view(true);
  if (is_cpu_address_exist) {
    if (device_address->from_mem_pool()) {
      // If CPU address is exit, and address from pool, no need to copy.
      return;
    } else {
      // If not from the pool, the lifetime of the device ptr is guaranteed elsewhere.
      // Before applying for a new address, clear the address. Otherwise a warnging is generated.
      device_address->set_ptr(nullptr);
      if (device_context->GetDeviceType() != device_address->GetDeviceType()) {
        device_context = runtime::OpRunner::GetDeviceContext(kCPUDevice);
        MS_EXCEPTION_IF_NULL(device_context);
      }
    }
  }

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "ContiguousAllocMem", "");
  auto mem_type = tensor->is_parameter() ? device::tracker::MemType::kWeight : device::tracker::MemType::kPyNativeInput;
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", mem_type, device_address->GetSize(),
                                                 device_address.get());
  if ((device_address->GetPtr() == nullptr) &&
      (!device_context->device_res_manager_->AllocateMemory(device_address.get()))) {
    MS_LOG(EXCEPTION) << "Allocate memory failed";
  }

  auto tensor_size = LongToSize(tensor->data().nbytes());
  auto tensor_type = tensor->data_type();
  if (!device_address->SyncHostToDevice(tensor->shape(), tensor_size, tensor_type, "DefaultFormat",
                                        tensor->data_ptr())) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed";
  }
}
}  // namespace mindspore::compile
