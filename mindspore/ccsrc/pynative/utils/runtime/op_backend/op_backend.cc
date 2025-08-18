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

#include "include/pynative/utils/runtime/op_backend/op_backend.h"

#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "primitive/structure_op_name.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/backend/common/device_address_utils.h"
#include "include/runtime/pipeline/pipeline.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "ir/tensor_new.h"
#include "ir/dtype/tensor_type.h"
#include "utils/stream_guard.h"

namespace mindspore::compile {
namespace {
#if !defined(__APPLE__)
bool EnablePyNativeSyncRunning() {
  bool sync_stream = runtime::RuntimeConf::GetInstance()->launch_blocking();
  return sync_stream;
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

void WaitTasksFinish() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                     runtime::kDefaultOpName);
  GilReleaseWithCheck gil_release;
  runtime::Pipeline::Get().backend_stage()->Wait();
  runtime::Pipeline::Get().launch_stage()->Wait();
}

std::vector<KernelTensorPtr> CreateGraphOutputKernelTensor(const OpCompilerInfoPtr &op_compiler_info,
                                                           const abstract::AbstractBasePtr &out_abstract,
                                                           size_t stream_id) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  auto device_context = op_compiler_info->device_context_;
  const auto &output_edges = op_compiler_info->simple_graph_->outputs_;
  size_t output_num = output_edges.size();

  std::vector<KernelTensorPtr> output_kernel_tensor_list;
  output_kernel_tensor_list.reserve(output_num);

  for (size_t i = 0; i < output_num; ++i) {
    const auto &edge = output_edges[i];
    MS_EXCEPTION_IF_NULL(edge);
    auto kernel_tensor = edge->kernel_tensor_;
    if (kernel_tensor != nullptr) {
      MS_LOG(DEBUG) << "Already have output kernel tensor for ref output";
      output_kernel_tensor_list.push_back(kernel_tensor);
      continue;
    }

    const auto &[output_node, index] = edge->node_with_index_;
    auto cache_output_kernel_tensor = edge->origin_kernel_tensor_;

    auto real_abstract = out_abstract;
    if (out_abstract->isa<abstract::AbstractSequence>()) {
      auto abstract_tuple = out_abstract->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(abstract_tuple);
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
    const auto &shape = shape_vector->shape();
    MS_EXCEPTION_IF_NULL(cache_output_kernel_tensor->device_address());

    auto output_type = cache_output_kernel_tensor->dtype_id();
    const auto &output_format = kernel::GetFormatFromEnumToStr(cache_output_kernel_tensor->format());
    auto address_size = runtime::DeviceAddressUtils::GetTensorDeviceSize(device_context, output_node, shape,
                                                                         output_format, output_type, index);
    const auto &new_kernel_tensor = AnfAlgo::CreateKernelTensor(
      real_abstract->GetShape()->Clone(), real_abstract->GetType()->Clone(), real_abstract->GetValue(), nullptr,
      address_size, output_format, output_type, shape,
      device::GetDeviceNameByType(device_context->device_context_key().device_type_),
      device_context->device_context_key().device_id_, cache_output_kernel_tensor->user_data());
    new_kernel_tensor->set_stream_id(stream_id);
    MS_LOG(DEBUG) << "Create addr for node:" << output_node->DebugString()
                  << " kernel tensor:" << new_kernel_tensor->ToString();
    output_kernel_tensor_list.push_back(new_kernel_tensor);
    edge->kernel_tensor_ = new_kernel_tensor;
  }
  return output_kernel_tensor_list;
}
}  // namespace

void OpBackend::Run(const BackendOpRunInfoPtr &op_run_info, device::DeviceType device_type, uint32_t device_id,
                    VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  ViewBackend::ContiguousInputByRunInfo(op_run_info);
  auto device_name = device::GetDeviceNameByType(device_type);
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
    WaitTasksFinish();
  }
  if (!single_op_cache_hit) {
    pynative::OpCompiler::GetInstance().KernelBuild(op_compiler_info, device_context, false);
  }
  const auto &tensors_without_value_mask = runtime::OpRunner::GetTensorWithoutValueMask(op_run_info);
  runtime::OpRunner::UpdateDeviceAddress(graph, tensors_without_value_mask, device_context, true,
                                         op_run_info->base_op_run_info.stream_id);

  runtime::OpRunner::RunSingleOpGraph(op_run_info, op_compiler_info, tensors_without_value_mask);

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
    runtime::DynamicOpRunner::UpdateInputDeviceAddress(op_compiler_info, input_tensors, false,
                                                       op_run_info->base_op_run_info.stream_id);
    auto kernel_tensor_list = CreateGraphOutputKernelTensor(op_compiler_info, op_run_info->base_op_run_info.abstract,
                                                            op_run_info->base_op_run_info.stream_id);
    // Create output tensor
    post_run_.UpdateOutputDynamic(op_run_info, op_compiler_info, kernel_tensor_list, outputs);
    DispatchOpTaskDynamic(outputs, op_compiler_info, op_run_info, kernel_tensor_list);
    return;
  }
  MS_LOG(DEBUG) << "Async exec disabled, op: " << op_run_info->base_op_run_info.op_name;
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (!op_executor.RunQueueEmpty()) {
    WaitTasksFinish();
  }
  auto input_tensors = runtime::OpRunner::GetTensorWithoutValueMask(op_run_info);
  runtime::DynamicOpRunner::UpdateInputDeviceAddress(op_compiler_info, input_tensors, true,
                                                     op_run_info->base_op_run_info.stream_id);
  runtime::DynamicOpRunner::RunSingleOpGraph(op_run_info, op_compiler_info, input_tensors);

  const auto &kernel_tensor_list = GetOutputKernelTensor(op_compiler_info);
  // Create output tensor
  post_run_.UpdateOutputDynamic(op_run_info, op_compiler_info, kernel_tensor_list, outputs);
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
                                         op_compiler_info->device_context_, false,
                                         op_run_info->base_op_run_info.stream_id);
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
  MS_EXCEPTION_IF_NULL(context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, context->is_pynative_infer());
  runtime::OpRunner::RunSingleOpGraph(context->op_run_info(), context->op_compiler_info(),
                                      runtime::OpRunner::GetTensorWithoutValueMask(context->op_run_info()));

  MS_EXCEPTION_IF_NULL(context->op_run_info());
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
                                      const std::vector<KernelTensorPtr> &kernel_tensor_list) {
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
  MS_EXCEPTION_IF_NULL(context);
  auto infer_flag = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, context->is_pynative_infer());

  runtime::DynamicOpRunner::RunSingleOpGraph(context->op_run_info(), context->op_compiler_info(),
                                             runtime::OpRunner::GetTensorWithoutValueMask(context->op_run_info()));
  post_run_.ClearOpInputOutput(context->op_compiler_info());
  // Reset PyNative infer flag.
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, infer_flag);
  MS_LOG(DEBUG) << "OpRunCallback end";
}

std::vector<KernelTensorPtr> OpBackend::GetOutputKernelTensor(const OpCompilerInfoPtr &op_compiler_info) const {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  const auto &output_edges = op_compiler_info->simple_graph_->outputs_;
  std::vector<KernelTensorPtr> output_kernel_tensors;
  output_kernel_tensors.reserve(output_edges.size());
  std::transform(output_edges.begin(), output_edges.end(), std::back_inserter(output_kernel_tensors),
                 [](const pynative::EdgePtr &edge) { return edge->kernel_tensor_; });
  return output_kernel_tensors;
}

void OpBackend::RunViewKernelTask(const pynative::BaseOpRunInfo &base_op_run_info,
                                  const runtime::KernelTaskType &task_type, bool enable_async) const {
  view_backend_.RunViewKernelTask(base_op_run_info, task_type, enable_async);
}

void OpBackend::RunAllocMemTask(DeviceContext *device_context, const tensor::TensorPtr &tensor,
                                bool enable_async) const {
  view_backend_.RunAllocMemTask(device_context, tensor, enable_async);
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

tensor::TensorPtr PostRunOp::CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index) const {
  MS_EXCEPTION_IF_NULL(output_node);
  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, output_index, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);

  const auto &user_data = kernel_tensor->user_data();

  device_tensor->SetNodeIndex(output_node, output_index);
  device_tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));

  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto tensor =
    tensor::from_spec(kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector(), device::DeviceType::kNone);

  // Put device tensor into host tensor.
  tensor->set_device_address(device_tensor);
  tensor->set_format(kernel_tensor->format());
  tensor->set_storage_info(kernel_tensor->tensor_storage_info());
  tensor->set_sync_status(kNeedSyncDeviceToHost);
  return tensor;
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
      auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, i, false);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto device_address = kernel_tensor->device_address();
      if (device_address == nullptr) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(device_context);
      auto new_kernel_tensor = runtime::DeviceAddressUtils::CloneEmptyKernelTensor(kernel_tensor, device_context);
      auto &new_device_address = new_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(new_device_address);
      if (is_gradient_out) {
        new_device_address->set_from_persistent_mem(true);
      }
      AnfAlgo::SetOutputAddr(new_device_address, new_kernel_tensor->tensor_storage_info(), i, node);
    }

    // Clear old workspace device address of kernel
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_lists.size(); ++i) {
      if (!AnfAlgo::WorkspaceAddrExist(node, i)) {
        continue;
      }
      const auto &kernel_tensor = AnfAlgo::GetWorkspaceKernelTensor(node, i);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto new_kernel_tensor = runtime::DeviceAddressUtils::CloneEmptyKernelTensor(kernel_tensor, device_context);
      AnfAlgo::SetWorkspaceKernelTensor(new_kernel_tensor, i, node.get());
    }
  }
}

void PostRunOp::ClearInputDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  for (const auto &node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<Parameter>()) {
      auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, 0, false);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto device_address = kernel_tensor->device_address();
      if (device_address == nullptr) {
        continue;
      }
      auto new_kernel_tensor = runtime::DeviceAddressUtils::CloneEmptyKernelTensor(kernel_tensor, device_context);
      AnfAlgo::SetOutputAddr(new_kernel_tensor->device_address(), new_kernel_tensor->tensor_storage_info(), 0, node);
    }
  }
}

void PostRunOp::ClearOpInputOutput(const OpCompilerInfoPtr &op_compiler_info) const {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  const auto &all_edges = op_compiler_info->simple_graph_->all_edges_;
  for (const auto &edge : all_edges) {
    MS_EXCEPTION_IF_NULL(edge);
    if (edge->type_ != pynative::EdgeType::kValueNodeEdge) {
      // Just set edge address to null rather than clone empty address.
      // Clone empty address in next RunOp if needed.
      edge->kernel_tensor_ = nullptr;
    }
  }
}

void PostRunOp::UpdateOutputAbstract(const VectorRef &outputs, const session::BackendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto output_size = outputs.size();
  if (output_size == 1 && op_run_info->base_op_run_info.op_name != kGetNextOpName) {
    auto output_tensor = utils::cast<tensor::TensorPtr>(outputs[0]);
    MS_EXCEPTION_IF_NULL(output_tensor);
    op_run_info->base_op_run_info.abstract = output_tensor->ToAbstract();
    MS_LOG(DEBUG) << "Update output abstract of " << op_run_info->base_op_run_info.op_name << " to "
                  << op_run_info->base_op_run_info.abstract->ToString();
    return;
  }
  AbstractBasePtrList elements;
  for (size_t i = 0; i < output_size; ++i) {
    auto output_tensor = utils::cast<tensor::TensorPtr>(outputs[i]);
    MS_EXCEPTION_IF_NULL(output_tensor);
    (void)elements.emplace_back(output_tensor->ToAbstract());
  }
  op_run_info->base_op_run_info.abstract = std::make_shared<abstract::AbstractTuple>(elements);
  MS_LOG(DEBUG) << "Update output abstract of " << op_run_info->base_op_run_info.op_name << " to "
                << op_run_info->base_op_run_info.abstract->ToString();
}

void PostRunOp::UpdateOutputDynamic(const session::BackendOpRunInfoPtr &op_run_info,
                                    const OpCompilerInfoPtr &op_compiler_info,
                                    const std::vector<KernelTensorPtr> &kernel_tensor_list, VectorRef *outputs) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "No promise, just create tensor and address, op " << op_run_info->base_op_run_info.op_name;
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  auto output_nodes = op_compiler_info->graph_output_nodes_;
  auto outputs_size = output_nodes.size();
  if (op_compiler_info->graph_outputs_tensor_num_.size() != outputs_size) {
    MS_LOG(EXCEPTION) << "The size of graph_outputs_tensor_num_:" << op_compiler_info->graph_outputs_tensor_num_.size()
                      << " is not equal to outputs_size:" << outputs_size;
  }

  if (kernel_tensor_list.size() != outputs_size) {
    MS_LOG(EXCEPTION) << "The size of kernel_tensor_list:" << kernel_tensor_list.size()
                      << " is not equal to outputs_size:" << outputs_size;
  }

  for (size_t i = 0; i < outputs_size; ++i) {
    auto item_with_index = output_nodes[i];
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (op_compiler_info->graph_outputs_tensor_num_[i] == 0) {
      continue;
    }
    auto output_kernel_tensor = kernel_tensor_list[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_tensor = CreateOutputTensorDynamicImpl(op_compiler_info, item_with_index.first, item_with_index.second,
                                                       output_kernel_tensor, i);
    MS_EXCEPTION_IF_NULL(output_tensor);
    output_tensor->set_need_pipeline_sync(true);
    outputs->emplace_back(output_tensor);
  }
}

tensor::TensorPtr PostRunOp::CreateOutputTensorDynamicImpl(const OpCompilerInfoPtr &op_compiler_info,
                                                           const AnfNodePtr &output_node, size_t output_index,
                                                           const KernelTensorPtr &kernel_tensor,
                                                           size_t idx_in_graph_outputs) const {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(op_compiler_info);

  const auto &user_data = kernel_tensor->user_data();

  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  const auto &address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(address);

  auto tensor =
    tensor::from_spec(kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector(), device::DeviceType::kNone);

  // Put device tensor into host tensor.
  address->SetNodeIndex(output_node, output_index);
  address->set_padding_type(op_compiler_info->graph_outputs_padding_type_[idx_in_graph_outputs]);
  tensor->set_device_address(address);
  tensor->set_format(kernel_tensor->format());
  tensor->set_storage_info(kernel_tensor->tensor_storage_info());
  return tensor;
}

void ViewBackend::ContiguousInputByRunInfo(const BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(contiguous_func_);
  contiguous_func_(op_run_info);
}

void ViewBackend::RunViewKernelTask(const pynative::BaseOpRunInfo &base_op_run_info,
                                    const runtime::KernelTaskType &task_type, bool enable_async) const {
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {base_op_run_info.device_target, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);

  std::vector<tensor::TensorPtr> input_tensors;
  for (size_t idx = 0; idx < base_op_run_info.expanded_input_values.size(); idx++) {
    auto input_tensor = base_op_run_info.expanded_input_values[idx]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(input_tensor);
    // always false.
    if (input_tensor->device_address() == nullptr) {
      if (idx == 0) {
        MS_LOG(EXCEPTION) << "First tensor can not be nullptr, op name:" << base_op_run_info.op_name;
      }
      auto address_size = GetTypeByte(TypeIdToType(input_tensor->data_type())) * SizeOf(input_tensor->shape());

      auto kernel_tensor = AnfAlgo::CreateKernelTensor(
        nullptr, address_size, Format::DEFAULT_FORMAT, input_tensor->data_type(), input_tensor->shape(),
        device::GetDeviceNameByType(device_context->device_context_key().device_type_),
        device_context->device_context_key().device_id_);
      MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString();
      kernel_tensor->SetType(std::make_shared<TensorType>(input_tensor->Dtype()));
      kernel_tensor->SetShape(std::make_shared<abstract::TensorShape>(input_tensor->shape()));
      kernel_tensor->set_stream_id(base_op_run_info.stream_id);
      auto input_addr = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(input_addr);

      input_tensor->set_device_address(input_addr);
      input_tensor->set_format(kernel_tensor->format());
      input_tensor->set_storage_info(kernel_tensor->tensor_storage_info());
      RunAllocMemTask(device_context, input_tensor, enable_async);
      (void)input_tensors.emplace_back(input_tensor);
    } else {
      auto input_addr = std::static_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
      MS_EXCEPTION_IF_NULL(input_addr);
      if (input_addr->GetDeviceType() == device::DeviceType::kCPU) {
        RunAllocMemTask(device_context, input_tensor, enable_async);
      }
      (void)input_tensors.emplace_back(input_tensor);
    }
  }

  const auto &output_tensors = base_op_run_info.output_tensors;

  if (enable_async) {
    RunViewKernelTaskAsyncImpl(task_type, device_context, input_tensors, output_tensors, base_op_run_info.stream_id);
  } else {
    WaitTasksFinish();
    runtime::OpRunner::LaunchKernelTask(task_type, device_context, input_tensors, output_tensors,
                                        base_op_run_info.stream_id);
  }
}

void ViewBackend::RunAllocMemTask(DeviceContext *device_context, const tensor::TensorPtr &tensor,
                                  bool enable_async) const {
  if (!enable_async) {
    WaitTasksFinish();
    return AllocateMemForTensor(tensor, device_context);
  }
  auto alloc_mem_func = [this, device_context, tensor]() { AllocateMemForTensor(tensor, device_context); };
  runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
    std::make_shared<runtime::PassthroughDeviceTask>(alloc_mem_func));
}

void ViewBackend::RunViewKernelTaskAsyncImpl(const runtime::KernelTaskType &task_type, DeviceContext *device_context,
                                             const tensor::TensorPtrList &input_tensors,
                                             const tensor::TensorPtrList &output_tensors,
                                             const size_t &stream_id) const {
  static auto kernel_task_func = [stream_id, task_type, input_tensors, output_tensors, device_context]() {
    runtime::OpRunner::LaunchKernelTask(task_type, device_context, input_tensors, output_tensors, stream_id);
  };

  runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
    std::make_shared<runtime::PassthroughDeviceTask>(kernel_task_func));
}

void ViewBackend::AllocateMemForTensor(const tensor::TensorPtr &tensor, DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context);

  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);

  if (device_address->GetPtr() != nullptr) {
    MS_LOG(DEBUG) << "Input device address already allocated.";
    return;
  }
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "ContiguousAllocMem", "");
  auto mem_type =
    tensor->is_parameter() ? memory::mem_pool::MemType::kWeight : memory::mem_pool::MemType::kPyNativeInput;
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", mem_type, device_address->GetSize(),
                                                 device_address.get());
  if (!device_context->device_res_manager_->AllocateMemory(device_address.get(), CurrentStream::id())) {
    MS_LOG(EXCEPTION) << "Allocate memory failed";
  }
  MS_LOG(DEBUG) << "Start lazy copy for tensor " << tensor->ToString();
  runtime::DeviceAddressUtils::LazyCopy(tensor, CurrentStream::id());

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
    MarkTensorAsOutput, "PyNative", device::GetDeviceNameByType(device_address->GetDeviceType()),
    device_address->GetPtr(), tensor->data_type(), tensor->shape(), tensor->storage_info());
}
}  // namespace mindspore::compile
