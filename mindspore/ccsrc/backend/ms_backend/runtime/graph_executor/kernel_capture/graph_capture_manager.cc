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

#include "backend/ms_backend/runtime/graph_executor/kernel_capture/graph_capture_manager.h"
#include <string>
#include <algorithm>
#include <queue>
#include <vector>
#include <utility>
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/llm_manager.h"

namespace mindspore {
namespace runtime {
GraphCaptureManager &GraphCaptureManager::GetInstance() noexcept {
  static GraphCaptureManager instance{};
  return instance;
}

bool GraphCaptureManager::GetEnableGraphCapture() const {
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  return runtime_conf_instance->GetEnableKernelLaunchCapture();
}

void GraphCaptureManager::SetEnableGraphCapture(bool enable_graph_capture) {
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  runtime_conf_instance->SetEnableKernelLaunchCapture(enable_graph_capture);
}

bool GraphCaptureManager::CheckKernelSupportCapture(const KernelRunnerPtr &kernel_runner,
                                                    const DeviceContext *expected_device_context) {
  MS_EXCEPTION_IF_NULL(kernel_runner);
  const auto &kernel = kernel_runner->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  const auto &kernel_mod = kernel_runner->kernel_mod();
  MS_EXCEPTION_IF_NULL(kernel_mod);

  auto kernel_type = AnfAlgo::GetKernelType(kernel);
  if (kernel_type == KernelType::ACL_KERNEL) {
    return false;
  }

  if ((kernel_runner->device_contexts())[0]->GetDeviceType() != expected_device_context->GetDeviceType()) {
    MS_LOG(EXCEPTION) << "Capture graph mode can not support cpu kernel: " << kernel->fullname_with_scope();
  }

  if (kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
    MS_LOG(EXCEPTION)
      << "Capture graph mode can not support computed depend kernel(whose shape need update after launch.): "
      << kernel->fullname_with_scope();
  }

  auto &llm_manager = LLMManager::GetInstance();
  if (llm_manager.need_force_resize(kernel_mod->kernel_name()) || kernel_runner->is_dynamic_value()) {
    return false;
  }

  const auto &op_capture_skip = RuntimeConf::GetInstance()->GetNotCaptureOpList();
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_runner->kernel());
  std::transform(kernel_name.begin(), kernel_name.end(), kernel_name.begin(), ::tolower);

  for (const auto &not_capture_op : op_capture_skip) {
    auto lower_op = not_capture_op;
    std::transform(lower_op.begin(), lower_op.end(), lower_op.begin(), ::tolower);

    if (kernel_name == lower_op) {
      MS_LOG(INFO) << "Not capturing op: " << not_capture_op;
      return false;
    }
  }

  return true;
}

bool GraphCaptureManager::FindSupportCaptureKernelPositions(const std::vector<KernelRunnerPtr> &kernel_runners,
                                                            const DeviceContext *expected_device_context) {
  if (!capture_kernel_range_positions_.empty()) {
    MS_LOG(EXCEPTION) << "GraphCaptureManager has already initialized.";
  }
  init_ = true;
  size_t start = 0;
  size_t end = 0;
  bool find_kernel_can_capture = false;
  size_t kernel_num = kernel_runners.size();
  if (kernel_num < 1) {
    return false;
  }
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel_runner = kernel_runners[i];
    if (kernel_runner == nullptr) {
      continue;
    }

    if (CheckKernelSupportCapture(kernel_runner, expected_device_context)) {
      if (!find_kernel_can_capture) {
        start = i;
        end = i;
        find_kernel_can_capture = true;
      } else {
        end = i;
      }
    } else {
      if (find_kernel_can_capture) {
        capture_kernel_range_positions_.emplace_back(start, end);
        executors_.emplace_back(CAPTURE_GRAPH, (capture_kernel_range_positions_.size() - static_cast<size_t>(1)));
      }
      executors_.emplace_back(KERNEL, i);
      find_kernel_can_capture = false;
    }
  }

  if (find_kernel_can_capture) {
    capture_kernel_range_positions_.emplace_back(start, end);
    executors_.emplace_back(CAPTURE_GRAPH, (capture_kernel_range_positions_.size() - static_cast<size_t>(1)));
  }

  capture_graph_num_ = capture_kernel_range_positions_.size();
  MS_LOG(INFO) << "Capture graph num: " << capture_graph_num_;
  auto executor_size = executors_.size();
  MS_LOG(DEBUG) << "Dump executor info for capture graph: ";
  for (size_t i = 0; i < executor_size; i++) {
    std::string executor_mode = (executors_[i].first == CAPTURE_GRAPH ? "capture graph" : "kernel");
    std::ostringstream executor_mode_info;
    if (executors_[i].first == CAPTURE_GRAPH) {
      const auto &range_pair = capture_kernel_range_positions_.at(executors_[i].second);
      executor_mode_info << "executor range:[" << std::to_string(range_pair.first) << ", "
                         << std::to_string(range_pair.second) << "].";
    } else {
      executor_mode_info << "executor order:[" << std::to_string(executors_[i].second) << "]";
    }
    MS_LOG(DEBUG) << "The executor[" << i << "] is " << executor_mode << ", " << executor_mode_info.str();
  }

  return capture_graph_num_ > 0;
}

void GraphCaptureManager::Initialize(const DeviceContext *device_context) {
  std::vector<CaptureGraphPtr> cur_capture_graphs;
  for (size_t i = 0; i < capture_graph_num_; i++) {
    cur_capture_graphs.push_back(device_context->device_res_manager_->CreateCaptureGraph());
  }
  capture_graphs_[shape_key_] = cur_capture_graphs;
}

bool GraphCaptureManager::LaunchAllKernelsWithCapture(OpContext<KernelTensor> *const context,
                                                      const std::vector<KernelRunnerPtr> &kernel_runners,
                                                      SuperKernelActor *super_kernel_actor, bool hp_mode) {
  MS_LOG(INFO) << "Begin launch all kernels with capture graph.";
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first == CAPTURE_GRAPH) {
      size_t start = capture_kernel_range_positions_[executor.second].first;
      size_t end = capture_kernel_range_positions_[executor.second].second;
      const auto &cur_capture_graph = capture_graphs_[shape_key_][executor.second];
      MS_EXCEPTION_IF_NULL(cur_capture_graph);
      if (!cur_capture_graph->CaptureBegin(0)) {
        MS_LOG(EXCEPTION)
          << "Capture graph failed, most likely because the number of subgraphs you captured exceeded the "
             "hardware limit. Currently captured shape count: "
          << (capture_graphs_.size() - 1)
          << ", Please set export MS_DEV_RUNTIME_CONF='graph_capture_max_number:" << (capture_graphs_.size() - 1)
          << " to control the maximum number of captured shapes.";
      }
      MS_LOG(DEBUG) << "Begin captrue graph, executor index: " << i << ", range[" << start << ", " << end << "].";

      for (size_t j = start; j <= end; j++) {
        const auto &kernel_runner = kernel_runners[j];
        if (kernel_runner == nullptr) {
          continue;
        }
        if (!super_kernel_actor->LaunchKernelForCaptureGraph(context, kernel_runner, j, true)) {
          MS_LOG(ERROR) << "Launch kernel in capture mode failed: " << kernel_runner->kernel()->fullname_with_scope();
          return false;
        }
        RecordGraphOutputKernelInfo(context, kernel_runner, j);
      }
      cur_capture_graph->CaptureEnd(0);
      MS_LOG(DEBUG) << "Begin replay captrue graph, executor index: " << i << ", range[" << start << ", " << end
                    << "].";
      cur_capture_graph->ExecuteCaptureGraph(0);
    } else {
      auto &kernel_runner = kernel_runners[executor.second];
      MS_LOG(DEBUG) << "Begin launch kernel, executor order index: " << executor.second
                    << ", kernel: " << kernel_runner->kernel()->fullname_with_scope();
      if (!super_kernel_actor->LaunchKernelForCaptureGraph(context, kernel_runner, executor.second, false)) {
        MS_LOG(ERROR) << "Launch kernel failed: " << kernel_runner->kernel()->fullname_with_scope();
        return false;
      }
    }
  }
  MS_LOG(INFO) << "End launch all kernels with capture graph.";
  return true;
}

bool GraphCaptureManager::LaunchAllKernelsWithReplayGraph(OpContext<KernelTensor> *const context,
                                                          const std::vector<KernelRunnerPtr> &kernel_runners,
                                                          SuperKernelActor *super_kernel_actor, bool hp_mode) {
  MS_LOG(INFO) << "Begin launch all kernels with replay graph.";
  size_t executor_num = executors_.size();
  RecoverGraphOutputKernelInfo();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first == CAPTURE_GRAPH) {
      MS_EXCEPTION_IF_NULL(capture_graphs_[shape_key_][executor.second]);
      capture_graphs_[shape_key_][executor.second]->ExecuteCaptureGraph(0);
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End launch sub graph in replay step";
    } else {
      auto &kernel_runner = kernel_runners[executor.second];
      FetchNonFixedInput(kernel_runner, context, 0);
      if (!super_kernel_actor->LaunchKernelForReplayGraph(context, kernel_runner, executor.second)) {
        MS_LOG(ERROR) << "Launch kernel failed: " << kernel_runner->kernel()->fullname_with_scope();
        return false;
      }
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End launch single op in replay step";
    }
  }
  MS_LOG(INFO) << "End launch all kernels with replay graph.";
  return true;
}

void GraphCaptureManager::RecordGraphOutputKernelInfo(OpContext<KernelTensor> *const context,
                                                      const KernelRunnerPtr &kernel_actor, size_t index) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Record current kernel actor: "
                                       << kernel_actor->kernel()->fullname_with_scope();
  const auto &cur_output_kernel_tensors = kernel_actor->output_kernel_tensors();
  const auto &is_output_kernels = kernel_actor->is_output_kernel();
  CaptureKernelInfoList fix_output_graph_kernel_tensors;
  std::vector<size_t> output_kernel_indices;
  for (size_t i = 0; i < cur_output_kernel_tensors.size(); ++i) {
    if (is_output_kernels[i]) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Record graph output for capture graph, output kernel tensor info: "
        << cur_output_kernel_tensors[i]->ToString();
      fix_output_graph_kernel_tensors.emplace_back(std::make_shared<CaptureKernelInfo>(
        cur_output_kernel_tensors[i]->device_ptr(), cur_output_kernel_tensors[i]->size(),
        cur_output_kernel_tensors[i]->GetShape()->Clone()));
      output_kernel_indices.emplace_back(i);
    }
  }
  if (!output_kernel_indices.empty()) {
    fix_replay_graph_output_info_[shape_key_][std::make_pair(kernel_actor, index)] = fix_output_graph_kernel_tensors;
    recorded_kernel_output_for_graph_output_[shape_key_].emplace_back(std::make_pair(kernel_actor, index),
                                                                      output_kernel_indices);
  }
}

void GraphCaptureManager::RecoverGraphOutputKernelInfo() {
  const auto &cur_fix_output_graph_kernel_tensor_infos = recorded_kernel_output_for_graph_output_[shape_key_];
  for (auto cur_kernel_info : cur_fix_output_graph_kernel_tensor_infos) {
    auto kernel_with_idx = cur_kernel_info.first;
    const auto &cur_fix_output_graph_kernel_tensor_info = fix_replay_graph_output_info_[shape_key_][kernel_with_idx];
    auto cur_output_kernel_tensors = kernel_with_idx.first->output_kernel_tensors();
    size_t tmp = 0;
    for (auto cur_idx : cur_kernel_info.second) {
      cur_output_kernel_tensors[cur_idx]->set_device_ptr(cur_fix_output_graph_kernel_tensor_info[tmp]->device_ptr);
      cur_output_kernel_tensors[cur_idx]->SetShape(cur_fix_output_graph_kernel_tensor_info[tmp]->shape);
      cur_output_kernel_tensors[cur_idx]->set_size(cur_fix_output_graph_kernel_tensor_info[tmp]->size);
      tmp++;
    }
  }
}

void GraphCaptureManager::HandleFirstUserMemoryFree(const KernelTensorPtr &kernel_tensor,
                                                    const KernelRunnerPtr &kernel_actor,
                                                    std::queue<std::vector<KernelTensorPtr>> *memory_free_lists) {
  if (kernel_tensor->new_ref_count() != SIZE_MAX) {
    memory_free_lists->back().emplace_back(kernel_tensor);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Add memory free list for tensor:" << kernel_tensor->ToString();
  }
}

bool GraphCaptureManager::IsNonFixedInput(GraphParameterStore *cur_graph_parameter_store, const AnfNodePtr &node,
                                          size_t parameter_idx) {
  bool is_weight = cur_graph_parameter_store->GetPositionWeight(parameter_idx);
  std::string cur_node_name = node->fullname_with_scope();
  bool is_kv_cache =
    (cur_node_name.find("key_cache") != std::string::npos || cur_node_name.find("value_cache") != std::string::npos);
  return is_weight || is_kv_cache;
}

void GraphCaptureManager::FetchAllInputsBeforeCaptureGraph(
  OpContext<KernelTensor> *const context, const std::vector<KernelRunnerPtr> &kernel_runners,
  std::queue<std::vector<KernelTensorPtr>> *memory_free_lists) {
  MS_LOG(INFO) << "Begin fetch all kernels inputs before capture graph.";
  InitFixedInputInfoForSingleOp(kernel_runners);
  size_t kernel_num = kernel_runners.size();
  auto cur_graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(cur_graph_parameter_store);
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel_actor = kernel_runners[i];
    if (kernel_actor == nullptr) {
      continue;
    }
    auto kernel_with_idx = std::make_pair(kernel_actor, i);
    for (const auto &parameter_index : kernel_actor->parameter_indexs()) {
      size_t kernel_input_index = parameter_index.first;
      auto outer_index = parameter_index.second.second;
      auto node = parameter_index.second.first.first;
      bool is_first_user = kernel_actor->is_first_used_params()[kernel_input_index];
      auto kernel_tensor = FetchParameter(parameter_index.second, kernel_actor->GetAID(), is_first_user, 0, false);
      const auto &device_tensor = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_tensor);
      auto cur_device_context = kernel_actor->device_contexts()[0];
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Actor: " << kernel_actor->GetAID().Name() << ", input index: " << kernel_input_index
        << ", device tensor: " << device_tensor << ", ptr: " << device_tensor->GetPtr()
        << " new ref count:" << kernel_tensor->new_ref_count()
        << " super kernel actor context:" << cur_device_context->device_context_key().ToString()
        << " kernel actor context:" << cur_device_context->device_context_key().ToString();
      auto real_input_data_infos = kernel_actor->real_input_data_infos();
      auto &real_input_info = real_input_data_infos[kernel_input_index];
      if ((device_tensor->GetDeviceType() != cur_device_context->GetDeviceType()) ||
          !AnfAlgo::IsEquivalentFormat(kernel_tensor->format(), real_input_info->format_) ||
          kernel_tensor->dtype_id() != real_input_info->type_id_) {
        MS_EXCEPTION(RuntimeError) << "Does not support heterogeneous scenarios";
      }
      // deal weight/KV Cache
      if (IsNonFixedInput(cur_graph_parameter_store, node, outer_index)) {
        // Save the weight or kv value for the subsequent CheckParameterNotChange function.
        if (weight_kv_addrs_[shape_key_].find(parameter_index.second.first) == weight_kv_addrs_[shape_key_].end()) {
          weight_kv_addrs_[shape_key_][parameter_index.second.first] = {kernel_tensor, parameter_index.second.second,
                                                                        kernel_actor};
        }
        kernel_actor->SetInputDeviceTensor(kernel_tensor, kernel_input_index);
        continue;
      }
      // deal with normal inputs
      if (fixed_addrs_for_set_inputs_[shape_key_].find(parameter_index.second.first) ==
          fixed_addrs_for_set_inputs_[shape_key_].end()) {
        const auto storage_info = kernel_tensor->tensor_storage_info();
        if (storage_info) {
          MS_LOG(EXCEPTION)
            << "The input[" << kernel_input_index << "] of kernel(" << kernel_actor->GetAID().Name()
            << ") got a non-contiguous memory layout tensor, and framework will automatically convert it "
               "to contiguous memory layout."
               " The capture graph feature can not work in this case, please find the source of "
               "non-contiguous input and convert it to contiguous memory layout, or disable capture graph "
               "feature by set_kernel_launch_capture(False). Note: Disabling the capture "
               "graph feature will degrade cpu execute performance, which may reduce network "
               "execution performance.";
        }
        auto strategy = kernel_actor->get_strategy();
        auto fix_kernel_tensor = AnfAlgo::CreateKernelTensor(
          kernel_tensor->GetShape(), kernel_tensor->GetType(), kernel_tensor->GetValueTrack(), nullptr,
          real_input_info->size_, kernel::GetFormatFromEnumToStr(real_input_info->format_), real_input_info->type_id_,
          real_input_info->shape_, device::GetDeviceNameByType(cur_device_context->device_context_key().device_type_),
          cur_device_context->device_context_key().device_id_, kernel_tensor->user_data());
        MS_EXCEPTION_IF_NULL(kernel_tensor->GetShape());
        fix_kernel_tensor->SetShape(kernel_tensor->GetShape()->Clone());
        fix_kernel_tensor->set_size(device_tensor->GetSize());
        auto fix_device_tensor = fix_kernel_tensor->device_address();
        if (fix_device_tensor->GetPtr() == nullptr) {
          device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, kernel_actor->GetAID().Name(),
                                                         memory::mem_pool::MemType::kOther,
                                                         fix_device_tensor->GetSize(), fix_device_tensor.get());
          if (!cur_device_context->device_res_manager_->AllocateMemory(fix_device_tensor.get(), GetStreamId())) {
            SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy, *context, *(cur_device_context),
                                                        kernel_actor->GetAID().Name(), fix_device_tensor->GetSize());
          }
        }
        if (!AsyncCopy(fix_kernel_tensor.get(), kernel_tensor.get(), 0)) {
          MS_LOG(EXCEPTION) << "Async copy failed, src kernel tensor: " << kernel_tensor->ToString()
                            << ", dst kernel tensor: " << fix_kernel_tensor->ToString();
        }
        fix_kernel_tensor->set_new_ref_count(SIZE_MAX);

        // The fixed_addrs_for_set_inputs_ is to set input for kernel actors during the capture phase.
        fixed_addrs_for_set_inputs_[shape_key_][parameter_index.second.first] = fix_kernel_tensor;
        // The fixed_addrs_for_update_ is to update the fix_addr again before the replay phase.
        fixed_addrs_for_update_[shape_key_].emplace_back(parameter_index, fix_kernel_tensor, kernel_actor);
      }
      if (IsSingleOp(kernel_runners, i)) {
        fix_network_input_for_replay_single_op_[shape_key_][kernel_with_idx][kernel_input_index] =
          fixed_addrs_for_set_inputs_[shape_key_][parameter_index.second.first];
      }
      kernel_actor->SetInputDeviceTensor(fixed_addrs_for_set_inputs_[shape_key_][parameter_index.second.first],
                                         kernel_input_index);

      if (is_first_user) {
        HandleFirstUserMemoryFree(kernel_tensor, kernel_actor, memory_free_lists);
      }
    }
  }
}

bool GraphCaptureManager::IsSingleOp(const std::vector<KernelRunnerPtr> &kernel_runners, size_t kernel_index) {
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first != CAPTURE_GRAPH && executor.second == kernel_index) {
      return true;
    }
  }
  return false;
}

bool IsPositiveInteger(const std::string &str) {
  if (str.empty()) {
    return false;
  }
  for (char c : str) {
    if (!std::isdigit(c)) return false;
  }
  return str != "0";
}

bool GraphCaptureManager::IsExceedMaxCaptureCount() {
  auto graph_capture_max_number = runtime::GetRuntimeConfigValue(runtime::kRuntimeGraphCaptureMaxNumber);
  if (graph_capture_max_number.empty()) {
    MS_LOG(INFO) << "Get max capture count failed, max capture count config is empty, graph_capture_max_number: "
                 << graph_capture_max_number;
    return false;
  }
  if (!IsPositiveInteger(graph_capture_max_number)) {
    MS_EXCEPTION(RuntimeError)
      << "Max capture dynamic shape number config is not a positive integer, graph_capture_max_number: "
      << graph_capture_max_number;
  }
  size_t max_count = std::stoul(graph_capture_max_number);
  MS_LOG(INFO) << "Max capture count is " << max_count << ", current capture graph count is "
               << (capture_graphs_.size() - 1);
  return capture_graphs_.size() >= max_count;
}

void GraphCaptureManager::InitFixedInputInfoForSingleOp(const std::vector<KernelRunnerPtr> &kernel_runners) {
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first != CAPTURE_GRAPH) {
      auto kernel_idx = executor.second;
      auto &kernel_runner = kernel_runners[kernel_idx];
      auto kernel_with_idx = std::make_pair(kernel_runner, kernel_idx);
      auto cur_input_kernel_tensors = kernel_runner->input_kernel_tensors();
      std::vector<KernelTensorPtr> fix_input_kernel_tensors_for_single_op;
      fix_input_kernel_tensors_for_single_op.resize(cur_input_kernel_tensors.size());
      fix_network_input_for_replay_single_op_[shape_key_][kernel_with_idx] = fix_input_kernel_tensors_for_single_op;
    }
  }
}

bool GraphCaptureManager::IsNonFixedInputInReplay(const KernelRunnerPtr &kernel_runner, size_t kernel_input_index) {
  for (const auto &parameter_index : kernel_runner->parameter_indexs()) {
    size_t cur_kernel_input_index = parameter_index.first;
    if (cur_kernel_input_index == kernel_input_index) {
      auto outer_index = parameter_index.second.second;
      auto node = parameter_index.second.first.first;
      auto cur_graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
      MS_EXCEPTION_IF_NULL(cur_graph_parameter_store);
      if (IsNonFixedInput(cur_graph_parameter_store, node, outer_index)) {
        return true;
      }
      return false;
    }
  }
  return false;
}

void GraphCaptureManager::FetchNonFixedInput(const KernelRunnerPtr &kernel_actor,
                                             OpContext<KernelTensor> *const context, size_t stream_id) {
  auto cur_graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(cur_graph_parameter_store);
  for (const auto &parameter_index : kernel_actor->parameter_indexs()) {
    auto outer_index = parameter_index.second.second;
    auto node = parameter_index.second.first.first;
    if (IsNonFixedInput(cur_graph_parameter_store, node, outer_index)) {
      size_t kernel_input_index = parameter_index.first;
      bool is_first_user = kernel_actor->is_first_used_params()[kernel_input_index];
      bool has_h2d_copy = false;
      auto kernel_tensor =
        FetchParameter(parameter_index.second, kernel_actor->GetAID(), is_first_user, stream_id, false, &has_h2d_copy);
      if (has_h2d_copy) {
        MS_LOG(EXCEPTION) << "current parameter device address has changed!!!"
                          << " kernel_actor: " << kernel_actor->GetAID().Name()
                          << ", front node: " << node->DebugString()
                          << ", with index: " << parameter_index.second.first.second << ", addr index: " << outer_index;
      }
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Actor: " << kernel_actor->GetAID().Name() << ", input index: " << kernel_input_index
        << ", kernel tensor info: " << kernel_tensor->ToString()
        << " super kernel actor context:" << kernel_actor->device_contexts()[0]->device_context_key().ToString()
        << " kernel actor context:" << kernel_actor->device_contexts()[0]->device_context_key().ToString();
      kernel_actor->SetInputDeviceTensor(kernel_tensor, kernel_input_index);
    }
  }
}

void GraphCaptureManager::UpdateFixAddressBeforeReplayGraph(
  size_t stream_id, std::queue<std::vector<KernelTensorPtr>> *memory_free_lists) {
  MS_LOG(INFO) << "Begin update all fixed inputs before replay graph.";
  for (const auto &fix_pair : fixed_addrs_for_update_[shape_key_]) {
    auto parameter_index = std::get<kIndex0>(fix_pair);
    auto fix_kernel_tensor = std::get<kIndex1>(fix_pair);
    auto kernel_actor = std::get<kIndex2>(fix_pair);
    size_t kernel_input_index = parameter_index.first;
    MS_EXCEPTION_IF_NULL(kernel_actor);
    MS_EXCEPTION_IF_NULL(fix_kernel_tensor);
    auto cur_device_context = kernel_actor->device_contexts()[0];
    auto real_input_data_infos = kernel_actor->real_input_data_infos();
    auto &real_input_info = real_input_data_infos[kernel_input_index];
    bool is_first_user = kernel_actor->is_first_used_params()[kernel_input_index];
    auto kernel_tensor = FetchParameter(parameter_index.second, kernel_actor->GetAID(), true, stream_id, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    const auto &device_tensor = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_tensor);
    const auto storage_info = kernel_tensor->tensor_storage_info();
    if (storage_info) {
      MS_LOG(EXCEPTION) << "The input[" << kernel_input_index << "] of kernel(" << kernel_actor->GetAID().Name()
                        << ") got a non-contiguous memory layout tensor, and framework will automatically convert it "
                           "to contiguous memory layout."
                           " The capture graph feature can not work in this case, please find the source of "
                           "non-contiguous input and convert it to contiguous memory layout, or disable capture graph "
                           "feature by set_kernel_launch_capture(False). Note: Disabling the capture "
                           "graph feature will degrade cpu execute performance, which may reduce network "
                           "execution performance.";
    }
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Actor: " << kernel_actor->GetAID().Name() << ", input index: " << kernel_input_index
      << ", device tensor: " << device_tensor << ", ptr: " << device_tensor->GetPtr()
      << " new ref count:" << kernel_tensor->new_ref_count()
      << " super kernel actor context:" << cur_device_context->device_context_key().ToString();
    if ((device_tensor->GetDeviceType() != cur_device_context->GetDeviceType()) ||
        !AnfAlgo::IsEquivalentFormat(kernel_tensor->format(), real_input_info->format_) ||
        kernel_tensor->dtype_id() != real_input_info->type_id_) {
      MS_EXCEPTION(RuntimeError) << "Does not support heterogeneous scenarios";
    }
    if (!AsyncCopy(fix_kernel_tensor.get(), kernel_tensor.get(), 0)) {
      MS_LOG(EXCEPTION) << "Async copy failed, src kernel tensor: " << kernel_tensor->ToString()
                        << ", dst kernel tensor: " << fix_kernel_tensor->ToString();
    }
    if (is_first_user) {
      HandleFirstUserMemoryFree(kernel_tensor, kernel_actor, memory_free_lists);
    }
  }
}

bool GraphCaptureManager::CheckParameterNotChange(size_t stream_id) {
  for (const auto &weight_kv_addr : weight_kv_addrs_[shape_key_]) {
    auto old_kernel_tensor = std::get<kIndex0>(weight_kv_addr.second);
    MS_EXCEPTION_IF_NULL(old_kernel_tensor);
    auto outer_idx = std::get<kIndex1>(weight_kv_addr.second);
    auto kernel_actor = std::get<kIndex2>(weight_kv_addr.second);
    MS_EXCEPTION_IF_NULL(kernel_actor);
    auto front_node = weight_kv_addr.first.first;
    MS_EXCEPTION_IF_NULL(front_node);
    bool has_copy_weight = false;
    auto kernel_tensor = FetchParameter({weight_kv_addr.first, outer_idx}, kernel_actor->GetAID(), true, stream_id,
                                        false, &has_copy_weight);
    if (kernel_tensor->GetSize() != old_kernel_tensor->GetSize() ||
        kernel_tensor->device_ptr() != old_kernel_tensor->device_ptr() ||
        kernel_tensor->GetShape() != old_kernel_tensor->GetShape() || has_copy_weight) {
      MS_LOG(ERROR) << "current parameter device address has changed!!!"
                    << " kernel_actor: " << kernel_actor->GetAID().Name()
                    << ", front node: " << front_node->DebugString() << ", with index: " << weight_kv_addr.first.second
                    << ", addr index: " << outer_idx;
      return false;
    }
  }
  return true;
}

void GraphCaptureManager::SetShapeKey() {
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  const auto &host_tensor_shape = graph_parameter_store->GetHostTensorsShape();
  std::stringstream ss;
  for (size_t i = 0; i < host_tensor_shape.size(); ++i) {
    for (size_t j = 0; j < host_tensor_shape[i].size(); ++j) {
      if (i > 0 || j > 0) {
        ss << "-";
      }
      ss << host_tensor_shape[i][j];
    }
  }
  shape_key_ = ss.str();
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Cur shape: " << shape_key_;
}

bool GraphCaptureManager::HasCapturedGraph() {
  auto it = capture_graphs_.find(shape_key_);
  if (it != capture_graphs_.end()) {
    return true;
  }
  return false;
}

void GraphCaptureManager::RecodeInfoForSingleOp(const KernelRunnerPtr &kernel_actor, size_t index) {
  const auto &cur_input_kernel_tensors = kernel_actor->input_kernel_tensors();
  const auto &cur_output_kernel_tensors = kernel_actor->output_kernel_tensors();
  const auto &cur_workspace_kernel_tensors = kernel_actor->workspace_kernel_tensors();
  CaptureKernelInfoList fix_input_kernel_infos;
  CaptureKernelInfoList fix_output_kernel_infos;
  CaptureKernelInfoList fix_workspace_kernel_infos;

  fix_input_kernel_infos.reserve(cur_input_kernel_tensors.size());
  fix_output_kernel_infos.reserve(cur_output_kernel_tensors.size());
  fix_workspace_kernel_infos.reserve(cur_workspace_kernel_tensors.size());

  for (size_t i = 0; i < cur_input_kernel_tensors.size(); ++i) {
    fix_input_kernel_infos.emplace_back(std::make_shared<CaptureKernelInfo>(
      cur_input_kernel_tensors[i]->device_ptr(), cur_input_kernel_tensors[i]->size(),
      cur_input_kernel_tensors[i]->GetShape()->Clone()));
  }
  for (size_t i = 0; i < cur_output_kernel_tensors.size(); ++i) {
    fix_output_kernel_infos.emplace_back(std::make_shared<CaptureKernelInfo>(
      cur_output_kernel_tensors[i]->device_ptr(), cur_output_kernel_tensors[i]->size(),
      cur_output_kernel_tensors[i]->GetShape()->Clone()));
  }
  for (size_t i = 0; i < cur_workspace_kernel_tensors.size(); ++i) {
    fix_workspace_kernel_infos.emplace_back(std::make_shared<CaptureKernelInfo>(
      cur_workspace_kernel_tensors[i]->device_ptr(), cur_workspace_kernel_tensors[i]->size(),
      cur_workspace_kernel_tensors[i]->GetShape()->Clone()));
  }
  fix_single_op_input_info_[shape_key_][std::make_pair(kernel_actor, index)] = fix_input_kernel_infos;
  fix_single_op_output_info_[shape_key_][std::make_pair(kernel_actor, index)] = fix_output_kernel_infos;
  fix_single_op_workspace_info_[shape_key_][std::make_pair(kernel_actor, index)] = fix_workspace_kernel_infos;
}

void GraphCaptureManager::RecoverInfoForSingleOp(const KernelRunnerPtr &kernel_actor, size_t index) {
  auto kernel_with_idx = std::make_pair(kernel_actor, index);
  const auto &cur_fix_input_kernel_tensor_info = fix_single_op_input_info_[shape_key_][kernel_with_idx];
  const auto &cur_fix_output_kernel_tensor_info = fix_single_op_output_info_[shape_key_][kernel_with_idx];
  const auto &cur_fix_workspace_kernel_tensor_info = fix_single_op_workspace_info_[shape_key_][kernel_with_idx];
  for (size_t i = 0; i < kernel_actor->input_kernel_tensors().size(); ++i) {
    if (IsNonFixedInputInReplay(kernel_actor, i)) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "skip fetch kv_cache and weight index: " << i;
      continue;
    }
    if (fix_network_input_for_replay_single_op_[shape_key_][kernel_with_idx][i] == nullptr) {
      kernel_actor->input_kernel_tensors()[i]->set_device_ptr(cur_fix_input_kernel_tensor_info[i]->device_ptr);
      kernel_actor->input_kernel_tensors()[i]->SetShape(cur_fix_input_kernel_tensor_info[i]->shape);
      kernel_actor->input_kernel_tensors()[i]->set_size(cur_fix_input_kernel_tensor_info[i]->size);
    } else {
      kernel_actor->SetInputDeviceTensor(fix_network_input_for_replay_single_op_[shape_key_][kernel_with_idx][i], i);
    }
  }
  for (size_t i = 0; i < kernel_actor->output_kernel_tensors().size(); ++i) {
    kernel_actor->output_kernel_tensors()[i]->set_device_ptr(cur_fix_output_kernel_tensor_info[i]->device_ptr);
    kernel_actor->output_kernel_tensors()[i]->SetShape(cur_fix_output_kernel_tensor_info[i]->shape);
    kernel_actor->output_kernel_tensors()[i]->set_size(cur_fix_output_kernel_tensor_info[i]->size);
  }
  for (size_t i = 0; i < kernel_actor->workspace_kernel_tensors().size(); ++i) {
    kernel_actor->workspace_kernel_tensors()[i]->set_device_ptr(cur_fix_workspace_kernel_tensor_info[i]->device_ptr);
    kernel_actor->workspace_kernel_tensors()[i]->SetShape(cur_fix_workspace_kernel_tensor_info[i]->shape);
    kernel_actor->workspace_kernel_tensors()[i]->set_size(cur_fix_workspace_kernel_tensor_info[i]->size);
  }
}

void GraphCaptureManager::ResetInfoForSingleOp(const std::vector<KernelRunnerPtr> &kernel_runners) {
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; ++i) {
    auto &executor = executors_[i];
    if (executor.first != CAPTURE_GRAPH) {
      auto &kernel_runner = kernel_runners[executor.second];
      auto kernel_with_idx = std::make_pair(kernel_runner, executor.second);
      const auto &is_output_kernels = kernel_runner->is_output_kernel();
      for (size_t j = 0; j < kernel_runner->input_kernel_tensors().size(); ++j) {
        if (IsNonFixedInputInReplay(kernel_runner, j)) {
          MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "skip free kv_cache and weight index: " << i;
          continue;
        }
        if (fix_network_input_for_replay_single_op_[shape_key_][kernel_with_idx][j] == nullptr) {
          kernel_runner->input_kernel_tensors()[j]->set_device_ptr(nullptr);
        } else {
          kernel_runner->input_launch_tensors()[j] = nullptr;
          kernel_runner->input_kernel_tensors()[j] = nullptr;
          kernel_runner->input_kernel_tensors_for_infer()[j] = nullptr;
        }
      }
      for (size_t j = 0; j < kernel_runner->output_kernel_tensors().size(); ++j) {
        if (!is_output_kernels[j]) {
          kernel_runner->output_kernel_tensors()[j]->set_device_ptr(nullptr);
        }
      }
      for (size_t j = 0; j < kernel_runner->workspace_kernel_tensors().size(); ++j) {
        kernel_runner->workspace_kernel_tensors()[j]->set_device_ptr(nullptr);
      }
    }
  }
}

void GraphCaptureManager::Finalize() {
  capture_graph_ = nullptr;
  if (!capture_graphs_.empty()) {
    capture_graphs_.clear();
  }
  if (!fixed_addrs_for_update_.empty()) {
    fixed_addrs_for_update_.clear();
  }
  if (!fixed_addrs_for_set_inputs_.empty()) {
    fixed_addrs_for_set_inputs_.clear();
  }
  if (!weight_kv_addrs_.empty()) {
    weight_kv_addrs_.clear();
  }
  if (!fix_single_op_input_info_.empty()) {
    fix_single_op_input_info_.clear();
  }
  if (!fix_single_op_output_info_.empty()) {
    fix_single_op_output_info_.clear();
  }
  if (!fix_single_op_workspace_info_.empty()) {
    fix_single_op_workspace_info_.clear();
  }
  if (!fix_replay_graph_output_info_.empty()) {
    fix_replay_graph_output_info_.clear();
  }
  if (!fix_network_input_for_replay_single_op_.empty()) {
    fix_network_input_for_replay_single_op_.clear();
  }
  if (!recorded_kernel_output_for_graph_output_.empty()) {
    recorded_kernel_output_for_graph_output_.clear();
  }
}
}  // namespace runtime
}  // namespace mindspore
