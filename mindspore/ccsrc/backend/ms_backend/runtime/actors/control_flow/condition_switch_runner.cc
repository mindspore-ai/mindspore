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

#include "backend/ms_backend/runtime/actors/control_flow/condition_switch_runner.h"
#include <utility>
#include "backend/ms_backend/runtime/actors/control_flow/condition_gather_runner.h"
#include "backend/ms_backend/runtime/graph_executor/pipeline/runtime_pipeline.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/mem_use_analyzer.h"

namespace mindspore {
namespace runtime {
ConditionSwitchRunner::ConditionSwitchRunner(const std::string &name, const CNodePtr &kernel,
                                             const DeviceContext *device_context, const AID &memory_manager_aid,
                                             const AID *debug_aid, const AID *recorder_aid,
                                             GraphExecutionStrategy strategy,
                                             const std::set<size_t> &modifiable_ref_input_indexes,
                                             const std::set<size_t> &modifiable_ref_output_indexes,
                                             const KernelTransformType &type)
    : KernelRunner(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                   modifiable_ref_input_indexes, modifiable_ref_output_indexes, type) {
  need_wait_pipeline_ = true;
}

void ConditionSwitchRunner::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != runtime::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  kernel_info_ = dynamic_cast<KernelInfo *>(kernel_->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info_);
  kernel_mod_ = kernel_info_->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  const auto &output_kernel_tensors = kernel_info_->output_kernel_tensor_list();
  for (size_t i = 0; i < output_kernel_tensors.size(); ++i) {
    auto &output_kernel_tensor = output_kernel_tensors[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_address = output_kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(output_address);
    if (output_address->stream_id() != kernel_info_->stream_id()) {
      MS_LOG(DEBUG) << "Output address : " << output_address << " stream id :" << output_address->stream_id()
                    << " is not equal kernel info stream id : " << kernel_info_->stream_id() << ".";
    }
    (void)output_kernel_tensors_.emplace_back(output_kernel_tensor);
    (void)output_launch_tensors_.emplace_back(output_kernel_tensor.get());
  }

  real_input_num_ = common::AnfAlgo::GetInputTensorNum(kernel_);
  InitIsMonadInput();
  InitInputInfo();

  for (size_t index : output_free_index_) {
    if (index >= output_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << index << " output size:" << output_kernel_tensors_.size()
                        << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(output_kernel_tensors_[index]);
  }

  if (!kernel_->HasAttr(kInlineSubGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get inline graph name by actor:" << GetAID();
  }
  const auto &inline_sub_graph_names = kernel_->GetAttr(kInlineSubGraphName);
  MS_EXCEPTION_IF_NULL(inline_sub_graph_names);
  MS_LOG(DEBUG) << "inline sub graph name:" << inline_sub_graph_names->ToString() << " for actor:" << GetAID();
  if (!inline_sub_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid input subgraph name:" << inline_sub_graph_names->ToString()
                      << " for actor:" << GetAID();
  }
  const auto &tuple_name = inline_sub_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  for_each(tuple_name->value().begin(), tuple_name->value().end(),
           [this](const auto &value) { branch_names_.emplace_back(GetValue<std::string>(value)); });
  MS_LOG(DEBUG) << "Branch names:" << branch_names_ << " for actor:" << GetAID();
}

void ConditionSwitchRunner::UpdateRefDeviceAddress(OpContext<KernelTensor> *const context, bool increase_ref_count) {
  if (input_kernel_tensors_.size() != output_kernel_tensors_.size() + 1) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size:" << input_kernel_tensors_.size()
                      << " and output device tensor size:" << output_kernel_tensors_.size()
                      << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < output_kernel_tensors_.size(); ++i) {
    if (input_kernel_tensors_[i + 1] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input device tensor index:" << i + 1 << " for actor:" << GetAID();
    }
    if (output_kernel_tensors_[i] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input device tensor index:" << i + 1 << " for actor:" << GetAID();
    }
    auto input_device_tensor = input_kernel_tensors_[i + 1]->device_address().get();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto output_device_tensor = output_kernel_tensors_[i]->device_address().get();
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    output_kernel_tensors_[i]->set_pointer_ref_count(input_kernel_tensors_[i + 1].get());

    // Check and synchronize device type
    auto input_device_type = input_device_tensor->GetDeviceType();
    auto output_device_type = output_device_tensor->GetDeviceType();
    if (input_device_type != output_device_type) {
      MS_LOG(DEBUG) << "Adjust device type from " << output_device_type << " to " << input_device_type << " for tensor "
                    << output_kernel_tensors_[i]->ToString();
      output_device_tensor->SetDeviceType(input_device_type);
    }

    output_kernel_tensors_[i]->IncreaseNewRefCount(GetAID().Name());
    MS_LOG(DEBUG) << "Actor:" << GetAID() << " increase new ref count:" << output_kernel_tensors_[i]->new_ref_count()
                  << " and set ref kernel tensor:" << output_kernel_tensors_[i]->ToString()
                  << " ref input kernel tensor:" << input_kernel_tensors_[i + 1]->ToString();
  }
}

void ConditionSwitchRunner::ExecuteInferShapeTask(OpContext<KernelTensor> *const context, bool high_perf) {
  MS_EXCEPTION_IF_NULL(kernel_);
  MS_LOG(DEBUG) << "Begin InferShape for kernel: " << kernel_->fullname_with_scope();

  if (EnableRuntimeNewPipeline()) {
    auto resize_task = [context, this, high_perf]() {
      KernelAsyncResizeActor::GetInstance()->ResizeKernelModV2(context, this, high_perf);
    };
    RuntimePipeline::GetInstance().resize_queue()->Push(std::move(resize_task));
  } else {
    Async(kernel_async_resize_aid_, &KernelAsyncResizeActor::ResizeKernelModV2, context, this, high_perf);
  }
  MS_LOG(DEBUG) << "End InferShape for kernel: " << kernel_->fullname_with_scope();
}

void ConditionSwitchRunner::ExecuteResizeKernelModTask(OpContext<KernelTensor> *const context, bool) {
  if (EnableRuntimeNewPipeline()) {
    auto launch_task = [context, this]() { KernelAsyncLaunchActor::GetInstance()->LaunchKernelV2(context, this); };
    RuntimePipeline::GetInstance().launch_queue()->Push(std::move(launch_task));
  } else {
    Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernelV2, context, this);
  }
}

void ConditionSwitchRunner::UpdateMemoryFreeList(OpContext<KernelTensor> *const context) {
  // Set input device address to memory free list by free index.
  for (size_t free_list_index = 0; free_list_index < input_free_index_.size(); ++free_list_index) {
    size_t input_list_index = input_free_index_[free_list_index];
    if (free_list_index >= new_memory_free_list_.size() || input_list_index >= input_kernel_tensors_.size() ||
        input_list_index >= pre_input_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid free position:" << free_list_index
                        << " free list size:" << new_memory_free_list_.size() << " or input index:" << input_list_index
                        << " input size:" << input_kernel_tensors_.size()
                        << " pre input size:" << pre_input_kernel_tensors_.size() << " for actor:" << GetAID();
    }
    new_memory_free_list_[free_list_index] =
      (pre_input_kernel_tensors_[input_list_index] == nullptr ? input_kernel_tensors_[input_list_index]
                                                              : pre_input_kernel_tensors_[input_list_index]);
    MS_LOG(DEBUG) << "Add new memory free list for input index:" << input_list_index
                  << " input kernel tensor:" << input_kernel_tensors_[input_list_index]
                  << " and pre input kernel tensor:" << pre_input_kernel_tensors_[input_list_index]
                  << " for kernel actor:" << GetAID();
  }
}

void ConditionSwitchRunner::ExecuteLaunchKernelTask(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(kernel_);
  MS_LOG(DEBUG) << "Begin launch kernel: " << kernel_->fullname_with_scope();

  if (enable_hyper_offload_slide_) {
    MemUseAnalyzer::GetInstance().LaunchTaskBefore(this, device_contexts_[0]);
  }

  if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
    MS_LOG(INFO) << "Run failed and early stop.";
    return;
  }
  MS_EXCEPTION_IF_NULL(input_kernel_tensors_[0]);
  bool index = input_kernel_tensors_[0]->GetValueWithCheck<bool>();
  if (IsSkippedLaunch()) {
    index = true;
  }
  MS_LOG(DEBUG) << "Index:" << index << " for actor:" << GetAID();
  if (index >= branch_names_.size()) {
    std::string error_info = "Invalid index:" + std::to_string(index) +
                             " and branch size:" + std::to_string(branch_names_.size()) +
                             " for actor:" + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
  }
  MS_EXCEPTION_IF_NULL(gather_branch_name_);
  *gather_branch_name_ = branch_names_[index];
  MS_EXCEPTION_IF_NULL(branch_flags_);
  branch_flags_.get()[index] = true;
  MS_LOG(DEBUG) << "Enable flag:" << &(branch_flags_.get()[index]) << " by index:" << index
                << " branch name:" << branch_names_[index] << " in actor:" << GetAID();
  new_memory_free_list_.clear();
  for (size_t input_index : input_free_index_) {
    if (input_index >= input_kernel_tensors_.size() || input_kernel_tensors_[input_index] == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get input device tensor index:" << input_index
                        << " total input size:" << input_kernel_tensors_.size()
                        << " for node:" << kernel_->DebugString() << " for actor:" << GetAID();
    }
    auto free_kernel_tensor =
      (pre_input_kernel_tensors_[input_index] == nullptr ? input_kernel_tensors_[input_index]
                                                         : pre_input_kernel_tensors_[input_index]);
    new_memory_free_list_.emplace_back(free_kernel_tensor);
    MS_LOG(DEBUG) << "Add decrease new ref count for kernel tensor:" << free_kernel_tensor->ToString()
                  << " in actor:" << GetAID();
    pre_input_kernel_tensors_[input_index] = nullptr;
  }
  if (branch_output_free_index_.find(branch_names_[index]) != branch_output_free_index_.end()) {
    for (size_t output_index : branch_output_free_index_[branch_names_[index]]) {
      if (output_index >= output_kernel_tensors_.size() || output_kernel_tensors_[output_index] == nullptr) {
        MS_LOG(EXCEPTION) << "Invalid output device tensor index:" << output_index
                          << "total size:" << output_kernel_tensors_.size() << " for actor:" << GetAID();
      }
      new_memory_free_list_.emplace_back(output_kernel_tensors_[output_index]);
      MS_LOG(DEBUG) << "Add decrease new ref count for kernel tensor:" << output_kernel_tensors_[output_index]
                    << " in actor:" << GetAID();
    }
  }

  if (new_memory_free_list_.size() > 0) {
    SendMemoryFreeReq(context);
  }
  MS_LOG(DEBUG) << "End launch kernel: " << kernel_->fullname_with_scope();

  if (enable_hyper_offload_slide_) {
    MemUseAnalyzer::GetInstance().LaunchTaskAfter(this, device_contexts_[0]);
  }
}

void ConditionSwitchRunner::ExecuteLaunchKernelTaskHP(OpContext<KernelTensor> *const context) {
  ExecuteLaunchKernelTask(context);
}

void ConditionSwitchRunner::CollectMemoryFreeList(size_t index) {
  memory_free_list_.clear();
  memory_free_list_.insert(memory_free_list_.end(), input_kernel_tensors_.begin(), input_kernel_tensors_.end());
  memory_free_list_.insert(memory_free_list_.end(), input_kernel_tensors_.begin() + 1, input_kernel_tensors_.end());
  for (size_t i = 0; i < branch_origin_ref_count_.size(); ++i) {
    if (i == index) {
      continue;
    }
    if (branch_origin_ref_count_[i].size() + 1 != input_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid origin ref count size:" << branch_origin_ref_count_[i]
                        << " and input size:" << input_kernel_tensors_.size() << " for actor:" << GetAID();
    }
    MS_LOG(DEBUG) << "Free memory for branch:" << i << " for actor:" << GetAID();
    for (size_t j = 0; j < branch_origin_ref_count_[i].size(); ++j) {
      std::fill_n(back_inserter(memory_free_list_), branch_origin_ref_count_[i][j], input_kernel_tensors_[j + 1]);
    }
  }
}

void ConditionSwitchRunner::FetchParameterInput(OpContext<KernelTensor> *const context) {
  // Fetch parameter input tensor from graph parameter store.
  if (!enable_input_optimize_) {
    return;
  }

  for (auto &parameter_index : parameter_indexs_) {
    auto kernel_tensor = FetchParameter(parameter_index.second, GetAID());
    if (kernel_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get graph parameter store failed: " + parameter_index.second.first.first->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    if (parameter_index.first >= input_kernel_tensors_.size()) {
      std::string error_info = "The input index is out of range, need:" + std::to_string(parameter_index.first) +
                               " current:" + std::to_string(input_kernel_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_kernel_tensors_[parameter_index.first] = kernel_tensor;
  }
}
}  // namespace runtime
}  // namespace mindspore
