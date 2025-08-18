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

#include "backend/ms_backend/runtime/actors/control_flow/condition_gather_runner.h"
#include <utility>
#include <set>
#include <string>
#include <memory>
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "backend/ms_backend/runtime/graph_executor/pipeline/runtime_pipeline.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/mem_use_analyzer.h"

namespace mindspore {
namespace runtime {
ConditionGatherRunner::ConditionGatherRunner(const std::string &name, const CNodePtr &kernel,
                                             const DeviceContext *device_context, const AID &memory_manager_aid,
                                             const AID *debug_aid, const AID *recorder_aid,
                                             GraphExecutionStrategy strategy,
                                             const std::set<size_t> &modifiable_ref_input_indexes,
                                             const std::set<size_t> &modifiable_ref_output_indexes,
                                             const KernelTransformType &type)
    : KernelRunner(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                   modifiable_ref_input_indexes, modifiable_ref_output_indexes, type) {}

ConditionGatherRunner::~ConditionGatherRunner() {
  for_each(need_clean_ptr_device_addresses_.begin(), need_clean_ptr_device_addresses_.end(),
           [](const device::DeviceAddressPtr &device_address) { device_address->set_ptr(nullptr); });
}

void ConditionGatherRunner::ExecuteInferShapeTask(OpContext<KernelTensor> *const context, bool high_perf) {
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

void ConditionGatherRunner::ExecuteResizeKernelModTask(OpContext<KernelTensor> *const context, bool) {
  if (EnableRuntimeNewPipeline()) {
    auto launch_task = [context, this]() { KernelAsyncLaunchActor::GetInstance()->LaunchKernelV2(context, this); };
    RuntimePipeline::GetInstance().launch_queue()->Push(std::move(launch_task));
  } else {
    Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernelV2, context, this);
  }
}

void ConditionGatherRunner::ExecuteLaunchKernelTask(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(kernel_);
  MS_LOG(DEBUG) << "Begin launch kernel: " << kernel_->fullname_with_scope();

  if (enable_hyper_offload_slide_) {
    MemUseAnalyzer::GetInstance().LaunchTaskBefore(this, device_contexts_[0]);
  }

  new_memory_free_list_.clear();
  for (size_t i = 0; i < branch_names_.size(); ++i) {
    branch_flags_.get()[i] = false;
  }
  if (input_kernel_tensors_.size() != output_kernel_tensors_.size() * branch_names_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size:" << input_kernel_tensors_.size()
                      << " and output device tensor size:" << output_kernel_tensors_.size()
                      << " branch name size:" << branch_names_ << " for actor:" << GetAID();
  }
  // Current branch name is set by the condition switch actor, it is used to make sure the real output index.
  auto iter = std::find(branch_names_.begin(), branch_names_.end(), current_branch_name_);
  if (iter == branch_names_.end()) {
    MS_LOG(EXCEPTION) << "Invalid branch name :" << current_branch_name_ << " all branch name:" << branch_names_.size()
                      << " for actor:" << GetAID();
  }
  size_t index = LongToSize(iter - branch_names_.begin());

  // Collect the device address should be freed.
  for (size_t input_index : input_free_index_) {
    if (input_index < index * output_kernel_tensors_.size() ||
        input_index >= (index + 1) * output_kernel_tensors_.size()) {
      continue;
    }
    if (input_kernel_tensors_[input_index] == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get input device tensor index:" << input_index
                        << " for node:" << kernel_->DebugString() << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(input_kernel_tensors_[input_index]);
    MS_LOG(DEBUG) << "Add decrease new ref count for device address:" << input_kernel_tensors_[input_index]
                  << " in actor:" << GetAID();
  }

  for (size_t output_index : output_free_index_) {
    if (output_index >= output_kernel_tensors_.size() || output_kernel_tensors_[output_index] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid output device tensor index:" << output_index
                        << "total size:" << output_kernel_tensors_.size() << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(output_kernel_tensors_[output_index]);
    MS_LOG(DEBUG) << "Add decrease new ref count for device address:" << output_kernel_tensors_[output_index]
                  << " in actor:" << GetAID();
  }
  if (new_memory_free_list_.size() > 0) {
    SendMemoryFreeReq(context);
  }
  MS_LOG(DEBUG) << "End launch kernel: " << kernel_->fullname_with_scope();

  if (enable_hyper_offload_slide_) {
    MemUseAnalyzer::GetInstance().LaunchTaskAfter(this, device_contexts_[0]);
  }
}

void ConditionGatherRunner::ExecuteLaunchKernelTaskHP(OpContext<KernelTensor> *const context) {
  ExecuteLaunchKernelTask(context);
}

void ConditionGatherRunner::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != runtime::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  MS_EXCEPTION_IF_NULL(kernel_);
  if (!kernel_->HasAttr(kAttrBranchOutputNum)) {
    MS_LOG(EXCEPTION) << "Failed to get branch output num by actor:" << GetAID();
  }
  const auto &output_value = kernel_->GetAttr(kAttrBranchOutputNum);
  MS_EXCEPTION_IF_NULL(output_value);
  branch_output_num_ = GetValue<size_t>(output_value);
  MS_LOG(DEBUG) << "branch output num:" << branch_output_num_ << " for actor:" << GetAID();

  if (!kernel_->HasAttr(kAttrBranchGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get inline graph name by actor:" << GetAID();
  }
  const auto &branch_graph_names = kernel_->GetAttr(kAttrBranchGraphName);
  MS_EXCEPTION_IF_NULL(branch_graph_names);
  MS_LOG(DEBUG) << "Branch graph name:" << branch_graph_names->ToString() << " for actor:" << GetAID();
  if (!branch_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid branch group name:" << branch_graph_names->ToString() << " for actor:" << GetAID();
  }
  const auto &tuple_name = branch_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  std::for_each(tuple_name->value().begin(), tuple_name->value().end(),
                [this](const auto &value) { branch_names_.emplace_back(GetValue<std::string>(value)); });
  MS_LOG(DEBUG) << "Branch names:" << branch_names_ << " for actor:" << GetAID();

  size_t input_num = branch_output_num_ * branch_names_.size();
  input_launch_tensors_.resize(input_num);
  pre_input_kernel_tensors_.resize(input_num);
  input_kernel_tensors_.resize(input_num);
  input_kernel_tensors_for_infer_.resize(input_num);
  memory_free_list_.resize(input_num);

  kernel_info_ = dynamic_cast<KernelInfo *>(kernel_->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info_);
  kernel_mod_ = kernel_info_->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  const auto &output_kernel_tensors = kernel_info_->output_kernel_tensor_list();
  const auto &somas_outputs = kernel_info_->somas_output_result();
  if (output_kernel_tensors.size() != somas_outputs.size()) {
    MS_LOG(DEBUG) << "Invalid output address size:" << output_kernel_tensors.size()
                  << " and somas output size:" << somas_outputs.size() << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < output_kernel_tensors.size(); ++i) {
    auto &output_kernel_tensor = output_kernel_tensors[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto &output_address = output_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(output_address);
    if (output_address->stream_id() != kernel_info_->stream_id()) {
      MS_LOG(DEBUG) << "Output address : " << output_address << " stream id :" << output_address->stream_id()
                    << " is not equal kernel info stream id : " << kernel_info_->stream_id() << ".";
    }
    (void)output_kernel_tensors_.emplace_back(output_kernel_tensor);
    // The output taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, i)) {
      // Somas outputs use the info of kernelMod, and output address use the info of device address.
      if (somas_outputs[i].second < output_address->GetSize()) {
        MS_LOG(DEBUG) << GetAID().Name() << " check somas size warning, output index:" << i
                      << " somas aligned size:" << somas_outputs[i].second
                      << " is smaller than address size:" << output_address->GetSize();
      }
      // Used to keep graph output address when somas block memory free, and reused by the ref conut in other graphs.
      if (somas_graph_output_indexes_.count(i) > 0) {
        MS_LOG(DEBUG) << "Somas keep output device address:" << output_address << " ptr:" << output_address->GetPtr();
        MS_EXCEPTION_IF_NULL(somas_info_);
        (void)somas_info_->InsertGraphOutputInfo(output_address.get(), somas_outputs[i].first, somas_outputs[i].second);
        output_address->set_from_mem_pool(true);
        need_clean_ptr_device_addresses_.emplace_back(output_address);
      }
    }
  }

  for (size_t i = 0; i < input_num; ++i) {
    const auto &input_kernel_tensor = AnfAlgo::GetPrevNodeOutputKernelTensor(kernel_, i, false);
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    const auto &input_device_tensor = input_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    (void)real_input_data_infos_.emplace_back(
      std::make_shared<InputDataInfo>(input_kernel_tensor->format(), input_kernel_tensor->GetShapeVector(),
                                      input_device_tensor->GetSize(), input_kernel_tensor->dtype_id()));
  }

  for (size_t index : input_free_index_) {
    if (index >= input_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << index << " output size:" << input_kernel_tensors_.size()
                        << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(input_kernel_tensors_[index]);
  }
  for (size_t index : output_free_index_) {
    if (index >= output_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << index << " output size:" << output_kernel_tensors_.size()
                        << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(output_kernel_tensors_[index]);
  }

  if (output_kernel_tensors_.size() * branch_names_.size() != input_kernel_tensors_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size:" << input_kernel_tensors_.size()
                      << " branch size:" << branch_names_.size() << " and output size:" << output_kernel_tensors_.size()
                      << " for actor:" << GetAID();
  }
}

void ConditionGatherRunner::UpdateRefDeviceAddress(OpContext<KernelTensor> *const context, bool increase_ref_count) {
  if (input_kernel_tensors_.size() != output_kernel_tensors_.size() * branch_names_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size:" << input_kernel_tensors_.size()
                      << " and output device tensor size:" << output_kernel_tensors_.size()
                      << " branch name size:" << branch_names_ << " for actor:" << GetAID();
  }
  auto iter = std::find(branch_names_.begin(), branch_names_.end(), current_branch_name_);
  if (iter == branch_names_.end()) {
    MS_LOG(EXCEPTION) << "Invalid branch name :" << current_branch_name_ << " all branch name:" << branch_names_.size()
                      << " for actor:" << GetAID();
  }

  // Actor output should be ref to the current branch.
  size_t index = LongToSize(iter - branch_names_.begin());
  for (size_t i = 0; i < output_kernel_tensors_.size(); ++i) {
    size_t input_index = i + index * output_kernel_tensors_.size();
    if (input_kernel_tensors_[input_index] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input device tensor index:" << input_index << " for actor:" << GetAID();
    }
    if (output_kernel_tensors_[i] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input device tensor index:" << input_index << " for actor:" << GetAID();
    }

    const auto &somas_outputs = kernel_info_->somas_output_result();
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, i) && i < somas_outputs.size() &&
        somas_outputs[i].second > 0 && somas_graph_output_indexes_.count(i) > 0) {
      MS_LOG(DEBUG) << "Skip set ref for output index:" << i << " for actor:" << GetAID();
      continue;
    }

    MS_EXCEPTION_IF_NULL(output_kernel_tensors_[i]->device_address());
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[input_index]->device_address());
    output_kernel_tensors_[i]->set_tensor_storage_info(input_kernel_tensors_[input_index]->tensor_storage_info());
    output_kernel_tensors_[i]->set_pointer_ref_count(input_kernel_tensors_[input_index].get());
    output_kernel_tensors_[i]->IncreaseNewRefCount(GetAID().Name());
    MS_LOG(DEBUG) << "Actor:" << GetAID() << " increase new ref count:" << output_kernel_tensors_[i]->new_ref_count()
                  << " and set ref kernel tensor:" << output_kernel_tensors_[i]->ToString()
                  << " ref input kernel tensor:" << input_kernel_tensors_[input_index]->ToString();
  }
  new_memory_free_list_.resize(input_free_index_.size() + output_free_index_.size());
}
}  // namespace runtime
}  // namespace mindspore
