/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/kernel_actor.h"

#include <mutex>
#include <algorithm>

#include "runtime/device/multi_stream_controller.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/recorder_actor.h"
#include "runtime/graph_scheduler/actor/debug_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "kernel/framework_utils.h"
#include "mindspore/core/ops/framework_ops.h"
#include "utils/compile_config.h"

namespace mindspore {
namespace runtime {
namespace {
bool IsSomasEnable(const SomasInfo *somas_info) {
  return ((somas_info != nullptr) && (somas_info->whole_block_size_ != 0));
}

void CheckDryRun(const CNodePtr &kernel_) {
  static const bool is_dry_run_mode = (common::GetEnv(kSimulationLevel) == kSimulationLevelCompileKernel);
  static auto enabled_profile = common::GetCompileConfig("COMPILE_PROFILE") == "1";
  if (is_dry_run_mode && !enabled_profile) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_)
      << "The dry run mode can not support dynamic shape graph which contains value depend kernel:"
      << kernel_->fullname_with_scope()
      << ", launch kernel is skipped for dry run mode, which leads to fail to GetValue for infer "
         "shape of these value depend kernel. You can only simulate compile graph and not do "
         "InferShape and Resize by `export MS_SIMULATION_LEVEL=0` instead.";
  }
}
}  // namespace

using distributed::collective::CollectiveManager;
using distributed::recovery::RecoveryContext;

void KernelActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

  MS_EXCEPTION_IF_NULL(kernel_);
  real_input_num_ = common::AnfAlgo::GetInputTensorNum(kernel_);
  kernel_info_ = dynamic_cast<KernelInfo *>(kernel_->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info_);
  kernel_mod_ = kernel_info_->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  is_dynamic_value_ = common::AnfAlgo::IsDynamicValue(kernel_);
  if (is_dynamic_shape_ && IsSomasEnable(somas_info_)) {
    MS_LOG(EXCEPTION) << "Not support the somas for the dynamic shape: " << GetAID().Name();
  }
  is_dynamic_type_ = common::AnfAlgo::IsAnyTypeOutput(kernel_);
  has_dynamic_ = is_dynamic_shape_ || is_dynamic_type_ || is_dynamic_value_;

  if (is_dynamic_value_ && (is_dynamic_shape_ || is_dynamic_type_)) {
    CheckDryRun(kernel_);
  }

  // Check whether the kernel has input node which is a computed depend kernel.
  launch_ignored_inputs_ = kernel_mod_->GetLaunchIgnoredInputAddressIdx();

  stream_ = device_contexts_[0]->device_res_manager_->GetStream(kernel_info_->stream_id());
  // Init the device tensors and kernel launch info.
  InitInputInfo();
  InitOutputInfo();
  InitWorkspaceInfo();

  // Init the output data.
  InitOutputData();
  if (output_data_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "The output data size is wrong: " << GetAID().Name();
  }
  size_t output_data_index = 0;
  for (auto &data_arrow : output_data_arrows_) {
    auto data = output_data_[output_data_index].first.get();
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) >= output_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID().Name();
    }
    data->data_ = output_device_tensors_[IntToSize(data_arrow->from_output_index_)];
    ++output_data_index;
  }

  auto device_context = device_contexts_[0];
  // cpu kernel does not need multi stream process, and gpu kernel has not adapt it currently.
  if (device_context->GetDeviceType() == device::DeviceType::kCPU ||
      device_context->GetDeviceType() == device::DeviceType::kGPU) {
    MS_LOG(DEBUG) << "Kernel : " << kernel_->fullname_with_scope() << " device type is "
                  << device_context->GetDeviceType() << ", will skip multi stream process.";
    is_multi_stream_process_skipped_ = true;
  }

  // Share pointer of task id on stream with output kernel tensor.
  for (auto &output_kernel_tensor : output_kernel_tensors_) {
    output_kernel_tensor->set_task_id_on_stream(task_id_on_stream_);
  }
  is_stream_recv_actor_ = IsPrimitiveCNode(kernel_, prim::kPrimStreamRecv);
  // kernel_ may be ValueNode<FuncGraph>, skip exception situation.
  auto cnode = kernel_->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }

  // shape depend need kernel is cnode.
  InitShapeDependInfo();

  auto input0 = cnode->input(kAnfPrimitiveIndex);
  if (IsValueNode<FuncGraph>(input0)) {
    MS_LOG(INFO) << "Cnode is not a func graph value node : " << kernel_->fullname_with_scope() << ".";
    return;
  }

  auto multi_stream_safe_value = cnode->GetAttr(kAttrInputMultiStreamSafe);
  if (multi_stream_safe_value != nullptr) {
    is_multi_stream_safe_ = GetValue<bool>(multi_stream_safe_value);
    MS_LOG(DEBUG) << "cnode : " << cnode->DebugString() << " is thread safe.";
  }
}

void KernelActor::InitInputInfo() {
  for (size_t i = 0; i < real_input_num_; ++i) {
    const auto &input_device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel_, i, false);
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    (void)real_input_data_infos_.emplace_back(
      std::make_shared<InputDataInfo>(input_device_tensor->format(), input_device_tensor->host_shape(),
                                      input_device_tensor->GetSize(), input_device_tensor->type_id()));
  }

  copy_input_device_tensors_.resize(real_input_num_);
  input_device_tensors_.resize(real_input_num_);
  input_kernel_tensors_.resize(real_input_num_);
  input_kernel_tensors_for_infer_.resize(real_input_num_);
  for (auto &input_address : input_device_tensors_) {
    (void)memory_free_list_.emplace_back(input_address);
    if (recorder_aid_ != nullptr || debug_aid_ != nullptr) {
      (void)mem_info_.inputs_.emplace_back(std::make_shared<Address>());
    }
  }

  if (EnableKbkSubGraphExecute()) {
    memory_free_list_.clear();
    for (size_t i = 0; i < real_input_num_; ++i) {
      auto input_node_with_idx = common::AnfAlgo::GetPrevNodeOutput(kernel_, i, false);
      MS_EXCEPTION_IF_NULL(input_node_with_idx.first);
      if (!input_node_with_idx.first->isa<CNode>()) {
        continue;
      }

      if (IsSkippedKernelActor(input_node_with_idx.first)) {
        input_node_with_idx = common::AnfAlgo::GetPrevNodeOutput(input_node_with_idx.first, 0, false);
      }

      const auto &input_device_address =
        AnfAlgo::GetMutableOutputAddr(input_node_with_idx.first, input_node_with_idx.second, false);
      MS_EXCEPTION_IF_NULL(input_device_address);
      input_device_tensors_[i] = input_device_address.get();
      input_kernel_tensors_[i] = input_device_tensors_[i]->kernel_tensor().get();
      input_kernel_tensors_for_infer_[i] = input_device_tensors_[i]->kernel_tensor();

      if (!IsSomasEnable(somas_info_)) {
        memory_free_list_.emplace_back(input_device_address.get());
      }
    }
  }
}

void KernelActor::InitOutputInfo() {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  const auto &output_addresses = kernel_info_->output_address_list();
  const auto &somas_outputs = kernel_info_->somas_output_result();
  bool output_need_somas = false;
  for (size_t i = 0; i < output_addresses.size(); ++i) {
    auto &output_address = output_addresses[i];
    MS_EXCEPTION_IF_NULL(output_address);

    if (output_address->stream_id() != kernel_info_->stream_id()) {
      MS_LOG(DEBUG) << "Output address : " << output_address << " stream id :" << output_address->stream_id()
                    << " is not equal kernel info stream id : " << kernel_info_->stream_id() << ".";
    }

    (void)output_device_tensors_.emplace_back(output_address.get());
    (void)output_kernel_tensors_.emplace_back(output_address->kernel_tensor().get());
    MS_LOG(DEBUG) << "Init output[" << i << "] info for node:" << kernel_->fullname_with_scope()
                  << " addr:" << output_address << " type:" << output_address->type_id()
                  << ", kernel tensor addr:" << output_address->kernel_tensor().get()
                  << ", kernel tensor: " << output_address->kernel_tensor()->ToString();
    if (recorder_aid_ != nullptr || debug_aid_ != nullptr) {
      (void)mem_info_.outputs_.emplace_back(std::make_shared<Address>());
    }
    // The output taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, i)) {
      output_address->kernel_tensor()->set_managed_by_somas(true);
      MS_LOG(INFO) << "Device address : " << output_address << ", kernel tensor : " << output_address->kernel_tensor()
                   << " is managed by somas.";
      // Somas outputs use the info of kernelMod, and output address use the info of device address.
      if (somas_outputs[i].second < output_address->GetSize()) {
        MS_LOG(INFO) << GetAID().Name() << " check somas size warning, output index:" << i
                     << " somas aligned size:" << somas_outputs[i].second
                     << " is smaller than address size:" << output_address->GetSize();
      }
      // Used to keep graph output address when somas block memory free, and reused by the ref conut in other graphs.
      if (somas_graph_output_indexes_.count(i) > 0) {
        MS_LOG(DEBUG) << "Somas keep output device address:" << output_address << " ptr:" << output_address->GetPtr();
        (void)somas_info_->InsertGraphOutputInfo(output_address.get(), somas_outputs[i].first, somas_outputs[i].second);
      } else {
        UpdateRefCount(output_address.get(), true);
      }
      output_need_somas = true;
    } else {
      (void)memory_alloc_list_.emplace_back(output_address.get());
      if (output_address->original_ref_count() == SIZE_MAX) {
        max_ref_cnt_output_list_.emplace_back(output_address.get());
      }
      (void)memory_free_list_.emplace_back(output_address.get());
    }
  }

  if (output_need_somas && (!IsSomasEnable(somas_info_))) {
    MS_LOG(EXCEPTION) << "The somas is not enable for: " << GetAID().Name();
  }

  if (IsSomasEnable(somas_info_)) {
    MS_EXCEPTION_IF_CHECK_FAIL((output_device_tensors_.size() >= somas_outputs.size()), "The output num is wrong.");
  }

  for (auto &external_reference_tensor : external_reference_tensors_) {
    (void)memory_free_list_.emplace_back(external_reference_tensor);
  }
}

void KernelActor::InitWorkspaceInfo() {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  // The size of workspace maybe changed in dynamic shape, so put workspace_address in the end of memory_alloc_list_ and
  // memory_free_list_, for the operation of dynamic_shape condition in FetchWorkspaceDeviceTensor.
  const auto &workspace_addresses = kernel_info_->workspace_address_list();
  const auto &somas_workspace = kernel_info_->somas_workspace_result();
  bool workspace_need_somas = false;
  for (size_t i = 0; i < workspace_addresses.size(); ++i) {
    auto &workspace_address = workspace_addresses[i];
    MS_EXCEPTION_IF_NULL(workspace_address);
    (void)workspace_device_tensors_.emplace_back(workspace_address.get());
    (void)workspace_kernel_tensors_.emplace_back(workspace_address->kernel_tensor().get());
    if (recorder_aid_ != nullptr || debug_aid_ != nullptr) {
      (void)mem_info_.workspaces_.emplace_back(std::make_shared<Address>());
    }

    // The workspace taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_workspace, i)) {
      if (somas_workspace[i].second < workspace_address->GetSize()) {
        MS_LOG(INFO) << GetAID().Name() << " check somas size warning, workspace index:" << i
                     << " somas aligned size:" << somas_workspace[i].second
                     << " is smaller than address size:" << workspace_address->GetSize();
      }
      UpdateRefCount(workspace_address.get(), true);
      workspace_need_somas = true;
    } else {
      (void)memory_alloc_list_.emplace_back(workspace_address.get());
      (void)memory_free_list_.emplace_back(workspace_address.get());
    }
  }

  if (workspace_need_somas && (!IsSomasEnable(somas_info_))) {
    MS_LOG(EXCEPTION) << "The somas is not enable for: " << GetAID().Name();
  }

  if (IsSomasEnable(somas_info_)) {
    MS_EXCEPTION_IF_CHECK_FAIL((workspace_device_tensors_.size() >= somas_workspace.size()),
                               "The output num is wrong.");
  }
}

void KernelActor::InitShapeDependInfo() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (enable_infer_boost) {
    return;
  }
  // Shape kernel no need to decrease ref count.
  const auto &only_depend_shape_attr = common::AnfAlgo::GetCNodePrimitiveAttr(kernel_, kAttrOnlyDependShape);
  if (only_depend_shape_attr != nullptr) {
    auto only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
    MS_LOG(INFO) << "Init shape depend info, real_input_num_ : " << real_input_num_
                 << ", only_depend_shape size : " << only_depend_shape.size() << ".";
    for (size_t i = 0; i < only_depend_shape.size(); i++) {
      // shape depend, no need free this device tensor.
      MS_LOG(INFO) << "only_shape_depend[" << i << "] : " << only_depend_shape[i] << ".";
      depend_shape_input_list_.emplace_back(only_depend_shape[i]);
    }
  }
}

void KernelActor::Run(OpContext<DeviceTensor> *const context) {
  try {
    MS_EXCEPTION_IF_NULL(kernel_);
    MS_EXCEPTION_IF_NULL(kernel_->func_graph());
    if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), kernel_->fullname_with_scope(),
                                                     kernel_->func_graph()->ToString());
    }
    FetchInputDeviceTensor(context);

    if (ActorDispatcher::enable_runtime_multi_pipeline()) {
      RunWithMultiPipeline(context);
      return;
    }

    device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    if (has_dynamic_) {
      // Infer shape and resize for dynamic shape case.
      InferAndResize(context);
      FetchOutputDeviceTensor(context);
      FetchWorkspaceDeviceTensor();
    } else {
      FetchOutputDeviceTensor(context);
    }

    // Set the memory address for the tensors which use the somas.
    SetSomasMemory(context);

    if (ActorDispatcher::enable_async_launch_kernel()) {
      RunWithAsyncLaunchKernel(context);
      return;
    }

    if (!memory_alloc_list_.empty()) {
      // Allocate the memory address for other tensors which don't use the somas.
      SendMemoryAllocReq(context);
    }
    OnMemoryAllocFinish(context);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info =
      "#umsg#Kernel error:#umsg#run kernel[" + kernel_->fullname_with_scope() + "] failed, exception: " + e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
  }
}

void KernelActor::RunWithMultiPipeline(OpContext<DeviceTensor> *const context) {
  // 1. Set the memory address for the tensors which use the somas if need.
  SetSomasMemory(context);

  // If the kernel need user data and is dynamic, maybe need input kernel's output user data to infer shape, this value
  // depend case can not handle in KernelTensor auto sync phase currently.
  if (kernel_mod_->need_user_data() && has_dynamic_) {
    MS_LOG(DEBUG) << "Begin wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
    if (!WaitRuntimePipelineFinish(context)) {
      MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
      return;
    }
    MS_LOG(DEBUG) << "End wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
  }

  // 2. Push run task to pipeline.
  // Note: dynamic value or static shape also need push task into infer actor to make sure correct kernel execution
  // order.
  Async(kernel_async_infer_aid_, &KernelAsyncInferActor::InferShape, context, this);

  // The computed depend kernel should wait output shape update after kernel launch.
  if (kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
    MS_LOG(DEBUG) << "Begin wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
    if (!WaitRuntimePipelineFinish(context)) {
      MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
      return;
    }
    MS_LOG(DEBUG) << "End wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
  }

  // 3. Post run.
  EraseInput(context);
  SendOutput(context);
}

void KernelActor::RunWithAsyncLaunchKernel(OpContext<DeviceTensor> *const context) {
  Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernel, context, this);

  if (IsRunningFailed(context)) {
    MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
    return;
  }

  // PostLaunchKernel
  EraseInput(context);
  SendOutput(context);
}

void KernelActor::FetchWorkspaceDeviceTensor() {
  auto workspace_sizes = kernel_mod_->GetWorkspaceSizeList();
  // Resize of workspace_device_tensors_, memory_alloc_list_ and memory_free_list_, because of
  // the dynamic size of workspace.
  if (workspace_device_tensors_.size() > workspace_sizes.size()) {
    size_t size = workspace_device_tensors_.size() - workspace_sizes.size();
    (void)workspace_device_tensors_.erase(workspace_device_tensors_.end() - size, workspace_device_tensors_.end());
    if (recorder_aid_ != nullptr || debug_aid_ != nullptr) {
      (void)mem_info_.workspaces_.erase(mem_info_.workspaces_.end() - size, mem_info_.workspaces_.end());
    }

    MS_EXCEPTION_IF_CHECK_FAIL((memory_alloc_list_.size() >= size), "The memory alloc list size is wrong.");
    MS_EXCEPTION_IF_CHECK_FAIL((memory_free_list_.size() >= size), "The memory free list size is wrong.");
    (void)memory_alloc_list_.erase(memory_alloc_list_.end() - size, memory_alloc_list_.end());
    (void)memory_free_list_.erase(memory_free_list_.end() - size, memory_free_list_.end());
  } else if (workspace_device_tensors_.size() < workspace_sizes.size()) {
    if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
      MS_LOG(ERROR) << "Invalid device context for kernel actor:" + GetAID().Name();
      return;
    }
    for (size_t i = workspace_device_tensors_.size(); i < workspace_sizes.size(); ++i) {
      auto kernel_tensor = std::make_shared<kernel::KernelTensor>(
        nullptr, workspace_sizes[i], Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
        device_contexts_[0]->device_context_key().device_name_, device_contexts_[0]->device_context_key().device_id_);
      kernel_tensor->set_stream_id(kernel_info_->stream_id());
      auto device_address = device_contexts_[0]->device_res_manager_->CreateDeviceAddress(kernel_tensor);
      MS_EXCEPTION_IF_NULL(device_address);
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(kernel_)
                    << " addr:" << device_address;
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel_.get());  // set to kernel_info
      (void)workspace_device_tensors_.emplace_back(device_address.get());
      if (recorder_aid_ != nullptr || debug_aid_ != nullptr) {
        (void)mem_info_.workspaces_.emplace_back(std::make_shared<Address>());
      }
      (void)memory_alloc_list_.emplace_back(device_address.get());
      (void)memory_free_list_.emplace_back(device_address.get());
    }
  }
  // Set workspace address new size
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    workspace_device_tensors_[i]->SetSize(workspace_sizes[i]);
  }

  // Update workspace kernel tensors.
  workspace_kernel_tensors_.resize(workspace_device_tensors_.size());
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    workspace_kernel_tensors_[i] = workspace_device_tensors_[i]->kernel_tensor().get();
  }
}

void KernelActor::SetSomasMemory(OpContext<DeviceTensor> *const context) const {
  if (!IsSomasEnable(somas_info_)) {
    return;
  }

  // Set the memory address for the output tensors which use the somas.
  const auto &somas_outputs = kernel_info_->somas_output_result();
  for (size_t i = 0; i < somas_outputs.size(); ++i) {
    if (somas_outputs[i].second > 0) {
      auto device_ptr = GetSomasDevicePtr(somas_outputs[i].first);
      // In this scenario, the Init function can ensure that the pointer of the relevant operation is not nullptr.
      // In order to perform performance, the pointer validity is not checked here.
      // Check the graph output address need free.
      if (somas_graph_output_indexes_.count(i) && (output_device_tensors_[i]->GetPtr() != nullptr)) {
        MS_LOG(ERROR) << GetAID().Name() << " does not free address for graph output index: " << i;
        device_contexts_[0]->device_res_manager_->FreeMemory(output_device_tensors_[i]);
      }
      MS_LOG(DEBUG) << "Set ptr:" << device_ptr << " to device address:" << output_device_tensors_[i]
                    << " in actor:" << GetAID();
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(),
                                                     device::tracker::MemType::kInSideSomas,
                                                     output_device_tensors_[i]->GetSize(), output_device_tensors_[i]);
      output_device_tensors_[i]->set_ptr(device_ptr);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, output_device_tensors_[i], device_ptr);
    }
  }

  // Set the memory address for the workspace tensors which use the somas.
  const auto &somas_workspace = kernel_info_->somas_workspace_result();
  for (size_t i = 0; i < somas_workspace.size(); ++i) {
    if (somas_workspace[i].second > 0) {
      auto device_ptr = GetSomasDevicePtr(somas_workspace[i].first);
      // In this scenario, the Init function can ensure that the pointer of the relevant operation is not nullptr.
      // In order to perform performance, the pointer validity is not checked here.
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
        AddMemInfo, GetAID().Name(), device::tracker::MemType::kInSideSomas, workspace_device_tensors_[i]->GetSize(),
        workspace_device_tensors_[i]);
      workspace_device_tensors_[i]->set_ptr(device_ptr);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, workspace_device_tensors_[i], device_ptr);
    }
  }
}

void *KernelActor::GetSomasDevicePtr(size_t offset) const {
  // Get the ptr from the whole block.
  if (somas_info_->base_address_ != nullptr) {
    return AddressOffset(somas_info_->base_address_, offset);
  }

  // Get the ptr from the merged blocks.
  auto iter = somas_info_->merged_base_addresses_.upper_bound(offset);
  if (iter == somas_info_->merged_base_addresses_.begin()) {
    MS_LOG(ERROR) << GetAID().Name() << " can't find the merged block for offset: " << offset;
    return nullptr;
  }
  --iter;
  size_t real_offset = offset - iter->first;
  void *real_base_address = iter->second;
  if (real_base_address == nullptr) {
    MS_LOG(ERROR) << GetAID().Name() << " doesn't allocate the merged block base address for offset: " << iter->first;
    return nullptr;
  }
  return AddressOffset(real_base_address, real_offset);
}

void KernelActor::TraceDynamicMemory() {
  for (size_t i = 0; i < output_kernel_tensors_.size(); i++) {
    if (output_device_tensors_[i]->original_ref_count() != SIZE_MAX) {
      const auto &kernel_tensor = output_kernel_tensors_[i];
      MemoryTraceManager::GetInstance().AddKernelMemoryTraceBlock(
        std::make_shared<KernelMemoryTraceBlock>(kernel_, kernel_tensor->device_ptr(), kernel_tensor->size(),
                                                 kOutputMem, i),
        device_contexts_[0]);
    }
  }

  for (size_t i = 0; i < workspace_kernel_tensors_.size(); i++) {
    const auto &kernel_tensor = workspace_kernel_tensors_[i];
    MemoryTraceManager::GetInstance().AddKernelMemoryTraceBlock(
      std::make_shared<KernelMemoryTraceBlock>(kernel_, kernel_tensor->device_ptr(), kernel_tensor->size(),
                                               kWorkspaceMem, i),
      device_contexts_[0]);
  }
}

void KernelActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  if (device_contexts_[0]->device_res_manager_->swap_manager() != nullptr) {
    device_contexts_[0]->device_res_manager_->swap_manager()->SetSwappableBeforeMemAllocate(input_device_tensors_,
                                                                                            output_device_tensors_);
    MS_EXCEPTION_IF_NULL(kernel_info_);
    for (const auto &out_in : kernel_info_->out_in_ref_map()) {
      MS_EXCEPTION_IF_NULL(input_device_tensors_[out_in.second]);
      const auto &ptr = input_device_tensors_[out_in.second]->GetValidPtr(kDefaultStreamIndex);
      if (ptr == nullptr || output_device_tensors_[out_in.first] == nullptr ||
          output_device_tensors_[out_in.first]->GetPtr() != nullptr) {
        continue;
      }
      // Pointer in DeviceAddress which is reference output may not be updated to the same as the reference input
      // which is swapped out.
      MS_LOG(DEBUG) << "Set device ptr of " << out_in.first << "th ref output the same as input " << out_in.second
                    << ": " << ptr;
      output_device_tensors_[out_in.first]->set_ptr(ptr);
    }
  }

  MemoryManagerActor::GetInstance()->AllocateMemory(&memory_alloc_list_, device_contexts_[0], context, GetAID());

  if (ActorDispatcher::enable_trace_dynamic_memory()) {
    if (IsRunningFailed(context)) {
      return;
    }
    TraceDynamicMemory();
  }
}

void KernelActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  if (device_contexts_[0]->device_res_manager_->swap_manager() != nullptr) {
    device_contexts_[0]->device_res_manager_->swap_manager()->SetSwappableBeforeMemFree(
      input_device_tensors_, output_device_tensors_, kernel_info_);
  }
  if (depend_shape_input_list_.empty()) {
    MemoryManagerActor::GetInstance()->FreeMemory(&memory_free_list_, device_contexts_[0], context, GetAID());
  } else {
    MS_LOG(DEBUG) << "depend_shape_input_list size : " << depend_shape_input_list_.size() << ".";
    std::vector<DeviceTensor *> free_list;
    for (size_t i = 0; i < memory_free_list_.size(); i++) {
      const auto device_tensor = memory_free_list_[i];
      if (device_tensor->dynamic_ref_count() == INT32_MAX && device_tensor->ref_count() != SIZE_MAX &&
          i < depend_shape_input_list_.size() && depend_shape_input_list_[i]) {
        MS_LOG(DEBUG) << "Skip memory free for kernel actor : " << kernel_->fullname_with_scope() << " index : " << i
                      << ", device address : " << memory_free_list_[i] << ".";
        continue;
      }
      free_list.emplace_back(memory_free_list_[i]);
    }
    MemoryManagerActor::GetInstance()->FreeMemory(&free_list, device_contexts_[0], context, GetAID());
  }

  // Free the address that is the temp store for kernel input copy.
  for (auto &copy_input_device_tensor : copy_input_device_tensors_) {
    if ((copy_input_device_tensor != nullptr) && (copy_input_device_tensor->GetPtr() != nullptr)) {
      device_contexts_[0]->device_res_manager_->FreeMemory(copy_input_device_tensor.get());
    }
  }
}

void KernelActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  if (IsRunningFailed(context)) {
    MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
    return;
  }
  PreLaunchKernel(context);

  if (debug_aid_ != nullptr) {
    ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugPreLaunch, kernel_, input_device_tensors_,
                              output_device_tensors_, device_contexts_[0], context, &GetAID());
  }

  bool skip_launch = CollectiveManager::instance()->need_reinit() || IsSkippedLaunch(kernel_, nullptr);
  if (!LaunchKernel(context, skip_launch)) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_) << "#umsg#Kernel error:#umsg#Launch kernel failed: " +
                                              kernel_->fullname_with_scope()
                                         << trace::DumpSourceLines(kernel_);
  }

  // Record mem info, because async send may free device info.
  if (recorder_aid_ != nullptr || debug_aid_ != nullptr) {
    SetMemInfoForDebugAndRdr();
  }

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugPostLaunch, kernel_, input_device_tensors_,
                              output_device_tensors_, device_contexts_[0], context, &GetAID());
  }

  PostLaunchKernel(context);
}

void KernelActor::SetMemInfoForDebugAndRdr() {
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    mem_info_.inputs_[i]->addr = input_device_tensors_[i]->GetMutablePtr();
    mem_info_.inputs_[i]->size = input_device_tensors_[i]->GetSize();
  }
  for (size_t i = 0; i < output_device_tensors_.size(); ++i) {
    mem_info_.outputs_[i]->addr = output_device_tensors_[i]->GetMutablePtr();
    mem_info_.outputs_[i]->size = output_device_tensors_[i]->GetSize();
  }
  for (size_t i = 0; i < workspace_device_tensors_.size(); ++i) {
    mem_info_.workspaces_[i]->addr = workspace_device_tensors_[i]->GetMutablePtr();
    mem_info_.workspaces_[i]->size = workspace_device_tensors_[i]->GetSize();
  }
}

void KernelActor::CopyInputDeviceTensor(const OpData<DeviceTensor> *input_data,
                                        OpContext<DeviceTensor> *const context) {
  size_t input_data_index = IntToSize(input_data->index_);
  // The ignored input address that is not used in the kernel launch and no need copy.
  if (!launch_ignored_inputs_.empty() && (std::find(launch_ignored_inputs_.begin(), launch_ignored_inputs_.end(),
                                                    input_data_index) != launch_ignored_inputs_.end())) {
    MS_LOG(DEBUG) << GetAID().Name() << " ignore the input address for input index: " << input_data_index;
    return;
  }
  if (skip_launch_shape_related_op_) {
    return;
  }
  if (input_data_index >= real_input_data_infos_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The input index is of range.");
  }
  auto &real_input_info = real_input_data_infos_[input_data_index];
  if ((input_data->data_->GetDeviceType() == device_contexts_[0]->GetDeviceType()) &&
      AnfAlgo::IsEquivalentFormat(input_data->data_->format(), real_input_info->format_)) {
    return;
  }

  if (!WaitRuntimePipelineFinish(context)) {
    MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
    return;
  }
  if (inputs_continuous_memory_) {
    std::string error_info = GetAID().Name() + " inputs must be continuous memory and can't be copied for index " +
                             std::to_string(input_data_index);
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
  }
  if (input_data_index >= copy_input_device_tensors_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The input index is of range.");
  }
  if (copy_input_device_tensors_[input_data_index] == nullptr) {
    const auto &pre_kernel_tensor = AnfAlgo::GetPrevNodeOutputKernelTensor(kernel_, input_data_index);
    MS_EXCEPTION_IF_NULL(pre_kernel_tensor);
    auto new_kernel_tensor = std::make_shared<kernel::KernelTensor>(
      pre_kernel_tensor->GetShape(), pre_kernel_tensor->GetType(), pre_kernel_tensor->GetValueTrack(), nullptr,
      real_input_info->size_, real_input_info->format_, real_input_info->type_id_, real_input_info->shape_,
      device_contexts_[0]->device_context_key().device_name_, device_contexts_[0]->device_context_key().device_id_);
    MS_EXCEPTION_IF_NULL(new_kernel_tensor);
    auto pre_stream_id = pre_kernel_tensor->stream_id();
    if (pre_stream_id == UINT32_MAX) {
      auto stream_id = kernel_info_->stream_id();
      MS_LOG(DEBUG) << "Rewrite kernel tensor : " << new_kernel_tensor
                    << " stream id with kernel info stream id : " << stream_id << ".";
      new_kernel_tensor->set_stream_id(stream_id);
    } else {
      MS_LOG(DEBUG) << "Rewrite kernel tensor : " << new_kernel_tensor
                    << " stream id with pre kernel tensor stream id : " << pre_stream_id << ".";
      new_kernel_tensor->set_stream_id(pre_stream_id);
    }

    copy_input_device_tensors_[input_data_index] =
      device_contexts_[0]->device_res_manager_->CreateDeviceAddress(new_kernel_tensor);
    MS_EXCEPTION_IF_NULL(copy_input_device_tensors_[input_data_index]);
  }
  auto &new_device_tensor = copy_input_device_tensors_[input_data_index];
  MS_EXCEPTION_IF_NULL(new_device_tensor);

  MS_LOG(DEBUG) << "Prev stream id : " << input_device_tensors_[input_data_index]->stream_id()
                << " new stream id : " << new_device_tensor->stream_id() << ".";
  // Update the input device tensor.
  input_device_tensors_[input_data_index] = new_device_tensor.get();
  input_kernel_tensors_[input_data_index] = input_device_tensors_[input_data_index]->kernel_tensor().get();
  if (is_dynamic_shape_) {
    // Need update shape and size for dynamic shape case.
    input_kernel_tensors_for_infer_[input_data_index] = input_device_tensors_[input_data_index]->kernel_tensor();
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[input_data_index]);
    MS_EXCEPTION_IF_NULL(input_data->data_->kernel_tensor());
    MS_EXCEPTION_IF_NULL(input_data->data_->kernel_tensor()->GetShape());
    input_kernel_tensors_[input_data_index]->SetShape(input_data->data_->kernel_tensor()->GetShape()->Clone());
    input_kernel_tensors_[input_data_index]->set_size(input_data->data_->GetSize());
  }

  device::DynamicMemAllocatorDebugInfo::SetDebugInfo(GetAID().Name(), device::AllocatorType::kKernelOutput,
                                                     input_data_index);
  if (new_device_tensor->GetPtr() == nullptr) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(), device::tracker::MemType::kOther,
                                                   new_device_tensor->GetSize(), new_device_tensor.get());
    if (!device_contexts_[0]->device_res_manager_->AllocateMemory(new_device_tensor.get(), kDefaultStreamIndex)) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy_, *context, *(device_contexts_[0]), GetAID().Name(),
                                                  new_device_tensor->GetSize());
    }
  }

  MS_LOG(INFO) << GetAID().Name() << " the input position:" << input_data_index
               << " copy from device address:" << input_data->data_ << " ptr:" << input_data->data_->GetPtr()
               << ", type:" << input_data->data_->GetDeviceType() << ", format:" << input_data->data_->format()
               << " to device address:" << new_device_tensor.get() << " ptr:" << new_device_tensor->GetPtr()
               << ", type:" << new_device_tensor->GetDeviceType() << ", format:" << new_device_tensor->format();
  // Copy from the real parameter to formal parameter and insert the device tensor copy store.
  if (!Copy(new_device_tensor.get(), input_data->data_)) {
    std::string error_info = "Copy device tensor failed: " + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
  }
  if (modifiable_ref_input_indexes_.count(input_data->index_) > 0) {
    DeviceTensorCopyStore::GetInstance().Insert(new_device_tensor.get(), input_data->data_);
  }
}

void KernelActor::UpdateInputDeviceTensor(const OpData<DeviceTensor> *input_data,
                                          OpContext<DeviceTensor> *const context) {
  size_t input_index = IntToSize(input_data->index_);
  if (input_index >= input_device_tensors_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
      strategy_, (*context),
      "The input index:" + std::to_string(input_index) + " is out of vector size:" +
        std::to_string(input_device_tensors_.size()) + " for kernel:" + kernel_->fullname_with_scope());
  }

  // Update the input device tensor.
  if (input_device_tensors_[input_index] != input_data->data_) {
    input_device_tensors_[input_index] = input_data->data_;
    memory_free_list_[input_index] = input_data->data_;
  }

  // Update the input kernel tensor.
  const auto &kernel_tensor = input_device_tensors_[input_index]->kernel_tensor();
  if (input_kernel_tensors_[input_index] != kernel_tensor.get()) {
    input_kernel_tensors_[input_index] = kernel_tensor.get();
    if (is_dynamic_shape_) {
      input_kernel_tensors_for_infer_[input_index] = kernel_tensor;
    }
  }
}

void KernelActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) {
  // Collect the inputs from input data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      UpdateInputDeviceTensor(input_data, context);
      CopyInputDeviceTensor(input_data, context);
    }
  }

  // Collect the inputs from device tensor store.
  FetchInputByTensorStore(&input_device_tensors_, &input_kernel_tensors_, &input_kernel_tensors_for_infer_,
                          &memory_free_list_, context);
}

void KernelActor::FetchOutputDeviceTensor(OpContext<DeviceTensor> *const context) {
  auto &output_addresses = kernel_info_->output_address_list();
  const auto &output_size_list = kernel_mod_->GetOutputSizeList();

  // May exist in the kernel which does not support the dynamic shape.
  if (output_addresses.size() != output_size_list.size()) {
    std::string error_info = "The outputs number(" + std::to_string(output_size_list.size()) + ") is wrong, " +
                             GetAID().Name() + " may not support the dynamic shape, please check.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
  }

  // Update the size of output device tensor.
  for (size_t i = 0; i < output_addresses.size(); ++i) {
    if (output_size_list[i] == output_addresses[i]->GetSize()) {
      continue;
    }
    output_addresses[i]->SetSize(output_size_list[i]);
  }
}

void KernelActor::PreLaunchKernel(OpContext<DeviceTensor> *) {
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    if (!input_device_tensors_[i]->GetValidPtr(kernel_info_->stream_id())) {
      MS_LOG(DEBUG) << "For kernel: " << kernel_->fullname_with_scope() << ", input device tensor "
                    << input_device_tensors_[i] << " has no device ptr.";
    }
  }

  for (size_t i = 0; i < output_device_tensors_.size(); ++i) {
    if (!output_device_tensors_[i]->GetValidPtr(kernel_info_->stream_id())) {
      MS_LOG(DEBUG) << "For kernel: " << kernel_->fullname_with_scope() << ", output device tensor "
                    << output_device_tensors_[i] << " has no device ptr.";
    }
  }

  for (size_t i = 0; i < workspace_device_tensors_.size(); ++i) {
    if (!workspace_device_tensors_[i]->GetValidPtr(kernel_info_->stream_id())) {
      MS_LOG(DEBUG) << "For kernel: " << kernel_->fullname_with_scope() << ", workspace device tensor "
                    << workspace_device_tensors_[i] << " has no device ptr.";
    }
  }
}

void KernelActor::ExecuteInferShapeTask(OpContext<DeviceTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInfer, GetAID().Name());
  if (IsRunningFailed(context)) {
    MS_LOG(DEBUG) << "Run failed and early stop infer shape for kernel: " << kernel_->fullname_with_scope();
    return;
  }

  if (is_dynamic_type_) {
    InferShapeAndType();
  } else if (is_dynamic_shape_) {
    InferShape();
  }

  Async(kernel_async_resize_aid_, &KernelAsyncResizeActor::ResizeKernelMod, context, this);
}

void KernelActor::ExecuteResizeKernelModTask(OpContext<DeviceTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResize, GetAID().Name());
  if (IsRunningFailed(context)) {
    MS_LOG(DEBUG) << "Run failed and early stop resize for kernel: " << kernel_->fullname_with_scope();
    return;
  }

  if (has_dynamic_) {
    device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    ResizeKernelMod();

    FetchOutputDeviceTensor(context);
    FetchWorkspaceDeviceTensor();
  } else {
    FetchOutputDeviceTensor(context);
  }

  Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernel, context, this);
}

void KernelActor::ExecuteLaunchKernelTask(OpContext<DeviceTensor> *const context) {
  if (IsRunningFailed(context)) {
    MS_LOG(DEBUG) << "Run failed and early stop launch kernel: " << kernel_->fullname_with_scope();
    return;
  }
  // 1. Allocate memory.
  if (!ActorDispatcher::enable_use_trace_memory()) {
    if (!memory_alloc_list_.empty()) {
      SendMemoryAllocReq(context);
    }
  } else if (!max_ref_cnt_output_list_.empty()) {
    // Allocate dynamic memory for graph output.
    MemoryManagerActor::GetInstance()->AllocateMemory(&max_ref_cnt_output_list_, device_contexts_[0], context,
                                                      GetAID());
  }

  if (IsRunningFailed(context)) {
    MS_LOG(DEBUG) << "Run failed and early stop launch kernel: " << kernel_->fullname_with_scope();
    return;
  }
  // For performance, Only kernel need user data (such as PyExecute op) need call 'PreLaunchKernel', the
  // 'PreLaunchKernel' will be removed in the future.
  if (ActorDispatcher::has_kernel_need_user_data()) {
    PreLaunchKernel(context);
  }

  // 2. Launch kernel if need.
  device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);

  if (debug_aid_ != nullptr) {
    ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugPreLaunch, kernel_, input_device_tensors_,
                              output_device_tensors_, device_contexts_[0], context, &GetAID());
  }

  if (!LaunchKernel(context, IsSkippedLaunch(kernel_, nullptr))) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_) << "#umsg#Kernel error:#umsg#Launch kernel failed: " +
                                              kernel_->fullname_with_scope()
                                         << trace::DumpSourceLines(kernel_);
  }

  if (debug_aid_ != nullptr || recorder_aid_ != nullptr) {
    SetMemInfoForDebugAndRdr();

    if (debug_aid_ != nullptr) {
      ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugPostLaunch, kernel_, input_device_tensors_,
                                output_device_tensors_, device_contexts_[0], context, &GetAID());
    }
    if (recorder_aid_ != nullptr) {
      ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordInfo, kernel_->fullname_with_scope(), &mem_info_,
                            device_contexts_[0], context);
    }
  }

  if (is_dynamic_shape_ && kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
    kernel_mod_->UpdateOutputShapeAndSize(input_kernel_tensors_, output_kernel_tensors_);
  }

  if (kernel_mod_->need_user_data()) {
    for_each(output_device_tensors_.begin(), output_device_tensors_.end(),
             [](auto &device_tensor) { device_tensor->set_need_sync_user_data(true); });
  }

  if ((modifiable_ref_input_indexes_.size() != 0) || (modifiable_ref_output_indexes_.size() != 0)) {
    RefreshDeviceTensorCopyStore(context);
  }

  // 3. Free memory.
  if (!ActorDispatcher::enable_use_trace_memory()) {
    if (memory_free_list_.size() > 0) {
      SendMemoryFreeReq(context);
    }
  }
}

void KernelActor::InferAndResize(OpContext<DeviceTensor> *const context) {
  if (!enable_async_infer_) {
    // If the kernel need user data and is dynamic, maybe need input kernel's output user data to infer shape, this
    // value depend case can not handle in KernelTensor auto sync phase currently.
    if (ActorDispatcher::enable_async_launch_kernel() && kernel_mod_->need_user_data() &&
        !WaitRuntimePipelineFinish(context)) {
      MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
      return;
    }

    if (is_dynamic_type_) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInferAndResize, GetAID().Name());
      // For dynamic shape case, need Re-InferShape and Resize kernel mod.
      InferShapeAndType();
      ResizeKernelMod();
    } else if (is_dynamic_shape_) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInferAndResize, GetAID().Name());
      // For dynamic shape case, need Re-InferShape and Resize kernel mod.
      InferShape();
      ResizeKernelMod();
    } else if (is_dynamic_value_) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResize, GetAID().Name());
      ResizeKernelMod();
    }

    return;
  }

  if (is_dynamic_value_ && !is_dynamic_shape_ && !is_dynamic_type_) {
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResize, GetAID().Name());
    ResizeKernelMod();
  }
}

void KernelActor::InferShapeAndType() {
  MS_LOG(DEBUG) << "Begin InferShapeAnyType for kernel: " << kernel_->fullname_with_scope()
                << ", inputs: " << input_kernel_tensors_for_infer_;
  // 1. Infer operator's output's Shape and Type.
  auto abstract = opt::dynamic_shape::InferShapeAndType(kernel_mod_->primitive(), input_kernel_tensors_for_infer_);
  MS_EXCEPTION_IF_NULL(abstract);
  MS_LOG(DEBUG) << "End InferShapeAnyType for kernel: " << kernel_->fullname_with_scope()
                << ", abstract: " << abstract->ToString();
  // 2. Update shape of output kernel tensor.
  opt::dynamic_shape::UpdateKernelTensorType(abstract->GetType(), output_kernel_tensors_);
  opt::dynamic_shape::UpdateKernelTensorShape(abstract->GetShape(), output_kernel_tensors_);
}

void KernelActor::InferShape() {
  MS_LOG(DEBUG) << "Begin InferShape for kernel: " << kernel_->fullname_with_scope()
                << ", inputs: " << input_kernel_tensors_for_infer_;
  // 1. Infer operator's output's Shape.
  auto base_shape = opt::dynamic_shape::InferShape(kernel_mod_->primitive(), input_kernel_tensors_for_infer_);
  MS_EXCEPTION_IF_NULL(base_shape);
  MS_LOG(DEBUG) << "End InferShape for kernel: " << kernel_->fullname_with_scope()
                << ", shape: " << base_shape->ToString();

  // 2. Update shape of output kernel tensor.
  opt::dynamic_shape::UpdateKernelTensorShape(base_shape, output_kernel_tensors_);
}

void KernelActor::ResizeKernelMod() {
  MS_LOG(DEBUG) << "Begin Resize kernel mod for kernel: " << kernel_->fullname_with_scope();
  int ret = kernel_mod_->Resize(input_kernel_tensors_, output_kernel_tensors_);
  MS_LOG(DEBUG) << "End Resize kernel mod for kernel: " << kernel_->fullname_with_scope()
                << ", the output size list: " << kernel_mod_->GetOutputSizeList()
                << ", workspace size list: " << kernel_mod_->GetWorkspaceSizeList();
  if (ret != kernel::KRET_OK) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_) << "Resize failed for kernel: " << kernel_->fullname_with_scope();
  }
}
namespace {
void TrackInputMemory(const std::vector<DeviceTensor *> &input_device_tensors, const std::string &actor_name,
                      const std::vector<bool> &depend_shape_input_list) {
  for (size_t i = 0, end = input_device_tensors.size(); i < end; i++) {
    // Skip shape depend inputs.
    if (i < depend_shape_input_list.size() && depend_shape_input_list[i]) {
      continue;
    }
    auto device_addr = input_device_tensors[i];
    if (device_addr == nullptr || !device_addr->IsPtrValid()) {
      continue;
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(UseMemBlock, actor_name, device_addr->GetPtr());
  }
}
}  // namespace

bool KernelActor::LaunchKernel(OpContext<DeviceTensor> *const context, bool is_skip_launch) {
  if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    TrackInputMemory(input_device_tensors_, GetAID().Name(), depend_shape_input_list_);
  }
  if (is_skip_launch) {
    return true;
  }
  if (skip_launch_shape_related_op_) {
    MS_LOG(DEBUG) << "Skip launch real make tuple kernel: " << kernel_->fullname_with_scope()
                  << " input kernel tensor: " << input_kernel_tensors_;
    return true;
  }
  // Check the skipped launch condition.
  if (is_launch_skipped_) {
    MS_EXCEPTION_IF_CHECK_FAIL((input_device_tensors_.size() >= 1), "The inputs size is wrong.");
    MS_EXCEPTION_IF_CHECK_FAIL((output_device_tensors_.size() >= 1), "The outputs size is wrong.");
    MS_EXCEPTION_IF_NULL(input_device_tensors_[0]);
    MS_EXCEPTION_IF_NULL(output_device_tensors_[0]);
    if (input_device_tensors_[0]->GetPtr() == output_device_tensors_[0]->GetPtr()) {
      MS_LOG(DEBUG) << "Skipped launch kernel: " << kernel_->fullname_with_scope();
      return true;
    } else {
      MS_LOG(ERROR) << "Input address:" << input_device_tensors_[0]->GetPtr()
                    << " and output address:" << output_device_tensors_[0]->GetPtr()
                    << " are not equal of skipped launch actor: " << GetAID().Name();
      return false;
    }
  }

  // Cpu not support stream lock with LaunchKernel.
  if (!ActorDispatcher::enable_multi_stream() || is_multi_stream_process_skipped_) {
    MS_LOG(DEBUG) << "Begin launch kernel: " << kernel_->fullname_with_scope();
    auto ret = device_contexts_[0]->GetKernelExecutor(false)->LaunchKernel(
      kernel_, input_kernel_tensors_, workspace_kernel_tensors_, output_kernel_tensors_, kernel_mod_, stream_);
    MS_LOG(DEBUG) << "End launch kernel: " << kernel_->fullname_with_scope();
    return ret;
  }

  auto multi_stream_controller = device::MultiStreamController::GetInstance();
  bool ret = false;
  {
    std::lock_guard<std::mutex> lock(
      multi_stream_controller->GetStreamMutex(device_contexts_[0], kernel_info_->stream_id()));
    // Here should process multi stream first to make inputs is memory safe.
    ProcessMultiStreamBeforeKernelLaunch(context);
    MS_LOG(DEBUG) << "Begin launch kernel: " << kernel_->fullname_with_scope();
    ret = device_contexts_[0]->GetKernelExecutor(false)->LaunchKernel(
      kernel_, input_kernel_tensors_, workspace_kernel_tensors_, output_kernel_tensors_, kernel_mod_, stream_);
    MS_LOG(DEBUG) << "End launch kernel: " << kernel_->fullname_with_scope();
  }
  ProcessMultiStreamAfterKernelLaunch(context);
  return ret;
}

void KernelActor::ProcessMultiStreamBeforeKernelLaunch(OpContext<DeviceTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kProcessMultiStream, GetAID().Name());
  auto device_context = device_contexts_[0];
  auto stream_id = kernel_info_->stream_id();
  // Update output_kernel_tensors_ with task id on stream.
  auto multi_stream_controller = device::MultiStreamController::GetInstance();
  auto task_id_on_stream = multi_stream_controller->LaunchTaskIdOnStream(device_context, stream_id);
  MS_LOG(DEBUG) << "device context : " << device_context
                << ", name : " << device_context->device_context_key().device_name_ << ", stream id : " << stream_id
                << ", actor name : " << GetAID().Name() << ", task_id_on_stream : " << task_id_on_stream << ".";
  if (INT64_MAX == task_id_on_stream) {
    // Cpu kernel task id on stream is meanless.
    *task_id_on_stream_ = 0;
    MS_LOG(DEBUG) << "Skip ProcessMultiStreamBeforeKernelLaunch since kernel type is CPU.";
    return;
  }
  *task_id_on_stream_ = task_id_on_stream;

  // Process wait stream.
  if (is_stream_recv_actor_) {
    // Note: wait node start to launch. Event was record on send node, so, we can releases events on send node stream.
    // Release events on send node means memory stream id is recv node stream id and user stream id is send node
    // stream id.
    auto user_stream_id = kernel_mod_->record_stream_id();
    auto memory_stream_id = stream_id;
    if (stream_send_actor_ == nullptr) {
      // Gpu not add stream send/recv pair, nullptr is normal case.
      MS_LOG(DEBUG) << "Stream_send_actor_ is nullptr.";
      return;
    }
    MS_LOG(DEBUG) << "Process wait stream start, memory_stream_id : " << memory_stream_id
                  << ", send task id on stream : " << *(stream_send_actor_->task_id_on_stream_) << ".";
    // Here, need get task id on stream from send node.
    (void)multi_stream_controller->WaitEvent(device_context, *(stream_send_actor_->task_id_on_stream_), user_stream_id,
                                             memory_stream_id);
    return;
  }

  // Reset cross stream addresses.
  cross_stream_addresses_.clear();

  // Process inputs.
  if (input_kernel_tensors_.empty()) {
    return;
  }

  std::vector<KernelTensor *> cross_stream_kernel_tensors;
  for (const auto &input_kernel_tensor : input_kernel_tensors_) {
    if (input_kernel_tensor->stream_id() == stream_id) {
      continue;
    }
    if (input_kernel_tensor->task_id_on_stream() == nullptr) {
      MS_LOG(DEBUG) << "Input_kernel_tensor : " << input_kernel_tensor
                    << " task id on stream is nullptr, will skip multi stream process.";
      continue;
    }
    if (input_kernel_tensor->managed_by_somas()) {
      MS_LOG(DEBUG) << "Input_kernel_tensor : " << input_kernel_tensor << " is managed by somas.";
      continue;
    }
    // Nullptr device ptr is normal case, here need skip these inputs.
    if (input_kernel_tensor->device_ptr() == nullptr) {
      MS_LOG(DEBUG) << "Input kernel tensor device ptr is nullptr.";
      continue;
    }
    (void)cross_stream_addresses_.emplace_back(input_kernel_tensor->stream_id(), input_kernel_tensor->device_ptr());
    if (!is_multi_stream_safe_) {
      (void)cross_stream_kernel_tensors.emplace_back(input_kernel_tensor);
    }
  }

  // Dispatch record/wait.
  if (!is_multi_stream_safe_) {
    for (const auto &cross_stream_kernel_tensor : cross_stream_kernel_tensors) {
      // Nullptr of task id on stream is normal case.
      // If cross_stream_kernel_tensor's task id on stream is nullptr, kernel tensor must be safe.
      // Data prepare actor, data source actor and so on has prepare device tensors without task id on stream, and
      // those device tensors is multi-stream safe.
      if (cross_stream_kernel_tensor->task_id_on_stream() == nullptr) {
        continue;
      }
      // Input kernel tensor is memory stream id, this is important.
      auto user_stream_id = stream_id;
      auto memory_stream_id = cross_stream_kernel_tensor->stream_id();
      auto memory_task_id_on_stream = *cross_stream_kernel_tensor->task_id_on_stream();
      auto safe_task_id_on_stream =
        multi_stream_controller->QueryTaskIdOnStream(device_context, user_stream_id, memory_stream_id);
      if (safe_task_id_on_stream >= memory_task_id_on_stream) {
        MS_LOG(DEBUG) << "Safe_task_id_on_stream : " << safe_task_id_on_stream
                      << " is bigger than memory_task_id_on_stream : " << memory_task_id_on_stream << ".";
        continue;
      }
      multi_stream_controller->DispatchRecordWaitEvent(device_context, user_stream_id, memory_stream_id);
      // Add recv process.
      user_stream_id = memory_stream_id;
      memory_stream_id = stream_id;
      auto last_task_id_on_stream = multi_stream_controller->GetTaskIdOnStream(device_context, user_stream_id);
      MS_LOG(DEBUG) << "Dispatch wait stream start, usert_stream_id : " << user_stream_id
                    << ", memory_stream_id : " << memory_stream_id
                    << ", last_task_id_on_stream : " << last_task_id_on_stream << ".";
      // Here, need get task id on stream from send node.
      (void)multi_stream_controller->WaitEvent(device_context, last_task_id_on_stream, user_stream_id,
                                               memory_stream_id);
    }
  }
}

void KernelActor::ProcessMultiStreamAfterKernelLaunch(OpContext<DeviceTensor> *const context) {
  auto stream_id = kernel_info_->stream_id();
  if (stream_id != kDefaultStreamIndex) {
    for (const auto &output_kernel_tensor : output_kernel_tensors_) {
      cross_stream_addresses_.emplace_back(kDefaultStreamIndex, output_kernel_tensor->device_ptr());
    }
  }

  // Record event.
  if (!cross_stream_addresses_.empty()) {
    MS_LOG(DEBUG) << "Record event for kernel : " << kernel_->fullname_with_scope()
                  << ", addresses size : " << cross_stream_addresses_.size() << ".";
    // Record event on stream.
    auto device_context = device_contexts_[0];
    auto multi_stream_controller = device::MultiStreamController::GetInstance();
    multi_stream_controller->RecordEvent(device_context, *task_id_on_stream_, stream_id, cross_stream_addresses_);
  }
}

void KernelActor::PostLaunchKernel(OpContext<DeviceTensor> *const context) {
  if (is_dynamic_shape_ && kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
    kernel_mod_->UpdateOutputShapeAndSize(input_kernel_tensors_, output_kernel_tensors_);
  }

  if (kernel_mod_->need_user_data()) {
    for_each(output_device_tensors_.begin(), output_device_tensors_.end(),
             [](auto &device_tensor) { device_tensor->set_need_sync_user_data(true); });
  }

  if ((modifiable_ref_input_indexes_.size() != 0) || (modifiable_ref_output_indexes_.size() != 0)) {
    RefreshDeviceTensorCopyStore(context);
  }

  // The input is invalid and needs to be erased when finish kernel launch.
  EraseInput(context);

  // Note that SendMemoryFreeReq must be in front of SendOutput, because SendOutput will trigger SendMemoryAllocReq
  // of the next actor and the actor is asynchronous execution. So it is necessary to ensure that SendMemoryFreeReq
  // of the current actor is in front of SendMemoryAllocReq of the next actor. One is to reuse the memory more
  // fully, the other is to ensure the execution order and avoid the illegal memory timing problem.
  if (memory_free_list_.size() > 0) {
    SendMemoryFreeReq(context);
  }

  SendOutput(context);
}

void KernelActor::RefreshDeviceTensorCopyStore(OpContext<DeviceTensor> *const context) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  for (auto &ref_input_index : modifiable_ref_input_indexes_) {
    if (ref_input_index >= input_device_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The input index is of range.");
    }
    auto &input_device_tensor = input_device_tensors_[ref_input_index];
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto need_refreshed_device_tensors = DeviceTensorCopyStore::GetInstance().Fetch(input_device_tensor);
    for (auto &new_device_tensor : need_refreshed_device_tensors) {
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the input position:" << ref_input_index
                   << " refresh from device address:" << input_device_tensor
                   << ", type:" << input_device_tensor->GetDeviceType() << ", format:" << input_device_tensor->format()
                   << " to device address:" << new_device_tensor << ", type:" << new_device_tensor->GetDeviceType()
                   << ", format:" << new_device_tensor->format();
      if (!Copy(new_device_tensor, input_device_tensor)) {
        std::string error_info = "Copy input device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }

  for (auto &ref_output_index : modifiable_ref_output_indexes_) {
    if (ref_output_index >= output_device_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The output index is of range.");
    }
    auto &output_device_tensor = output_device_tensors_[ref_output_index];
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    auto need_refreshed_device_tensors = DeviceTensorCopyStore::GetInstance().Fetch(output_device_tensor);
    for (auto &new_device_tensor : need_refreshed_device_tensors) {
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the output position:" << ref_output_index
                   << " refresh from device address:" << output_device_tensor
                   << ", type:" << output_device_tensor->GetDeviceType()
                   << ", format:" << output_device_tensor->format() << " to device address:" << new_device_tensor
                   << ", type:" << new_device_tensor->GetDeviceType() << ", format:" << new_device_tensor->format();
      if (!Copy(new_device_tensor, output_device_tensor)) {
        std::string error_info = "Copy output device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kPostLaunch, GetAID().Name(), false);
}

void KernelActor::SendRecorderInfo(OpContext<DeviceTensor> *const context) const {
  if (recorder_aid_ != nullptr && !ActorDispatcher::enable_async_launch_kernel()) {
    MS_EXCEPTION_IF_NULL(kernel_);
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordInfo, kernel_->fullname_with_scope(), &mem_info_,
                          device_contexts_[0], context);
  }
}

void KernelActor::SetInputDeviceTensor(DeviceTensor *input_device_tensor, size_t input_index) {
  input_device_tensors_[input_index] = input_device_tensor;
  input_kernel_tensors_[input_index] = input_device_tensor->kernel_tensor().get();
  input_kernel_tensors_for_infer_[input_index] = input_device_tensor->kernel_tensor();
}

}  // namespace runtime
}  // namespace mindspore
