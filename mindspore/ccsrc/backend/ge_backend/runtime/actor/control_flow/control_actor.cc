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

#include "backend/ge_backend/runtime/actor/control_flow/control_actor.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "utils/convert_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "primitive/framework_ops.h"
#include "utils/profile.h"
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
ControlActor::ControlActor(const std::string &name, KernelTransformType type, const AID &memory_manager_aid,
                           const std::vector<KernelWithIndex> &parameters, const AnfNodePtr &node)
    : MemoryAwareActor(name, type, nullptr, memory_manager_aid), formal_parameters_(parameters), node_(node) {
  input_partials_.resize(parameters.size());
  input_kernel_tensors_.resize(parameters.size());
  output_data_by_output_index_.resize(parameters.size());
}

void ControlActor::Init() {
  InitOutputData();
  if (output_data_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "The output data size is wrong: " << GetAID().Name();
  }

  size_t output_data_index = 0;
  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    const auto &data_arrow = output_data_arrows_[i];
    auto data = output_data_[output_data_index].first.get();
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) >= output_data_by_output_index_.size()) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID();
    }
    if (i < output_need_disable_dynamic_ref_counts_.size() && output_need_disable_dynamic_ref_counts_[i] &&
        data->data_ != nullptr) {
      data->data_->UpdateFlag(device::kDeviceAddressFlagNullptr);
      MS_LOG(INFO) << "Add null flag for device address:" << data->data_ << " in actor:" << GetAID();
    }
    (void)output_data_by_output_index_[IntToSize(data_arrow->from_output_index_)].emplace_back(data);
    ++output_data_index;
  }
}

void ControlActor::GetAllKernelTensors(const OpPartialPtr &op_partial, std::vector<KernelTensorPtr> *kernel_tensors) {
  MS_EXCEPTION_IF_NULL(op_partial);
  (void)std::transform(op_partial->kernel_tensors_.begin(), op_partial->kernel_tensors_.end(),
                       std::back_inserter(*kernel_tensors),
                       [](const auto &kernel_tensor) { return kernel_tensor.second; });

  // Foreach the op partial to fetch the device tensors.
  for (auto &partial : op_partial->partials_) {
    GetAllKernelTensors(partial.second, kernel_tensors);
  }
}

void ControlActor::GetAllKernelTensors(const OpRealParameterWithBranchID &op_real_parameter,
                                       std::vector<KernelTensorPtr> *kernel_tensors) {
  MS_EXCEPTION_IF_NULL(kernel_tensors);
  for (auto &kernel_tensor : op_real_parameter.kernel_tensors_) {
    (void)kernel_tensors->emplace_back(kernel_tensor.second);
  }

  // Foreach the op partial to fetch the device tensors.
  for (auto &partial : op_real_parameter.partials_) {
    GetAllKernelTensors(partial.second, kernel_tensors);
  }
}

void ControlActor::IncreaseDynamicRefCount(const OpData<KernelTensor> *op_data) const {
  MS_EXCEPTION_IF_NULL(op_data);
  MS_EXCEPTION_IF_NULL(op_data->data_);
  op_data->data_->IncreaseDynamicRefCount(GetAID().Name());
}

void ControlActor::IncreaseDynamicRefCount(const OpPartialPtr &op_partial) {
  if (op_partial == nullptr) {
    MS_LOG(EXCEPTION) << "Empty op partial for actor:" << GetAID();
  }
  std::vector<KernelTensorPtr> partial_kernel_tensors;
  GetAllKernelTensors(op_partial, &partial_kernel_tensors);
  for (auto &partial_kernel_tensor : partial_kernel_tensors) {
    MS_EXCEPTION_IF_NULL(partial_kernel_tensor);
    partial_kernel_tensor->IncreaseDynamicRefCount(GetAID().Name());
  }
}

void ControlActor::IncreaseDynamicRefCount(const OpRealParameterWithBranchID &op_real_parameter) {
  std::vector<KernelTensorPtr> partial_kernel_tensors;
  GetAllKernelTensors(op_real_parameter, &partial_kernel_tensors);
  for (auto &partial_kernel_tensor : partial_kernel_tensors) {
    MS_EXCEPTION_IF_NULL(partial_kernel_tensor);
    partial_kernel_tensor->IncreaseDynamicRefCount(GetAID().Name());
  }
}

size_t ControlActor::FetchNodePosition(const KernelWithIndex &node) const {
  const auto &iter = find(formal_parameters_.begin(), formal_parameters_.end(), node);
  if (iter == formal_parameters_.end()) {
    const auto &load_iter =
      std::find_if(formal_parameters_.begin(), formal_parameters_.end(), [&node](const KernelWithIndex &pair) {
        return pair.first != nullptr && common::AnfAlgo::CheckPrimitiveType(pair.first, prim::kPrimLoad) &&
               pair.first->cast<CNodePtr>()->input(1) == node.first && node.second == 0;
      });
    if (load_iter != formal_parameters_.end()) {
      return load_iter - formal_parameters_.begin();
    }
    const auto &get_item_iter =
      std::find_if(formal_parameters_.begin(), formal_parameters_.end(), [&node](const KernelWithIndex &pair) {
        return pair.first != nullptr && common::AnfAlgo::CheckPrimitiveType(pair.first, prim::kPrimTupleGetItem) &&
               common::AnfAlgo::GetTupleGetItemRealInput(pair.first->cast<CNodePtr>()) == node.first &&
               common::AnfAlgo::GetTupleGetItemOutIndex(pair.first->cast<CNodePtr>()) == node.second;
      });
    if (get_item_iter != formal_parameters_.end()) {
      MS_LOG(INFO) << "Input node:" << node.first->DebugString() << " fullname:" << node.first->fullname_with_scope()
                   << " node ptr:" << node.first << " index:" << node.second
                   << " match the tuple get item node:" << get_item_iter->first->DebugString()
                   << " fullname:" << get_item_iter->first->fullname_with_scope()
                   << " node ptr:" << get_item_iter->first << " index:" << get_item_iter->second;
      return get_item_iter - formal_parameters_.begin();
    }
    for (const auto &formal_parameter : formal_parameters_) {
      MS_LOG(WARNING) << "Actor:" << GetAID() << " formal parameter:"
                      << (formal_parameter.first != nullptr ? formal_parameter.first->DebugString() : "")
                      << " full name:"
                      << (formal_parameter.first != nullptr ? formal_parameter.first->fullname_with_scope() : "")
                      << " index:" << formal_parameter.second << " node ptr:" << formal_parameter.first;
    }
    MS_LOG_WITH_NODE(EXCEPTION, node.first)
      << "Invalid formal parameter:" << (node.first != nullptr ? node.first->DebugString() : "")
      << " full name:" << (node.first != nullptr ? node.first->fullname_with_scope() : "") << " node ptr:" << node.first
      << " index:" << node.second << " for actor:" << GetAID();
  }
  return iter - formal_parameters_.begin();
}

void ControlActor::Run(OpContext<KernelTensor> *const context) {
  try {
    // The exit actor is the output of kernel graph when the node_ is null.
    if (type_ == KernelTransformType::kExitActor && node_ == nullptr) {
      double end_time = GetTime();
      const size_t kSecondsToMilliseconds = 1000;
      MS_LOG(DEBUG) << "Kernel graph group exit actor:" << GetAID()
                    << " cost time:" << (end_time - start_time_) * kSecondsToMilliseconds;
    }

    FetchInput(context);
    if (IsRunningFailed(context)) {
      MS_LOG(INFO) << "Run failed and early stop.";
      return;
    }

    // Note that IncreaseDynamicRefCounts must be in front of SendMemoryFreeReq. SendMemoryFreeReq will decreasing the
    // dynamic ref count. Avoid the illegal timing problem that the dynamic reference count is decremented and then
    // incremented.
    IncreaseDynamicRefCounts(context);
    SendMemoryFreeReq(context);

    EraseInput(context);
    SendOutput(context);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info = "Actor fun failed:" + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
  }
}

void ControlActor::RunOpPartial(const OpPartialPtr &partial, size_t position, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_partials_[sequential_num].emplace_back(position, partial);

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op partial, position:" << position
                << " and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

void ControlActor::RunBranchID(int branch_id, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  input_branch_ids_[sequential_num].push(branch_id);

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input branch id and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

bool ControlActor::CheckRunningCondition(const OpContext<KernelTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);

  if (!AbstractActor::CheckRunningCondition(context)) {
    return false;
  }

  if (input_partials_num_ != 0) {
    const auto &partial_iter = input_op_partials_.find(context->sequential_num_);
    if (partial_iter == input_op_partials_.end()) {
      return false;
    }
    if (partial_iter->second.size() < input_partials_num_) {
      return false;
    } else if (partial_iter->second.size() > input_partials_num_) {
      MS_LOG(ERROR) << "Invalid input partial num:" << partial_iter->second.size() << " need:" << input_partials_num_
                    << " for actor:" << GetAID();
      return false;
    }
  }
  return true;
}

void ControlActor::FetchInput(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);

  // Fetch input device tensor from input data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) >= input_kernel_tensors_.size()) {
        std::string error_info = "Invalid index, need:" + std::to_string(input_data->index_) +
                                 " current:" + std::to_string(input_kernel_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_EXCEPTION_IF_NULL(input_data->data_);
      input_kernel_tensors_[IntToSize(input_data->index_)] = input_data->data_;
    }
  }

  // Fetch input device tensor from local device tensor.
  for (auto &local_kernel_tensor : local_kernel_tensors_) {
    MS_EXCEPTION_IF_NULL(local_kernel_tensor.second.first);
    MS_EXCEPTION_IF_NULL(local_kernel_tensor.second.first->device_address());
    if (local_kernel_tensor.first >= input_kernel_tensors_.size()) {
      std::string error_info = "Invalid local index:" + std::to_string(local_kernel_tensor.first) +
                               " current:" + std::to_string(local_kernel_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_kernel_tensors_[local_kernel_tensor.first] = local_kernel_tensor.second.first;
  }

  // Fetch input device tensor from device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto kernel_tensors = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get());
    if (kernel_tensors.empty()) {
      auto ms_context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(ms_context);
      const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
      MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
      std::string error_info = GetAID().Name() +
                               " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
                               ", device type:" + device_name;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    if (device_tensor_store_key.first >= input_kernel_tensors_.size()) {
      std::string error_info =
        "The input index is out of range, need:" + std::to_string(device_tensor_store_key.first) +
        " current:" + std::to_string(input_kernel_tensors_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    MS_EXCEPTION_IF_NULL(kernel_tensors[0]);
    input_kernel_tensors_[device_tensor_store_key.first] = kernel_tensors[0];
  }

  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    if (output_data_by_output_index_[i].empty()) {
      continue;
    }
    const auto &data = input_kernel_tensors_[i];
    MS_EXCEPTION_IF_NULL(data);
    for (auto &output_data : output_data_by_output_index_[i]) {
      MS_EXCEPTION_IF_NULL(output_data);
      output_data->data_ = data;
    }
  }

  // Fetch input partial from input data.
  const auto &partial_iter = input_op_partials_.find(context->sequential_num_);
  if (partial_iter != input_op_partials_.end()) {
    for (const auto &input_partial : partial_iter->second) {
      if (input_partial.first >= input_partials_.size()) {
        std::string error_info = "Invalid partial index:" + std::to_string(input_partial.first) +
                                 " vector size:" + std::to_string(input_partials_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_partials_[input_partial.first] = input_partial.second;
    }
  }
  // Fetch input partial from local partial.
  for (const auto &local_partial : local_partials_) {
    if (local_partial.first >= input_partials_.size()) {
      std::string error_info = "Invalid partial index:" + std::to_string(local_partial.first) +
                               " vector size:" + std::to_string(input_partials_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    MS_EXCEPTION_IF_NULL(local_partial.second);
    input_partials_[local_partial.first] = local_partial.second;
  }
  // Fetch branch id in stack.
  auto iter = input_branch_ids_.find(context->sequential_num_);
  if (iter != input_branch_ids_.end() && (!iter->second.empty())) {
    output_branch_id_ = iter->second.top();
  }
}

void ControlActor::IncreaseDynamicRefCounts(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  // Increase dynamic ref count by the output data.
  for (size_t i = 0; i < output_data_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_data_[i].first);
    if (output_data_[i].first->data_ == nullptr) {
      std::string error_info = GetAID().Name() + " fetches data null, data index:" + std::to_string(i) +
                               " to actor:" + output_data_[i].first->op_id_.Name() +
                               " index:" + std::to_string(output_data_[i].first->index_);
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    if (i < output_need_disable_dynamic_ref_counts_.size() && output_need_disable_dynamic_ref_counts_[i]) {
      MS_LOG(DEBUG) << "Disable dynamic ref count for device address:" << output_data_[i].first->data_
                    << " ptr:" << output_data_[i].first->data_->device_address()->GetPtr() << " for actor:" << GetAID();
      output_data_[i].first->data_->UpdateFlag(device::kDeviceAddressFlagNullptr);
      continue;
    }
    IncreaseDynamicRefCount(output_data_[i].first.get());
  }

  // Increase dynamic ref count by the output partial.
  for (const auto &output_partial_arrow : output_partial_arrows_) {
    MS_EXCEPTION_IF_NULL(output_partial_arrow);
    if (IntToSize(output_partial_arrow->from_output_index_) >= input_partials_.size()) {
      std::string error_info = "Invalid partial input:" + std::to_string(output_partial_arrow->from_output_index_) +
                               " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    auto output_partial = input_partials_[IntToSize(output_partial_arrow->from_output_index_)];
    IncreaseDynamicRefCount(output_partial);
  }
}

void ControlActor::SendMemoryFreeReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;

  // Collect the input device tensors.
  std::vector<KernelTensorPtr> memory_free_list;
  if (input_op_datas_.count(sequential_num) > 0) {
    for (auto &input_op_data : input_op_datas_[sequential_num]) {
      MS_EXCEPTION_IF_NULL(input_op_data);
      MS_EXCEPTION_IF_NULL(input_op_data->data_);
      if (input_need_disable_dynamic_ref_counts_.find(input_op_data->index_) !=
          input_need_disable_dynamic_ref_counts_.end()) {
        MS_LOG(DEBUG) << "Actor:" << GetAID() << " skip free dynamic ref count for:" << input_op_data->data_
                      << " index:" << input_op_data->index_;
        continue;
      }
      (void)memory_free_list.emplace_back(input_op_data->data_);
    }
  }

  if (input_op_partials_.count(sequential_num) > 0) {
    for (auto &input_op_partial : input_op_partials_[sequential_num]) {
      GetAllKernelTensors(input_op_partial.second, &memory_free_list);
    }
  }

  if (memory_free_list.size() > 0) {
    memory_free_lists_.push(memory_free_list);
    if (ActorDispatcher::is_memory_free_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                                context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()), context,
                            GetAID());
    }
  }
}

void ControlActor::EraseInput(const OpContext<KernelTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;
  AbstractActor::EraseInput(context);
  if (input_partials_num_ != 0) {
    auto ret = input_op_partials_.erase(sequential_num);
    if (ret == 0) {
      std::string error_info = "Erase input partial failed: " + GetAID().Name();
      // The sequential num may be invalid, can't set the promise value of context.
      MS_LOG(ERROR) << error_info << ", sequential_num: " << sequential_num;
    }
  }

  if (input_branch_ids_.find(sequential_num) != input_branch_ids_.end()) {
    input_branch_ids_[sequential_num].pop();
    if (input_branch_ids_[sequential_num].empty()) {
      auto ret = input_branch_ids_.erase(sequential_num);
      if (ret == 0) {
        MS_LOG(ERROR) << "Erase input branch id failed: " << GetAID() << ", sequential_num: " << sequential_num;
        return;
      }
    }
  }
}

void ControlActor::CreateHeterDeviceTensor(KernelTensor *const node_kernel_tensor,
                                           KernelTensor *const input_kernel_tensor, size_t index,
                                           OpContext<KernelTensor> *const context, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(input_kernel_tensor);
  MS_EXCEPTION_IF_NULL(node_kernel_tensor);
  const auto &input_device_tensor = input_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(input_device_tensor);
  const auto &node_device_tensor = node_kernel_tensor->device_address().get();
  MS_EXCEPTION_IF_NULL(node_device_tensor);
  MS_EXCEPTION_IF_NULL(node);
  auto new_kernel_tensor = node_kernel_tensor->CloneKernelTensor();
  MS_EXCEPTION_IF_NULL(new_kernel_tensor);
  new_kernel_tensor->set_device_ptr(nullptr);
  new_kernel_tensor->SetShape(input_kernel_tensor->GetShape());
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  auto new_device_tensor = new_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(new_device_tensor);
  UpdateRefCount(new_kernel_tensor, true);
  created_heter_kernel_tensors_[std::make_pair(index, new_device_tensor->GetDeviceType())] = new_kernel_tensor;
  MS_LOG(DEBUG) << "Actor:" << GetAID() << " create new device tensor:" << new_device_tensor->ToString()
                << " by node device tensor:" << node_device_tensor->ToString();

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), "UpdateOutputData", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(), memory::mem_pool::MemType::kOther,
                                                 new_device_tensor->GetSize(), new_device_tensor.get());
  auto data_stream_id = input_device_tensor->stream_id();
  auto device_tensor_stream_id = new_device_tensor->stream_id();
  if (device_tensor_stream_id != data_stream_id) {
    MS_LOG(INFO) << "Rewrite device tesnor stream id from : " << device_tensor_stream_id
                 << " to data stream id : " << data_stream_id << ".";
    new_device_tensor->set_stream_id(data_stream_id);
  }
  if ((new_device_tensor->GetPtr() == nullptr) &&
      (!host_context->device_res_manager_->AllocateMemory(new_device_tensor.get(), kDefaultStreamIndex))) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, node->DebugString(),
                                                new_device_tensor->GetSize());
  }
  if (!SyncCopy(new_kernel_tensor.get(), input_kernel_tensor, kDefaultStreamIndex)) {
    std::string error_info =
      "The formal parameter: " + node->DebugString() + " position:" + std::to_string(index) + " copy failed.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
}

void ControlActor::UpdateOutputData(OpData<KernelTensor> *const output_data, const DataArrowPtr &data_arrow,
                                    const AnfNodePtr &, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_data);
  MS_EXCEPTION_IF_NULL(data_arrow);
  auto formal_parameter_position = data_arrow->from_output_index_;
  // Has no the ref node formal parameter.
  if (ref_node_formal_parameter_kernel_tensors_.count(formal_parameter_position) == 0) {
    return;
  }

  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(output_data);
  MS_EXCEPTION_IF_NULL(output_data->data_);
  auto data_kernel_tensor = output_data->data_;
  auto data = data_kernel_tensor->device_address().get();
  MS_EXCEPTION_IF_NULL(data);
  if ((!data->IsPtrValid()) || (output_data->data_->ref_count() != SIZE_MAX)) {
    std::string error_info = "The address of the " + std::to_string(formal_parameter_position) +
                             " kernel tensor:" + output_data->data_->ToString() +
                             " position real parameter is nullptr or ref count is wrong for actor:" + GetAID().Name() +
                             ".";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
  if ((output_data->data_->dynamic_ref_count() != INT32_MAX)) {
    MS_LOG(DEBUG) << "Set dynamic ref count from:" << output_data->data_->dynamic_ref_count()
                  << " to int32 max for kernel tensor:" << output_data->data_->ToString() << " in actor:" << GetAID();
    output_data->data_->set_dynamic_ref_count(INT32_MAX);
  }

  // Foreach the device tensors to set the ptr from data, only the formal parameter device tensor of ref node need set
  // before kernel running, because it will be used by ref output node.
  KernelTensorPtr real_kernel_tensor = nullptr;
  for (auto &kernel_tensor : ref_node_formal_parameter_kernel_tensors_[formal_parameter_position]) {
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_tensor = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_tensor);
    if ((device_tensor.get() == data) || (device_tensor->GetMutablePtr() == data->GetMutablePtr())) {
      continue;
    }
    auto formal_parameter = device_tensor->GetNodeIndex();
    MS_EXCEPTION_IF_NULL(formal_parameter.first);
    if ((device_tensor->GetSize() != data->GetSize()) ||
        (kernel_tensor->dtype_id() != data_kernel_tensor->dtype_id())) {
      MS_LOG(WARNING) << "The formal parameter: " << formal_parameter.first->DebugString()
                      << " position:" << formal_parameter_position
                      << "please check the size and type id, formal parameter size:" << device_tensor->GetSize()
                      << " type id:" << kernel_tensor->dtype_id() << ", real parameter size:" << data->GetSize()
                      << " type id:" << data_kernel_tensor->dtype_id();
    }
    MS_LOG(DEBUG) << "Check copy for device address:" << device_tensor << " type:" << device_tensor->GetDeviceType()
                  << " and " << data << " type:" << data->GetDeviceType() << " ptr:" << data->GetPtr();
    // Copy from the real parameter to formal parameter and insert the device tensor copy store.
    if ((!AnfAlgo::IsEquivalentFormat(kernel_tensor->format(), data_kernel_tensor->format())) ||
        (device_tensor->GetDeviceType() != data->GetDeviceType())) {
      MS_LOG(INFO) << GetAID().Name() << " the input position:" << formal_parameter_position
                   << " copy from real parameter kernel tensor:" << data_kernel_tensor->ToString()
                   << " to formal parameter kernel tensor:" << kernel_tensor->ToString()
                   << ", formal parameter name:" << formal_parameter.first->DebugString();
      const auto &iter =
        created_heter_kernel_tensors_.find(std::make_pair(formal_parameter_position, device_tensor->GetDeviceType()));
      if (iter == created_heter_kernel_tensors_.end()) {
        CreateHeterDeviceTensor(kernel_tensor.get(), output_data->data_.get(), formal_parameter_position, context,
                                formal_parameter.first);
      }
      const auto dst_kernel_tensor =
        created_heter_kernel_tensors_[std::make_pair(formal_parameter_position, device_tensor->GetDeviceType())].get();
      MS_EXCEPTION_IF_NULL(dst_kernel_tensor);
      const auto &dst_device_tensor = dst_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(dst_device_tensor);
      MS_LOG(DEBUG) << "Set ptr:" << dst_device_tensor->GetPtr() << " from:" << dst_device_tensor
                    << " type:" << dst_device_tensor->GetDeviceType() << " to:" << device_tensor
                    << " type:" << device_tensor->GetDeviceType();
      device_tensor->set_ptr(dst_device_tensor->GetMutablePtr());
      kernel_tensor->SetShape(dst_kernel_tensor->GetShape());
      MS_LOG(DEBUG) << "Add device tensor copy store for device address:" << device_tensor
                    << " type:" << device_tensor->GetDeviceType() << " and " << data
                    << " type:" << data->GetDeviceType() << " for copy actor:" << GetAID();
      DeviceTensorCopyStore::GetInstance().Insert(device_tensor.get(), data);
      output_data->data_ = kernel_tensor;
      continue;
    }
    device_tensor->set_ptr(data->GetMutablePtr());
    kernel_tensor->SetShape(output_data->data_->GetShape());
    MS_LOG(DEBUG) << "Add device tensor copy store for device address:" << device_tensor
                  << " type:" << device_tensor->GetDeviceType() << " and " << data << " type:" << data->GetDeviceType()
                  << " for copy actor:" << GetAID();
    DeviceTensorCopyStore::GetInstance().Insert(device_tensor.get(), data);
    MS_LOG(DEBUG) << "Set the ptr: " << data->GetMutablePtr() << " from device address:" << data
                  << " type:" << data->GetDeviceType() << " to device address:" << device_tensor
                  << " type:" << device_tensor->GetDeviceType()
                  << " for the ref formal parameter: " << formal_parameter.first->DebugString()
                  << " index:" << formal_parameter_position << " in the actor: " << GetAID().Name();
  }
}

void ControlActor::SendOutput(OpContext<KernelTensor> *const context) {
  for (const auto &pair : created_heter_kernel_tensors_) {
    created_kernel_tensors_.emplace_back(pair.second);
  }
  created_heter_kernel_tensors_.clear();
  // Send branch id.
  for (const auto &branch_id_arrow : output_branch_id_arrows_) {
    ActorDispatcher::Send(branch_id_arrow, &ControlActor::RunBranchID, output_branch_id_, context);
  }

  // Send data in base class.
  AbstractActor::SendOutput(context);

  // Send Partial.
  for (const auto &partial_arrow : output_partial_arrows_) {
    MS_EXCEPTION_IF_NULL(partial_arrow);
    if (IntToSize(partial_arrow->from_output_index_) >= input_partials_.size()) {
      std::string error_info = "Invalid partial input:" + std::to_string(partial_arrow->from_output_index_) +
                               " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    auto output_partial = input_partials_[IntToSize(partial_arrow->from_output_index_)];
    MS_EXCEPTION_IF_NULL(output_partial);
    ActorDispatcher::Send(partial_arrow->to_op_id_, &ControlActor::RunOpPartial, output_partial,
                          IntToSize(partial_arrow->to_input_index_), context);
  }

  // Update the start time in end actor.
  for (const auto &actor : end_actors_) {
    MS_EXCEPTION_IF_NULL(actor);
    actor->set_start_time(GetTime());
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
