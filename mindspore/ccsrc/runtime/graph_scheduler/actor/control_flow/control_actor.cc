/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/control_flow/control_actor.h"
#include "runtime/hardware/device_context_manager.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "ops/framework_ops.h"
#include "utils/profile.h"

namespace mindspore {
namespace runtime {
ControlActor::ControlActor(const std::string &name, KernelTransformType type, const AID &memory_manager_aid,
                           const std::vector<KernelWithIndex> &parameters, const AnfNodePtr &node)
    : MemoryAwareActor(name, type, nullptr, memory_manager_aid), formal_parameters_(parameters), node_(node) {
  input_partials_.resize(parameters.size());
  input_device_tensors_.resize(parameters.size());
  backend_parameters_.resize(parameters.size());
  output_data_by_output_index_.resize(parameters.size());
}

void ControlActor::Init() {
  InitOutputData();
  if (output_data_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "The output data size is wrong: " << GetAID().Name();
  }

  size_t output_data_index = 0;
  for (auto &data_arrow : output_data_arrows_) {
    auto data = output_data_[output_data_index].first.get();
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) >= output_data_by_output_index_.size()) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID();
    }
    (void)output_data_by_output_index_[IntToSize(data_arrow->from_output_index_)].emplace_back(data);
    ++output_data_index;
  }
}

void ControlActor::GetAllDeviceTensors(const OpPartialPtr &op_partial, std::vector<DeviceTensor *> *device_tensors) {
  MS_EXCEPTION_IF_NULL(op_partial);
  (void)std::transform(op_partial->device_tensors_.begin(), op_partial->device_tensors_.end(),
                       std::back_inserter(*device_tensors),
                       [](const auto &device_tensor) { return device_tensor.second; });

  // Foreach the op partial to fetch the device tensors.
  for (auto &partial : op_partial->partials_) {
    GetAllDeviceTensors(partial.second, device_tensors);
  }
}

void ControlActor::GetAllDeviceTensors(const OpRealParameterWithBranchID &op_real_parameter,
                                       std::vector<DeviceTensor *> *device_tensors) {
  MS_EXCEPTION_IF_NULL(device_tensors);
  for (auto &device_tensor : op_real_parameter.device_tensors_) {
    (void)device_tensors->emplace_back(device_tensor.second);
  }

  // Foreach the op partial to fetch the device tensors.
  for (auto &partial : op_real_parameter.partials_) {
    GetAllDeviceTensors(partial.second, device_tensors);
  }
}

void ControlActor::IncreaseDynamicRefCount(const OpData<DeviceTensor> *op_data) const {
  MS_EXCEPTION_IF_NULL(op_data);
  MS_EXCEPTION_IF_NULL(op_data->data_);
  op_data->data_->IncreaseDynamicRefCount(GetAID().Name());
}

void ControlActor::IncreaseDynamicRefCount(const OpPartialPtr &op_partial) {
  if (op_partial == nullptr) {
    MS_LOG(EXCEPTION) << "Empty op partial for actor:" << GetAID();
  }
  std::vector<DeviceTensor *> partial_device_tensors;
  GetAllDeviceTensors(op_partial, &partial_device_tensors);
  for (auto &partial_device_tensor : partial_device_tensors) {
    MS_EXCEPTION_IF_NULL(partial_device_tensor);
    partial_device_tensor->IncreaseDynamicRefCount(GetAID().Name());
  }
}

void ControlActor::IncreaseDynamicRefCount(const OpRealParameterWithBranchID &op_real_parameter) {
  std::vector<DeviceTensor *> partial_device_tensors;
  GetAllDeviceTensors(op_real_parameter, &partial_device_tensors);
  for (auto &partial_device_tensor : partial_device_tensors) {
    MS_EXCEPTION_IF_NULL(partial_device_tensor);
    partial_device_tensor->IncreaseDynamicRefCount(GetAID().Name());
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
    for (const auto &formal_parameter : formal_parameters_) {
      MS_LOG(WARNING) << "Actor:" << GetAID() << " formal parameter:"
                      << (formal_parameter.first != nullptr ? formal_parameter.first->DebugString() : "")
                      << " index:" << formal_parameter.second << " node ptr:" << formal_parameter.first;
    }
    MS_LOG_WITH_NODE(EXCEPTION, node.first)
      << "Invalid formal parameter:" << (node.first != nullptr ? node.first->DebugString() : "")
      << " node ptr:" << node.first << " index:" << node.second << " for actor:" << GetAID();
  }
  return iter - formal_parameters_.begin();
}

void ControlActor::Run(OpContext<DeviceTensor> *const context) {
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

void ControlActor::RunOpPartial(const OpPartialPtr &partial, size_t position, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_partials_[sequential_num].emplace_back(position, partial);

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input op partial and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

void ControlActor::RunBranchID(int branch_id, OpContext<DeviceTensor> *const context) {
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

bool ControlActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
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

void ControlActor::FetchInput(OpContext<DeviceTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);

  // Fetch input device tensor from input data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) >= input_device_tensors_.size()) {
        std::string error_info = "Invalid index, need:" + std::to_string(input_data->index_) +
                                 " current:" + std::to_string(input_device_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_EXCEPTION_IF_NULL(input_data->data_);
      input_device_tensors_[IntToSize(input_data->index_)] = input_data->data_;
    }
  }

  // Fetch input device tensor from local device tensor.
  for (auto &local_device_tensor : local_device_tensors_) {
    MS_EXCEPTION_IF_NULL(local_device_tensor.second);
    if (local_device_tensor.first >= input_device_tensors_.size()) {
      std::string error_info = "Invalid local index:" + std::to_string(local_device_tensor.first) +
                               " current:" + std::to_string(local_device_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[local_device_tensor.first] = local_device_tensor.second;
  }

  // Fetch input device tensor from device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto device_tensors = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get());
    if (device_tensors.empty()) {
      auto &device_context = device_contexts_[device_tensor_store_key.first];
      MS_EXCEPTION_IF_NULL(device_context);
      MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
      std::string error_info = GetAID().Name() +
                               " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
                               ", device type:" + std::to_string(static_cast<int>(device_context->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    if (device_tensor_store_key.first >= input_device_tensors_.size()) {
      std::string error_info =
        "The input index is out of range, need:" + std::to_string(device_tensor_store_key.first) +
        " current:" + std::to_string(input_device_tensors_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    MS_EXCEPTION_IF_NULL(device_tensors[0]);
    input_device_tensors_[device_tensor_store_key.first] = device_tensors[0].get();
  }

  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    if (output_data_by_output_index_[i].empty()) {
      continue;
    }
    const auto &data = input_device_tensors_[i];
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

void ControlActor::IncreaseDynamicRefCounts(OpContext<DeviceTensor> *const context) {
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

void ControlActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;

  // Collect the input device tensors.
  std::vector<DeviceTensor *> memory_free_list;
  if (input_op_datas_.count(sequential_num) > 0) {
    for (auto &input_op_data : input_op_datas_[sequential_num]) {
      MS_EXCEPTION_IF_NULL(input_op_data);
      MS_EXCEPTION_IF_NULL(input_op_data->data_);
      (void)memory_free_list.emplace_back(input_op_data->data_);
    }
  }

  if (input_op_partials_.count(sequential_num) > 0) {
    for (auto &input_op_partial : input_op_partials_[sequential_num]) {
      GetAllDeviceTensors(input_op_partial.second, &memory_free_list);
    }
  }

  if (memory_free_list.size() > 0) {
    memory_free_lists_.push(memory_free_list);
    if (ActorDispatcher::is_memory_free_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                                device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                            device_contexts_[0], context, GetAID());
    }
  }
}

void ControlActor::EraseInput(const OpContext<DeviceTensor> *context) {
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

void ControlActor::UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                                    const AnfNodePtr &, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_data);
  MS_EXCEPTION_IF_NULL(data_arrow);
  auto formal_parameter_position = data_arrow->from_output_index_;
  // Has no the ref node formal parameter.
  if (ref_node_formal_parameter_device_tensors_.count(formal_parameter_position) == 0) {
    return;
  }

  MS_EXCEPTION_IF_NULL(context);
  const auto &data = output_data->data_;
  MS_EXCEPTION_IF_NULL(data);
  if ((!data->IsPtrValid()) || (data->ref_count() != SIZE_MAX) || (data->dynamic_ref_count() != INT32_MAX)) {
    std::string error_info = "The address of the " + std::to_string(formal_parameter_position) +
                             " position real parameter is nullptr or ref count is wrong.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  // Foreach the device tensors to set the ptr from data, only the formal parameter device tensor of ref node need set
  // before kernel running, because it will be used by ref output node.
  for (auto &device_tensor : ref_node_formal_parameter_device_tensors_[formal_parameter_position]) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if ((device_tensor.get() == data) || (device_tensor->GetMutablePtr() == data->GetMutablePtr())) {
      continue;
    }
    auto formal_parameter = device_tensor->GetNodeIndex();
    MS_EXCEPTION_IF_NULL(formal_parameter.first);
    if ((device_tensor->GetSize() != data->GetSize()) || (device_tensor->type_id() != data->type_id())) {
      MS_LOG(WARNING) << "The formal parameter: " << formal_parameter.first->DebugString()
                      << " position:" << formal_parameter_position
                      << "please check the size and type id, formal parameter size:" << device_tensor->GetSize()
                      << " type id:" << device_tensor->type_id() << ", real parameter size:" << data->GetSize()
                      << " type id:" << data->type_id();
    }

    // Copy from the real parameter to formal parameter and insert the device tensor copy store.
    if ((!AnfAlgo::IsEquivalentFormat(device_tensor->format(), data->format())) ||
        (device_tensor->GetDeviceType() != data->GetDeviceType())) {
      MS_LOG(INFO) << GetAID().Name() << " the input position:" << formal_parameter_position
                   << " copy from real parameter address:" << data << ", type:" << data->GetDeviceType()
                   << ", format:" << data->format() << " to formal parameter address:" << device_tensor.get()
                   << ", type:" << device_tensor->GetDeviceType() << ", format:" << device_tensor->format()
                   << ", formal parameter name:" << formal_parameter.first->DebugString();
      const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device_tensor->device_name(), device_tensor->device_id()});
      MS_EXCEPTION_IF_NULL(device_context);
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(GetAID().Name(), device::AllocatorType::kOther, 0);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), "UpdateOutputData", "");
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(), device::tracker::MemType::kOther,
                                                     device_tensor->GetSize(), device_tensor.get());
      auto data_stream_id = data->stream_id();
      auto device_tensor_stream_id = device_tensor->stream_id();
      if (device_tensor_stream_id != data_stream_id) {
        MS_LOG(INFO) << "Rewrite device tesnor stream id from : " << device_tensor_stream_id
                     << " to data stream id : " << data_stream_id << ".";
        device_tensor->set_stream_id(data_stream_id);
      }
      if ((device_tensor->GetPtr() == nullptr) &&
          (!device_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex))) {
        SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, *device_context,
                                                    formal_parameter.first->DebugString(), device_tensor->GetSize());
      }
      if (!Copy(device_tensor.get(), data)) {
        std::string error_info = "The formal parameter: " + formal_parameter.first->DebugString() +
                                 " position:" + std::to_string(formal_parameter_position) + " copy failed.";
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      DeviceTensorCopyStore::GetInstance().Insert(device_tensor.get(), data);
      output_data->data_ = device_tensor.get();
      continue;
    }

    // Ref node may use the ptr of device tensor as the output address, so need set the ptr from data.
    device_tensor->set_ptr(data->GetMutablePtr());
    MS_LOG(DEBUG) << "Set the ptr: " << data->GetMutablePtr()
                  << " for the ref formal parameter: " << formal_parameter.first->DebugString()
                  << " in the actor: " << GetAID().Name();
  }
}

void ControlActor::SendOutput(OpContext<DeviceTensor> *const context) {
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
namespace {
CNodePtr CreateRealMakeTuple(const std::vector<DeviceTensor *> &addr_list, const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimRealMakeTuple)};
  auto new_cnode = func_graph->NewCNode(inputs);
  std::vector<std::string> formats;
  MS_EXCEPTION_IF_NULL(new_cnode);
  std::vector<abstract::AbstractBasePtr> abs_list;
  for (const auto &addr : addr_list) {
    MS_EXCEPTION_IF_NULL(addr);
    auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(addr->type_id()), addr->host_shape());
    abs_list.emplace_back(abs);
    formats.emplace_back(addr->format());
    MS_LOG(DEBUG) << "Create new abstract:" << abs->ToString();
  }
  auto tuple_abs = std::make_shared<abstract::AbstractTuple>(abs_list);
  MS_LOG(DEBUG) << "Create abstract for real make tuple:" << tuple_abs->ToString();
  // Set dynamic len element abstract to check the abstract is dynamic len.
  abstract::AbstractBasePtr element_abs = (abs_list.empty() ? std::make_shared<abstract::AbstractTensor>(
                                                                TypeIdToType(TypeId::kNumberTypeInt64), ShapeVector())
                                                            : abs_list[0]);
  tuple_abs->set_dynamic_len_element_abs(element_abs);
  new_cnode->set_abstract(tuple_abs);

  // Create kernel info for node and set format for it.
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  kernel_info->set_select_kernel_build_info(builder->Build());
  new_cnode->set_kernel_info(kernel_info);
  builder->SetOutputsFormat(formats);
  return new_cnode;
}

void CheckDeviceAddressConsist(OpContext<DeviceTensor> *const context, const std::vector<DeviceTensor *> &addr_list,
                               const std::string &actor_name) {
  MS_EXCEPTION_IF_NULL(context);
  if (addr_list.empty() || addr_list[0] == nullptr) {
    return;
  }
  // Check consistence of device address.
  const auto &shape = addr_list[0]->host_shape();
  const auto &size = addr_list[0]->GetSize();
  const auto &type = addr_list[0]->type_id();
  const auto &device_name = addr_list[0]->device_name();
  for (size_t i = 1; i < addr_list.size(); ++i) {
    MS_EXCEPTION_IF_NULL(addr_list[i]);
    if (size != addr_list[i]->GetSize() || type != addr_list[i]->type_id()) {
      MS_LOG(ERROR) << "Failed to merge two device address, addr1:" << addr_list[0] << " size:" << size
                    << " shape:" << shape << " device name:" << device_name << " type:" << type
                    << " addr2:" << addr_list[i] << " size:" << addr_list[i]->GetSize()
                    << " shape:" << addr_list[i]->host_shape() << " device name:" << addr_list[i]->device_name()
                    << " type" << addr_list[i]->type_id() << " for actor:" << actor_name;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Failed to merge two device address");
    }
    if (shape != addr_list[i]->host_shape()) {
      MS_LOG(WARNING) << "Merge two device address with different shape, addr1 shape:" << shape
                      << " addr2 shape:" << addr_list[i]->host_shape() << " for actor:" << actor_name;
    }
  }
}
}  // namespace

void ControlActor::MergeDeviceAddress(OpContext<DeviceTensor> *const context,
                                      const std::vector<DeviceTensor *> &addr_list, DeviceTensor **device_tensor) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (addr_list.empty()) {
    MergeEmptyAddressDeviceAddress(context, addr_list, device_tensor);
    return;
  }

  CheckDeviceAddressConsist(context, addr_list, GetAID().Name());
  MS_EXCEPTION_IF_NULL(addr_list[0]);
  const auto &total_size = addr_list[0]->GetSize() * addr_list.size();
  ShapeVector total_shape = {SizeToLong(addr_list.size())};
  const auto &shape = addr_list[0]->host_shape();
  total_shape.insert(total_shape.end(), shape.begin(), shape.end());
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {addr_list[0]->device_name(), addr_list[0]->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  abstract::BaseShapePtrList shape_list(addr_list.size(), addr_list[0]->kernel_tensor()->GetShape());
  auto tuple_shape = std::make_shared<abstract::TupleShape>(shape_list);
  TypePtrList type_list(addr_list.size(), addr_list[0]->kernel_tensor()->GetType());
  auto tuple_type = std::make_shared<Tuple>(type_list);
  MS_LOG(DEBUG) << "Create kernel tensor by shape:" << tuple_shape->ToString() << " type:" << tuple_type->ToString()
                << " in device address:" << addr_list[0];
  const auto &kernel_tensor = std::make_shared<kernel::KernelTensor>(
    tuple_shape, tuple_type, nullptr, nullptr, total_size, addr_list[0]->format(), addr_list[0]->type_id(), total_shape,
    device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(addr_list[0]->stream_id());
  const auto &new_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
  MS_EXCEPTION_IF_NULL(new_device_tensor);

  MS_LOG(DEBUG) << "Create device tensor:" << new_device_tensor << " type:" << new_device_tensor->type_id();
  if (!device_context->device_res_manager_->AllocateMemory(new_device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, *device_context,
                                                GetAID().Name(), new_device_tensor->GetSize());
  }
  MS_EXCEPTION_IF_NULL(new_device_tensor->GetMutablePtr());

  // Create a new real maketuple node for new device address.
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  auto new_cnode = CreateRealMakeTuple(addr_list, fg);
  AnfAlgo::SetOutputAddr(new_device_tensor, 0, new_cnode.get());
  created_new_graphs_.emplace_back(fg);
  created_new_nodes_.emplace_back(new_cnode);
  new_device_tensor->SetNodeIndex(new_cnode, 0);
  new_device_tensor->set_from_persistent_mem(addr_list[0]->from_persistent_mem());
  new_device_tensor->set_dynamic_ref_count(0);
  new_device_tensor->set_original_ref_count(SIZE_MAX);
  new_device_tensor->ResetRefCount();

  // Merge device address list into a single device address.
  auto tmp_kernel_tensor = std::make_shared<kernel::KernelTensor>(
    new_device_tensor->GetMutablePtr(), addr_list[0]->GetSize(), kernel::GetFormatFromStrToEnum(addr_list[0]->format()),
    addr_list[0]->type_id(), shape, device_context->device_context_key().device_name_,
    device_context->device_context_key().device_id_);
  tmp_kernel_tensor->set_stream_id(addr_list[0]->stream_id());
  const auto &tmp_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(tmp_kernel_tensor);
  MS_EXCEPTION_IF_NULL(tmp_device_tensor);
  MS_LOG(DEBUG) << "Create device tensor:" << tmp_device_tensor << " type:" << tmp_device_tensor->type_id();
  std::shared_ptr<int64_t> max_task_id_on_stream = nullptr;
  for (size_t i = 0; i < addr_list.size(); ++i) {
    auto device_tensor_addr = addr_list[i];
    auto task_id_on_stream = device_tensor_addr->kernel_tensor()->task_id_on_stream();
    if (task_id_on_stream != nullptr) {
      if (max_task_id_on_stream == nullptr) {
        max_task_id_on_stream = task_id_on_stream;
      } else {
        if (*max_task_id_on_stream < *task_id_on_stream) {
          max_task_id_on_stream = task_id_on_stream;
        }
      }
    }
    bool ret = false;
    if (addr_list[i]->device_name() == addr_list[0]->device_name()) {
      ret = tmp_device_tensor->SyncDeviceToDevice(addr_list[i]);
    } else if (addr_list[0]->device_name() == kCPUDevice) {
      ret = addr_list[i]->SyncDeviceToHost(addr_list[i]->GetSize(), tmp_device_tensor->GetMutablePtr());
    } else if (addr_list[i]->device_name() == kCPUDevice) {
      ret = tmp_device_tensor->SyncHostToDevice(addr_list[i]->GetSize(), addr_list[i]->GetMutablePtr());
    } else {
      MS_LOG(ERROR) << "Invalid device name for addr1:" << addr_list[0] << " name:" << addr_list[0]->device_name()
                    << " and addr2:" << addr_list[i] << " name:" << addr_list[i]->device_name();
    }
    if (!ret) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Sync device to device failed.");
    }
    tmp_device_tensor->set_ptr((reinterpret_cast<char *>(tmp_device_tensor->GetMutablePtr())) +
                               addr_list[0]->GetSize());
  }
  new_device_tensor->kernel_tensor()->set_task_id_on_stream(max_task_id_on_stream);
  tmp_device_tensor->set_ptr(nullptr);
  created_device_tensors_.emplace_back(new_device_tensor);
  MS_LOG(DEBUG) << "actor:" << GetAID() << " create new device address:" << new_device_tensor
                << " for addr list size:" << addr_list.size()
                << " device address shape:" << new_device_tensor->host_shape();
  (*device_tensor) = new_device_tensor.get();
  return;
}

void ControlActor::MergeEmptyAddressDeviceAddress(OpContext<DeviceTensor> *const context,
                                                  const std::vector<DeviceTensor *> &addr_list,
                                                  DeviceTensor **device_tensor) {
  // Create device address for empty tuple.
  // Fetch the default device context for empty sequence.
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET), context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  auto tuple_shape = std::make_shared<abstract::TupleShape>();
  auto tuple_type = std::make_shared<Tuple>();
  const auto &kernel_tensor = std::make_shared<kernel::KernelTensor>(
    tuple_shape, tuple_type, nullptr, nullptr, 0, kOpFormat_DEFAULT, TypeId::kNumberTypeInt64, ShapeVector(),
    device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
  const auto &new_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
  MS_EXCEPTION_IF_NULL(new_device_tensor);
  new_device_tensor->set_dynamic_ref_count(0);
  new_device_tensor->set_original_ref_count(SIZE_MAX);
  new_device_tensor->ResetRefCount();
  if (!device_context->device_res_manager_->AllocateMemory(new_device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, *device_context,
                                                GetAID().Name(), new_device_tensor->GetSize());
  }
  created_device_tensors_.emplace_back(new_device_tensor);
  (*device_tensor) = new_device_tensor.get();
  MS_LOG(DEBUG) << "actor:" << GetAID() << " create new device address:" << new_device_tensor << " for empty addr list";
}
}  // namespace runtime
}  // namespace mindspore
