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
    MS_LOG(EXCEPTION) << "Invalid formal parameter:" << (node.first != nullptr ? node.first->DebugString() : "")
                      << " index:" << node.second << " for actor:" << GetAID();
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

    // Note that IncreaseDynamicRefCounts must be in front of SendMemoryFreeReq. SendMemoryFreeReq will decreasing the
    // dynamic ref count. Avoid the illegal timing problem that the dynamic reference count is decremented and then
    // incremented.
    IncreaseDynamicRefCounts(context);
    SendMemoryFreeReq(context);

    EraseInput(context);
    UpdateDynamicShapeInParameter();
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
      if ((device_tensor->GetPtr() == nullptr) &&
          (!device_context->device_res_manager_->AllocateMemory(device_tensor.get()))) {
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
    device_tensor->set_ptr(data->GetValidPtr(kDefaultStreamIndex));
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

void ControlActor::UpdateDynamicShapeInParameter() {
  for (size_t i = 0; i < backend_parameters_.size(); ++i) {
    if (backend_parameters_[i].empty() || input_device_tensors_[i] == nullptr) {
      continue;
    }

    auto node = input_device_tensors_[i]->GetNodeIndex().first;
    MS_EXCEPTION_IF_NULL(node);
    auto shape = trans::GetRuntimePaddingShape(node, input_device_tensors_[i]->GetNodeIndex().second);
    for (const auto &parameter : backend_parameters_[i]) {
      if (common::AnfAlgo::IsDynamicSequence(parameter)) {
        const auto &shapes = BaseShapeToShapeVector(node->Shape());
        std::vector<TypeId> types = std::vector(shapes.size(), input_device_tensors_[i]->type_id());
        common::AnfAlgo::SetScalarTupleOutputInferType(types, shapes, parameter);
        continue;
      }
      common::AnfAlgo::SetOutputInferTypeAndShape({input_device_tensors_[i]->type_id()}, {shape}, parameter.get());
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
