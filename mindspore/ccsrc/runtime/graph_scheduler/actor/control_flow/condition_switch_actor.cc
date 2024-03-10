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

#include "runtime/graph_scheduler/actor/control_flow/condition_switch_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/condition_gather_actor.h"

namespace mindspore {
namespace runtime {
ConditionSwitchActor::ConditionSwitchActor(const std::string &name, const CNodePtr &kernel,
                                           const DeviceContext *device_context, const AID &memory_manager_aid,
                                           const AID *debug_aid, const AID *recorder_aid,
                                           GraphExecutionStrategy strategy,
                                           const std::set<size_t> &modifiable_ref_input_indexes,
                                           const std::set<size_t> &modifiable_ref_output_indexes,
                                           const KernelTransformType &type)
    : KernelActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                  modifiable_ref_input_indexes, modifiable_ref_output_indexes, type) {}

void ConditionSwitchActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  input_device_tensors_.resize(common::AnfAlgo::GetInputTensorNum(kernel_));

  InitOutputData();
  output_data_by_output_index_.resize(AnfAlgo::GetOutputTensorNum(kernel_));
  if (output_data_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "The output data size is wrong: " << GetAID().Name();
  }

  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    const auto &output_data = output_data_[i].first;
    const auto &data_arrow = output_data_arrows_[i];
    MS_EXCEPTION_IF_NULL(output_data);
    MS_EXCEPTION_IF_NULL(data_arrow);
    const auto &from_index = data_arrow->from_output_index_;
    if (IntToSize(from_index) >= output_data_by_output_index_.size()) {
      MS_LOG(EXCEPTION) << "Invalid from index:" << from_index
                        << " and output size:" << output_data_by_output_index_.size() << " for actor:" << GetAID();
    }
    output_data_by_output_index_[from_index].emplace_back(output_data.get());
  }
}

void ConditionSwitchActor::SendOutput(OpContext<DeviceTensor> *const context, size_t index) {
  MS_EXCEPTION_IF_NULL(gather_aid_);
  MS_LOG(DEBUG) << "condition actor run for index:" << index << " branch name:" << branch_names_[index]
                << " for actor:" << GetAID();
  ActorDispatcher::Send(*gather_aid_, &ConditionGatherActor::RunBranchName, branch_names_[index], context);

  std::vector<AnfNodePtr> output_data_nodes;
  std::vector<std::pair<OpDataUniquePtr<DeviceTensor>, size_t>> output_data;
  std::vector<DataArrowPtr> output_data_arrows;
  if (output_data_arrows_.size() != output_data_nodes_.size() || output_data_nodes_.size() != output_data_.size() ||
      output_data_.size() != output_data_branch_indexes_.size()) {
    MS_LOG(EXCEPTION) << "Invalid data arrow size:" << output_data_arrows_.size()
                      << " node size:" << output_data_nodes_.size() << " data size:" << output_data_.size()
                      << " index size:" << output_data_branch_indexes_.size() << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < output_data_branch_indexes_.size(); ++i) {
    if (output_data_branch_indexes_[i] == index) {
      ActorDispatcher::Send(output_data_arrows_[i]->to_op_id_, &OpActor::RunOpData, output_data_[i].first.get(),
                            context);
    }
  }

  if (output_control_arrows_.size() != output_control_branch_indexes_.size()) {
    MS_LOG(EXCEPTION) << "Invalid control arrow size:" << output_control_arrows_.size()
                      << output_control_branch_indexes_.size() << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < output_control_branch_indexes_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_control_arrows_[i]);
    if (output_control_branch_indexes_[i] == index) {
      ActorDispatcher::Send(output_control_arrows_[i]->to_op_id_, &OpActor::RunOpControl, const_cast<AID *>(&GetAID()),
                            context);
    }
  }
}

void ConditionSwitchActor::Run(OpContext<DeviceTensor> *const context) {
  try {
    FetchInput(context);
    MS_EXCEPTION_IF_NULL(input_device_tensors_[0]);
    MS_EXCEPTION_IF_NULL(input_device_tensors_[0]->kernel_tensor());
    bool index = input_device_tensors_[0]->kernel_tensor()->GetValueWithCheck<bool>();
    MS_LOG(DEBUG) << "Index:" << index << " for actor:" << GetAID();
    EraseInput(context);
    CollectMemoryFreeList(index);
    if (memory_free_list_.size() > 0) {
      SendMemoryFreeReq(context);
    }
    MS_LOG(DEBUG) << "Launch kernel:" << kernel_->fullname_with_scope();
    SendOutput(context, index);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info =
      "#umsg#Kernel error:#umsg#run kernel[" + kernel_->fullname_with_scope() + "] failed, exception: " + e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
  }
}

void ConditionSwitchActor::CollectMemoryFreeList(size_t index) {
  memory_free_list_.clear();
  memory_free_list_.insert(memory_free_list_.end(), input_device_tensors_.begin(), input_device_tensors_.end());
  for (size_t i = 0; i < branch_origin_ref_count_.size(); ++i) {
    if (i == index) {
      continue;
    }
    if (branch_origin_ref_count_[i].size() + 1 != input_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid origin ref count size:" << branch_origin_ref_count_[i]
                        << " and input size:" << input_device_tensors_.size() << " for actor:" << GetAID();
    }
    MS_LOG(DEBUG) << "Free memory for branch:" << i << " for actor:" << GetAID();
    for (size_t j = 0; j < branch_origin_ref_count_[i].size(); ++j) {
      std::fill_n(back_inserter(memory_free_list_), branch_origin_ref_count_[i][j], input_device_tensors_[j + 1]);
    }
  }
}

void ConditionSwitchActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  // Fetch input device tensor from input data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) >= input_device_tensors_.size()) {
        std::string error_info = "Invalid input index, need:" + std::to_string(input_data->index_) +
                                 " current:" + std::to_string(input_device_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_EXCEPTION_IF_NULL(input_data->data_);
      input_device_tensors_[IntToSize(input_data->index_)] = input_data->data_;
    }
  }

  // Fetch input device tensor from device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    auto device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                device_contexts_[0]->GetDeviceType());
    if (device_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    if (device_tensor_store_key.first >= input_device_tensors_.size()) {
      std::string error_info =
        "The input index is out of range, need:" + std::to_string(device_tensor_store_key.first) +
        " current:" + std::to_string(input_device_tensors_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    MS_EXCEPTION_IF_NULL(device_tensor);
    input_device_tensors_[device_tensor_store_key.first] = device_tensor.get();
  }

  if (output_data_by_output_index_.size() + 1 != input_device_tensors_.size()) {
    MS_LOG(EXCEPTION) << "Invalid output size:" << output_data_by_output_index_.size()
                      << " and input device tensor size:" << input_device_tensors_.size() << " for actor:" << GetAID();
  }

  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    if (output_data_by_output_index_[i].empty()) {
      continue;
    }
    const auto &data = input_device_tensors_[i + 1];
    MS_EXCEPTION_IF_NULL(data);
    for (auto &output_data : output_data_by_output_index_[i]) {
      MS_EXCEPTION_IF_NULL(output_data);
      output_data->data_ = data;
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
