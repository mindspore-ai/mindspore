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

#include "runtime/graph_scheduler/actor/control_flow/condition_gather_actor.h"

namespace mindspore {
namespace runtime {
ConditionGatherActor::ConditionGatherActor(const std::string &name, const CNodePtr &kernel,
                                           const DeviceContext *device_context, const AID &memory_manager_aid,
                                           const AID *debug_aid, const AID *recorder_aid,
                                           GraphExecutionStrategy strategy,
                                           const std::set<size_t> &modifiable_ref_input_indexes,
                                           const std::set<size_t> &modifiable_ref_output_indexes,
                                           const KernelTransformType &type)
    : KernelActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                  modifiable_ref_input_indexes, modifiable_ref_output_indexes, type) {}

void ConditionGatherActor::RunBranchName(const std::string &branch_name, OpContext<DeviceTensor> *const context) {
  MS_LOG(DEBUG) << "Condition gather actor:" << GetAID() << " branch name:" << branch_name;
  current_branch_name_ = branch_name;
  if (branch_name_to_input_data_num_.find(current_branch_name_) == branch_name_to_input_data_num_.end()) {
    input_datas_num_ = 0;
  } else {
    input_datas_num_ = branch_name_to_input_data_num_[current_branch_name_];
  }
  if (branch_name_to_input_control_num_.find(current_branch_name_) == branch_name_to_input_control_num_.end()) {
    input_controls_num_ = 0;
  } else {
    input_controls_num_ = branch_name_to_input_control_num_[current_branch_name_];
  }
  if (input_datas_num_ == 0 && input_controls_num_ == 0) {
    MS_LOG(EXCEPTION) << "No input data and input control, branch id:" << current_branch_name_
                      << " for actor:" << GetAID();
  }
  MS_LOG(DEBUG) << "Input data num:" << input_datas_num_ << " control num:" << input_controls_num_
                << " for actor:" << GetAID();
}

void ConditionGatherActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  input_device_tensors_.resize(branch_output_num_);
  InitOutputData();
}

void ConditionGatherActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto iter = std::find(branch_names_.begin(), branch_names_.end(), current_branch_name_);
  if (iter == branch_names_.end()) {
    MS_LOG(EXCEPTION) << "Invalid current branch name:" << current_branch_name_ << " total:" << branch_names_
                      << " for actor:" << GetAID();
  }
  size_t start_index = branch_output_num_ * (iter - branch_names_.begin());

  memory_free_list_.clear();
  // Fetch input device tensor from input data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) < start_index ||
          IntToSize(input_data->index_) - start_index >= input_device_tensors_.size()) {
        std::string error_info =
          "Invalid input index:" + std::to_string(input_data->index_) + " start:" + std::to_string(start_index) +
          " total:" + std::to_string(input_device_tensors_.size()) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_EXCEPTION_IF_NULL(input_data->data_);
      input_device_tensors_[IntToSize(input_data->index_) - start_index] = input_data->data_;
      memory_free_list_.emplace_back(input_data->data_);
    }
  }

  // Fetch input device tensor from device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    if (device_tensor_store_key.first < start_index ||
        device_tensor_store_key.first - start_index >= input_device_tensors_.size()) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    auto device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                device_contexts_[0]->GetDeviceType());
    if (device_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[device_tensor_store_key.first - start_index] = device_tensor.get();
  }

  if (output_data_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "Invalid output data size:" << output_data_.size()
                      << " and output data arrow size:" << output_data_arrows_.size() << " for actor:" << GetAID();
  }

  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_data_arrows_[i]);
    MS_EXCEPTION_IF_NULL(output_data_[i].first);
    const auto &from_index = output_data_arrows_[i]->from_output_index_;
    if (IntToSize(from_index) >= input_device_tensors_.size() || from_index < 0) {
      MS_LOG(EXCEPTION) << "Invalid from index:" << from_index << " to actor:" << output_data_arrows_[i]->to_op_id_
                        << " to index:" << output_data_arrows_[i]->to_input_index_ << " for actor:" << GetAID();
    }
    if (input_device_tensors_[i] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get input device tensor index:" + std::to_string(from_index) + " failed.";
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    output_data_[i].first->data_ = input_device_tensors_[from_index];
  }
}

void ConditionGatherActor::Run(OpContext<DeviceTensor> *const context) {
  try {
    FetchInput(context);
    if (memory_free_list_.size() > 0) {
      SendMemoryFreeReq(context);
    }
    MS_LOG(DEBUG) << "My executor order log launch kernel:" << kernel_->fullname_with_scope();
    EraseInput(context);
    SendOutput(context);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info =
      "#umsg#Kernel error:#umsg#run kernel[" + kernel_->fullname_with_scope() + "] failed, exception: " + e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
  }
}
}  // namespace runtime
}  // namespace mindspore
