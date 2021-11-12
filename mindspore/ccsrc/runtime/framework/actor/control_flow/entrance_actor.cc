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

#include "runtime/framework/actor/control_flow/entrance_actor.h"
#include "runtime/framework/actor/control_flow/exit_actor.h"

namespace mindspore {
namespace runtime {
constexpr size_t kEntranceInputStartPos = 1;

void EntranceActor::RunOpDataWithBranchID(std::vector<DeviceTensor *> input_data, int branch_id,
                                          OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  input_op_data_with_branch_id_[sequential_num].emplace(input_data, branch_id);

  if (CheckRunningCondition(context)) {
    Run(context);
  }
}

void EntranceActor::Run(OpContext<DeviceTensor> *const context) {
  FetchInput(context);
  EraseInput(context);
  SendOutput(context);
  // The actor needs to be disabled after the actor is running, until no actor is running in the entire funcgraph.
  is_actor_ready_ = false;
}

void EntranceActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;

  // There are two kinds of run conditions for entrance actor:
  // 1.Data comes from the data source actor, it is in the form of data arrow.
  const auto &data_iter = input_op_datas_.find(sequential_num);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) >= input_device_tensors_.size()) {
        MS_LOG(ERROR) << "The input index is out of range, need:" << input_data->index_
                      << " current:" << input_device_tensors_.size() << " for actor:" << GetAID();
      }
      MS_EXCEPTION_IF_NULL(input_data->data_);
      input_device_tensors_[input_data->index_] = input_data->data_;
    }
    // If the data comes from the data source actor, use the default branch id.
    output_branch_id_ = 0;
  } else {
    // 2.Data comes from the gather actor, it is in the form of data with branch id.
    output_branch_id_ = input_op_data_with_branch_id_[sequential_num].front().second;
    const auto &device_tensors = input_op_data_with_branch_id_[sequential_num].front().first;
    if (device_tensors.size() != formal_parameters_.size()) {
      MS_LOG(ERROR) << "Invalid input num, need:" << formal_parameters_.size() << " current:" << device_tensors.size();
    }
    input_device_tensors_ = device_tensors;
  }

  // Init the device tensor in output data.
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
}

bool EntranceActor::CheckActorStatus(const OpContext<DeviceTensor> *const context) const {
  if (is_actor_ready_) {
    return true;
  }
  // During operation, entrance actor can be enabled only when receives all control arrows.
  if (input_controls_num_ != 0) {
    const auto &control_iter = input_op_controls_.find(context->sequential_num_);
    if (control_iter != input_op_controls_.end() && control_iter->second.size() == input_controls_num_) {
      return true;
    }
  }
  return false;
}

bool EntranceActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);

  // When the entrance actor is in the disabled state, it cannot be run.
  if (!CheckActorStatus(context)) {
    return false;
  }

  // Data comes from the data source actor.
  if (input_datas_num_ != 0) {
    const auto &data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter != input_op_datas_.end() && data_iter->second.size() == input_datas_num_) {
      return true;
    }
  }

  // Data comes from the gather actor.
  const auto &iter = input_op_data_with_branch_id_.find(context->sequential_num_);
  if (iter == input_op_data_with_branch_id_.end() || iter->second.empty()) {
    return false;
  }
  return true;
}

void EntranceActor::EraseInput(const OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;

  const auto &data_iter = input_op_datas_.find(sequential_num);
  if (data_iter != input_op_datas_.end()) {
    input_op_datas_.erase(data_iter);
    return;
  }

  const auto &control_iter = input_op_controls_.find(sequential_num);
  if (control_iter != input_op_controls_.end()) {
    input_op_controls_.erase(control_iter);
  }

  const auto &iter = input_op_data_with_branch_id_.find(sequential_num);
  if (iter == input_op_data_with_branch_id_.end() || iter->second.empty()) {
    MS_LOG(ERROR) << "Cannot find input in batch op result for actor:" << GetAID();
  }

  iter->second.pop();
  if (iter->second.empty()) {
    input_op_data_with_branch_id_.erase(sequential_num);
  }
}
}  // namespace runtime
}  // namespace mindspore
