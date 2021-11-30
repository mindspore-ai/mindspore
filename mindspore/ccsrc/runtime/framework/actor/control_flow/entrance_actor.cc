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

void EntranceActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  if (is_loop_body_execution_) {
    (void)loop_body_input_op_controls_[sequential_num].emplace_back(input_control);
  } else {
    (void)input_op_controls_[sequential_num].emplace_back(input_control);
  }

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input op control and check running condition:" << is_run
                << ", loop body execution:" << is_loop_body_execution_;
  if (is_run) {
    Run(context);
  }
}

void EntranceActor::RunOpRealParameterWithBranchID(OpRealParameterWithBranchID real_parameter_with_branch_id,
                                                   OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  real_parameters_with_branch_id_[sequential_num].emplace(real_parameter_with_branch_id);

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input op data with branch id and check running condition:" << is_run
                << ", loop body execution:" << is_loop_body_execution_;
  if (is_run) {
    Run(context);
  }
}

void EntranceActor::ClearDataOnStepEnd(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  is_loop_body_execution_ = false;

  if (loop_body_input_controls_nums_ != 0) {
    loop_body_input_op_controls_.clear();
  }
}

void EntranceActor::Run(OpContext<DeviceTensor> *const context) {
  FetchInput(context);
  EraseInput(context);
  SendOutput(context);
  // The begin execution of step is false and the others execution of step is true.
  is_loop_body_execution_ = true;
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
    output_branch_id_ = real_parameters_with_branch_id_[sequential_num].front().branch_id_;
    const auto &device_tensors = real_parameters_with_branch_id_[sequential_num].front().device_tensors_;
    const auto &partials = real_parameters_with_branch_id_[sequential_num].front().partials_;

    // Collect the device tensors.
    if (device_tensors.size() + partials.size() != formal_parameters_.size()) {
      MS_LOG(ERROR) << "Invalid input num, need:" << formal_parameters_.size()
                    << " device tensor num:" << device_tensors.size() << " partial num:" << partials.size();
    }
    for (const auto &device_tensor : device_tensors) {
      if (device_tensor.first >= input_device_tensors_.size()) {
        MS_LOG(ERROR) << "Invalid device tensor index:" << device_tensor.first
                      << " vector size:" << input_device_tensors_.size() << " for actor:" << GetAID();
      }
      input_device_tensors_[device_tensor.first] = device_tensor.second;
    }

    // Collect the partials.
    for (const auto &partial : partials) {
      if (partial.first >= input_partials_.size()) {
        MS_LOG(ERROR) << "Invalid partial index:" << partial.first << " vector size:" << partials.size()
                      << " for actor:" << GetAID();
      }
      input_partials_[partial.first] = partial.second;
    }
  }

  // Init the device tensor in output data.
  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    if (output_data_by_output_index_[i].empty()) {
      continue;
    }
    const auto &data = input_device_tensors_[i];
    if (data == nullptr) {
      MS_LOG(ERROR) << "Input data index:" << i << " for actor:" << GetAID() << " is empty!";
    }
    for (auto &output_data : output_data_by_output_index_[i]) {
      MS_EXCEPTION_IF_NULL(output_data);
      output_data->data_ = data;
    }
  }
}

bool EntranceActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);

  // Check the running condition in the begin execution of step.
  // The input controls and input data exist the begin execution of root graph, and there will only be one of the two.
  if (!is_loop_body_execution_) {
    if (input_controls_num_ != 0) {
      const auto &control_iter = input_op_controls_.find(context->sequential_num_);
      if ((control_iter != input_op_controls_.end()) && (control_iter->second.size() == input_controls_num_)) {
        return true;
      }
    }

    // Data comes from the data source actor.
    if (input_datas_num_ != 0) {
      const auto &data_iter = input_op_datas_.find(context->sequential_num_);
      if (data_iter != input_op_datas_.end() && data_iter->second.size() == input_datas_num_) {
        return true;
      }
    }
  }

  // Check the controls in the loop body execution of step.
  if (is_loop_body_execution_ && (loop_body_input_controls_nums_ != 0)) {
    const auto &control_iter = loop_body_input_op_controls_.find(context->sequential_num_);
    if ((control_iter == loop_body_input_op_controls_.end()) ||
        (control_iter->second.size() != loop_body_input_controls_nums_)) {
      return false;
    }
  }

  // Data comes from the gather actor.
  const auto &iter = real_parameters_with_branch_id_.find(context->sequential_num_);
  if (iter == real_parameters_with_branch_id_.end() || iter->second.empty()) {
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
  }

  const auto &control_iter = input_op_controls_.find(sequential_num);
  if (control_iter != input_op_controls_.end()) {
    input_op_controls_.erase(control_iter);
  }

  const auto &loop_body_control_iter = loop_body_input_op_controls_.find(sequential_num);
  if (loop_body_control_iter != loop_body_input_op_controls_.end()) {
    loop_body_input_op_controls_.erase(loop_body_control_iter);
  }

  const auto &iter = real_parameters_with_branch_id_.find(sequential_num);
  if (iter != real_parameters_with_branch_id_.end()) {
    iter->second.pop();
    if (iter->second.empty()) {
      real_parameters_with_branch_id_.erase(sequential_num);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
