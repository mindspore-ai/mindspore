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

#include "runtime/graph_scheduler/actor/control_flow/entrance_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/exit_actor.h"

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

void EntranceActor::RunOpRealParameterWithBranchID(const OpRealParameterWithBranchID &real_parameter_with_branch_id,
                                                   OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)real_parameters_with_branch_id_[sequential_num].emplace(real_parameter_with_branch_id);

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
  MS_EXCEPTION_IF_NULL(input_control);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the message of clearing data from:" << input_control->Name() << ".";

  is_loop_body_execution_ = false;

  if (loop_body_input_controls_nums_ != 0) {
    loop_body_input_op_controls_.clear();
  }
}

void EntranceActor::Run(OpContext<DeviceTensor> *const context) {
  // The begin execution of step is false and the others execution of step is true.
  is_loop_body_execution_ = true;

  FetchInput(context);

  // Note that IncreaseDynamicRefCount must be in front of SendMemoryFreeReq. SendMemoryFreeReq will decreasing the
  // dynamic ref count. Avoid the illegal timing problem that the dynamic reference count is decremented and then
  // incremented.
  IncreaseDynamicRefCounts(context);
  SendMemoryFreeReq(context);

  EraseInput(context);
  UpdateDynamicShapeInParameter();
  SendOutput(context);
}

void EntranceActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;

  // There are two kinds of run conditions for entrance actor:
  // 1.Data comes from the data source actor, it is in the form of data arrow.
  const auto &data_iter = input_op_datas_.find(sequential_num);
  const auto &control_iter = input_op_controls_.find(sequential_num);
  if (data_iter != input_op_datas_.end() || control_iter != input_op_controls_.end()) {
    // If the data comes from the data source actor, use the default branch id.
    output_branch_id_ = 0;

    if (data_iter == input_op_datas_.end()) {
      return;
    }

    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) >= input_device_tensors_.size()) {
        std::string error_info = "The input index is out of range, need:" + std::to_string(input_data->index_) +
                                 " current:" + std::to_string(input_device_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_EXCEPTION_IF_NULL(input_data->data_);
      input_device_tensors_[IntToSize(input_data->index_)] = input_data->data_;
    }
  } else {
    // 2.Data comes from the gather actor, it is in the form of data with branch id.
    output_branch_id_ = real_parameters_with_branch_id_[sequential_num].front().branch_id_;
    const auto &device_tensors = real_parameters_with_branch_id_[sequential_num].front().device_tensors_;
    const auto &partials = real_parameters_with_branch_id_[sequential_num].front().partials_;

    // Collect the device tensors.
    if (device_tensors.size() + partials.size() != formal_parameters_.size()) {
      std::string error_info = "Invalid input num, need:" + std::to_string(formal_parameters_.size()) +
                               " device tensor num:" + std::to_string(device_tensors.size()) +
                               " partial num:" + std::to_string(partials.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    for (const auto &device_tensor : device_tensors) {
      if (device_tensor.first >= input_device_tensors_.size()) {
        std::string error_info = "Invalid device tensor index:" + std::to_string(device_tensor.first) +
                                 " vector size:" + std::to_string(input_device_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_device_tensors_[device_tensor.first] = device_tensor.second;
    }

    // Collect the partials.
    for (const auto &partial : partials) {
      if (partial.first >= input_partials_.size()) {
        std::string error_info = "Invalid partial index:" + std::to_string(partial.first) +
                                 " vector size:" + std::to_string(partials.size()) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
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
      std::string error_info = "Input data index:" + std::to_string(i) + " for actor:" + GetAID().Name() + " is empty!";
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
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
    (void)input_op_datas_.erase(data_iter);
  }

  const auto &control_iter = input_op_controls_.find(sequential_num);
  if (control_iter != input_op_controls_.end()) {
    (void)input_op_controls_.erase(control_iter);
  }

  const auto &loop_body_control_iter = loop_body_input_op_controls_.find(sequential_num);
  if (loop_body_control_iter != loop_body_input_op_controls_.end()) {
    (void)loop_body_input_op_controls_.erase(loop_body_control_iter);
  }

  const auto &iter = real_parameters_with_branch_id_.find(sequential_num);
  if (iter != real_parameters_with_branch_id_.end()) {
    iter->second.pop();
    if (iter->second.empty()) {
      (void)real_parameters_with_branch_id_.erase(sequential_num);
    }
  }
}

void EntranceActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;

  // Collect the input device tensors.
  std::vector<DeviceTensor *> memory_free_list;
  if (input_op_datas_.count(sequential_num) > 0) {
    for (auto &input_data : input_op_datas_[sequential_num]) {
      MS_EXCEPTION_IF_NULL(input_data);
      MS_EXCEPTION_IF_NULL(input_data->data_);
      (void)memory_free_list.emplace_back(input_data->data_);
    }
  }

  const auto &iter = real_parameters_with_branch_id_.find(sequential_num);
  if (iter != real_parameters_with_branch_id_.end()) {
    if (iter->second.empty()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The real parameter with branch id is empty.");
    }
    auto &real_parameters_with_branch_id = iter->second.front();
    GetAllDeviceTensors(real_parameters_with_branch_id, &memory_free_list);
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
}  // namespace runtime
}  // namespace mindspore
