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

#include "runtime/graph_scheduler/actor/control_flow/stack_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/control_node_parser.h"

namespace mindspore {
namespace runtime {
StackActor::StackActor(const std::string &name, const AID &memory_manager_aid,
                       const std::vector<KernelWithIndex> &parameters)
    : ControlActor(name, KernelTransformType::kStackActor, memory_manager_aid, parameters, nullptr) {
  input_device_tensors_.resize(parameters.size());
}

void StackActor::Init() {
  ControlActor::Init();
  // The stack actor has 6 parts of input :
  // 1. Directly input data.
  // 2. Direct input partial.
  // 3. Weight.
  // 4. Local tensor.
  // 5. Call input data.
  // 6. Call input partial.
  input_datas_num_ = formal_parameters_.size() - input_stack_data_num_ - input_stack_partials_num_;
  if (input_stack_data_num_ < device_tensor_store_keys_.size() + local_device_tensors_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input stack data num:" << input_stack_data_num_
                      << " device store num:" << device_tensor_store_keys_.size()
                      << " local device tensor num:" << local_device_tensors_.size()
                      << " input stack data num:" << input_stack_data_num_
                      << " input stack partial num:" << input_stack_partials_num_ << " for actor:" << GetAID();
  }

  // Fetch the total number of input partial.
  size_t total_partials_num = 0;
  for (const auto &formal_parameter : formal_parameters_) {
    MS_EXCEPTION_IF_NULL(formal_parameter.first);
    const auto &abstract = formal_parameter.first->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = common::AnfAlgo::FetchAbstractByIndex(abstract, formal_parameter.second);
    MS_EXCEPTION_IF_NULL(real_abstract);
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      total_partials_num++;
    }
  }

  // Fetch call input data num.
  input_datas_num_ = formal_parameters_.size() - total_partials_num - input_stack_data_num_;
  input_partials_num_ = total_partials_num - input_stack_partials_num_;
  // Fetch call input partial num.
  input_stack_data_num_ -= (device_tensor_store_keys_.size() + local_device_tensors_.size());
  // Check if the input num is valid.
  if (input_stack_data_num_ + input_stack_partials_num_ + input_datas_num_ + input_partials_num_ +
        device_tensor_store_keys_.size() + local_device_tensors_.size() !=
      formal_parameters_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input num, input stack data num:" << input_stack_data_num_
                      << " input stack partial num:" << input_stack_partials_num_
                      << " input data num:" << input_datas_num_ << " input partial num:" << input_partials_num_
                      << " device tensor store size:" << device_tensor_store_keys_.size()
                      << " need total size:" << formal_parameters_.size() << " for actor:" << GetAID();
  }
}

void StackActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  // The parameters from the inside of the subgraph need to be put into the stack.
  if (IntToSize(input_data->index_) < input_stack_data_num_ + device_tensor_store_keys_.size() +
                                        input_stack_partials_num_ + local_device_tensors_.size()) {
    input_stack_data_[context->sequential_num_][input_data->index_].push(input_data->data_);
  } else {
    // The outputs of call nodes are placed directly in the input data.
    (void)input_op_datas_[context->sequential_num_].emplace_back(input_data);
  }

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op data and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

void StackActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  if (control_aid_to_indexs_.find(*input_control) != control_aid_to_indexs_.end()) {
    if ((input_stack_controls_.find(sequential_num) == input_stack_controls_.end()) ||
        (input_stack_controls_[sequential_num].find(control_aid_to_indexs_[*input_control]) ==
         input_stack_controls_[sequential_num].end())) {
      input_stack_controls_[sequential_num][control_aid_to_indexs_[*input_control]] = 1;
    } else {
      input_stack_controls_[sequential_num][control_aid_to_indexs_[*input_control]]++;
    }
  } else {
    (void)input_op_controls_[sequential_num].emplace_back(input_control);
  }

  if (CheckRunningCondition(context)) {
    Run(context);
  }
}

void StackActor::RunOpPartial(const OpPartialPtr &partial, size_t position, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto self_partial = std::make_shared<OpPartial>();
  *self_partial = *partial;
  // The parameters from the inside of the subgraph need to be put into the stack.
  if (position < input_stack_data_num_ + device_tensor_store_keys_.size() + input_stack_partials_num_ +
                   local_device_tensors_.size()) {
    input_stack_partials_[context->sequential_num_][position].push(self_partial);
  } else {
    (void)input_op_partials_[context->sequential_num_].emplace_back(position, self_partial);
  }

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input op partial and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

bool StackActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  if (!ControlActor::CheckRunningCondition(context)) {
    return false;
  }

  if (CheckStackDataRunningCondition(context) && CheckStackPartialRunningCondition(context) &&
      CheckStackControlRunningCondition(context)) {
    return true;
  }
  return false;
}

bool StackActor::CheckStackDataRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  auto iter = input_branch_ids_.find(context->sequential_num_);
  bool is_branch_id_available = (iter == input_branch_ids_.end() || iter->second.empty());

  if (input_stack_data_num_ != 0) {
    const auto &data_iter = input_stack_data_.find(context->sequential_num_);
    if (data_iter == input_stack_data_.end()) {
      return false;
    }
    if (data_iter->second.size() < input_stack_data_num_) {
      return false;
    } else if (data_iter->second.size() > input_stack_data_num_) {
      MS_LOG(ERROR) << "Invalid input stack data num:" << data_iter->second.size() << " need:" << input_stack_data_num_
                    << " for actor:" << GetAID();
      return false;
    }

    if (is_branch_id_available) {
      MS_LOG(ERROR) << "There is no branch id for actor:" << GetAID().Name();
      return false;
    }
    size_t branch_id_size = iter->second.size();
    for (const auto &one_stack : data_iter->second) {
      if (one_stack.second.size() < branch_id_size) {
        return false;
      } else if (one_stack.second.size() > branch_id_size) {
        MS_LOG(ERROR) << "Invalid input stack data num:" << one_stack.second.size()
                      << " for input index:" << one_stack.first << " need:" << branch_id_size
                      << " for actor:" << GetAID();
        return false;
      }
    }
  }
  return true;
}

bool StackActor::CheckStackPartialRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  auto iter = input_branch_ids_.find(context->sequential_num_);
  bool is_branch_id_available = (iter == input_branch_ids_.end() || iter->second.empty());

  if (input_stack_partials_num_ != 0) {
    const auto &partial_iter = input_stack_partials_.find(context->sequential_num_);
    if (partial_iter == input_stack_partials_.end()) {
      return false;
    }
    if (partial_iter->second.size() < input_stack_partials_num_) {
      return false;
    } else if (partial_iter->second.size() > input_stack_partials_num_) {
      MS_LOG(ERROR) << "Invalid input stack partial num:" << partial_iter->second.size()
                    << " need:" << input_stack_partials_num_ << " for actor:" << GetAID();
      return false;
    }

    if (is_branch_id_available) {
      MS_LOG(ERROR) << "There is no branch id for actor:" << GetAID().Name();
      return false;
    }
    size_t branch_id_size = iter->second.size();
    for (const auto &one_stack : partial_iter->second) {
      if (one_stack.second.size() < branch_id_size) {
        return false;
      } else if (one_stack.second.size() > branch_id_size) {
        MS_LOG(ERROR) << "Invalid input stack partial num:" << one_stack.second.size()
                      << " for input index:" << one_stack.first << " need:" << branch_id_size
                      << " for actor:" << GetAID();
        return false;
      }
    }
  }
  return true;
}

bool StackActor::CheckStackControlRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  auto iter = input_branch_ids_.find(context->sequential_num_);
  bool is_branch_id_available = (iter == input_branch_ids_.end() || iter->second.empty());

  if (input_stack_controls_num_ != 0) {
    const auto &control_iter = input_stack_controls_.find(context->sequential_num_);
    if (control_iter == input_stack_controls_.end()) {
      return false;
    }
    if (control_iter->second.size() < input_stack_controls_num_) {
      return false;
    } else if (control_iter->second.size() > input_stack_controls_num_) {
      MS_LOG(ERROR) << "Invalid input stack control num:" << control_iter->second.size()
                    << " need:" << input_stack_controls_num_ << " for actor:" << GetAID();
      return false;
    }

    if (is_branch_id_available) {
      MS_LOG(ERROR) << "There is no branch id for actor:" << GetAID().Name();
      return false;
    }
    size_t branch_id_size = iter->second.size();
    for (const auto &one_stack : control_iter->second) {
      if (one_stack.second < branch_id_size) {
        return false;
      } else if (one_stack.second > branch_id_size) {
        MS_LOG(ERROR) << "Invalid input stack control num:" << one_stack.second
                      << " for input actor index:" << one_stack.first << " need:" << branch_id_size
                      << " for actor:" << GetAID();
        return false;
      }
    }
  }
  return true;
}

void StackActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (input_stack_data_num_ != 0) {
    const auto &data_iter = input_stack_data_.find(context->sequential_num_);
    if (data_iter == input_stack_data_.end()) {
      std::string error_info = "Invalid input for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    for (const auto &one_stack : data_iter->second) {
      if (one_stack.first >= input_stack_data_num_ + device_tensor_store_keys_.size() + local_device_tensors_.size() +
                               input_stack_partials_num_) {
        std::string error_info = "Invalid input index:" + std::to_string(one_stack.first) +
                                 " need:" + std::to_string(input_stack_data_num_) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_EXCEPTION_IF_NULL(one_stack.second.top());
      input_device_tensors_[one_stack.first] = one_stack.second.top();
    }
  }

  if (input_stack_partials_num_ != 0) {
    const auto &partial_iter = input_stack_partials_.find(context->sequential_num_);
    if (partial_iter == input_stack_partials_.end()) {
      std::string error_info = "Invalid input for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    for (const auto &one_stack : partial_iter->second) {
      if (one_stack.first >= input_stack_data_num_ + device_tensor_store_keys_.size() + local_device_tensors_.size() +
                               input_stack_partials_num_) {
        std::string error_info = "Invalid input index:" + std::to_string(one_stack.first) +
                                 " need:" + std::to_string(input_stack_partials_num_) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_partials_[one_stack.first] = one_stack.second.top();
    }
  }
  ControlActor::FetchInput(context);
}

void StackActor::EraseInput(const OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::EraseInput(context);

  if (input_stack_data_num_ != 0) {
    const auto &data_iter = input_stack_data_.find(context->sequential_num_);
    if (data_iter == input_stack_data_.end()) {
      MS_LOG(ERROR) << "Invalid input for actor:" << GetAID();
      return;
    }

    for (auto &one_stack : data_iter->second) {
      if (one_stack.second.empty()) {
        MS_LOG(ERROR) << "Input index:" << one_stack.first << " is null in actor:" << GetAID();
        return;
      }
      one_stack.second.pop();
    }
  }

  if (input_stack_partials_num_ != 0) {
    const auto &partial_iter = input_stack_partials_.find(context->sequential_num_);
    if (partial_iter == input_stack_partials_.end()) {
      MS_LOG(ERROR) << "Invalid input for actor:" << GetAID();
      return;
    }

    for (auto &one_stack : partial_iter->second) {
      if (one_stack.second.empty()) {
        MS_LOG(ERROR) << "Input index:" << one_stack.first << " is null in actor:" << GetAID();
        return;
      }
      one_stack.second.pop();
    }
  }

  if (input_stack_controls_num_ != 0) {
    const auto &control_iter = input_stack_controls_.find(context->sequential_num_);
    if (control_iter == input_stack_controls_.end()) {
      MS_LOG(ERROR) << "Invalid input for actor:" << GetAID();
      return;
    }

    mindspore::HashMap<size_t, size_t> tmp_stack_controls;
    for (auto stack_iter = control_iter->second.begin(); stack_iter != control_iter->second.end(); ++stack_iter) {
      if (stack_iter->second == 0) {
        MS_LOG(ERROR) << "Input stack control aid:" << stack_iter->first << " is null in actor:" << GetAID();
        return;
      } else if (stack_iter->second == 1) {
        continue;
      } else {
        tmp_stack_controls[stack_iter->first] = stack_iter->second - 1;
      }
    }
    if (tmp_stack_controls.empty()) {
      (void)input_stack_controls_.erase(control_iter);
    } else {
      control_iter->second.swap(tmp_stack_controls);
    }
  }
}

void StackActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;

  // Collect the input device tensors.
  std::vector<DeviceTensor *> memory_free_list;
  if (input_op_datas_.find(sequential_num) != input_op_datas_.end()) {
    for (auto &input_data : input_op_datas_[sequential_num]) {
      MS_EXCEPTION_IF_NULL(input_data);
      MS_EXCEPTION_IF_NULL(input_data->data_);
      (void)memory_free_list.emplace_back(input_data->data_);
    }
  }

  if (input_op_partials_.find(sequential_num) != input_op_partials_.end()) {
    for (auto &input_partial_pair : input_op_partials_[sequential_num]) {
      GetAllDeviceTensors(input_partial_pair.second, &memory_free_list);
    }
  }

  if ((input_stack_data_num_ != 0) && (input_stack_data_.count(sequential_num) > 0)) {
    for (auto &stack_data_pair : input_stack_data_[sequential_num]) {
      if (!stack_data_pair.second.empty()) {
        (void)memory_free_list.emplace_back(stack_data_pair.second.top());
      }
    }
  }

  if ((input_stack_partials_num_ != 0) && (input_stack_partials_.count(sequential_num) > 0)) {
    for (auto &stack_partial_pair : input_stack_partials_[sequential_num]) {
      if (!stack_partial_pair.second.empty()) {
        GetAllDeviceTensors(stack_partial_pair.second.top(), &memory_free_list);
      }
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
}  // namespace runtime
}  // namespace mindspore
