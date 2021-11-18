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

#include "runtime/framework/actor/control_flow/stack_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/control_node_parser.h"

namespace mindspore {
namespace runtime {
StackActor::StackActor(const std::string &name, const std::vector<KernelWithIndex> &parameters)
    : ControlActor(name, KernelTransformType::kStackActor, parameters, nullptr) {
  input_device_tensors_.resize(parameters.size());
}

void StackActor::Init() {
  ControlActor::Init();
  for (const auto &formal_parameter : formal_parameters_) {
    if (AnfAlgo::IsCallNode(formal_parameter.first)) {
      break;
    }
    ++input_parameter_data_num_;
  }
  input_datas_num_ = formal_parameters_.size() - input_parameter_data_num_;
  if (input_parameter_data_num_ < device_tensor_store_keys_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input parameter data num:" << input_parameter_data_num_
                      << " device store num:" << device_tensor_store_keys_.size() << " for actor:" << GetAID();
  }
  input_parameter_data_num_ -= device_tensor_store_keys_.size();
}

void StackActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  auto &sequential_num = context->sequential_num_;
  // The parameters from the inside of the subgraph need to be put into the stack.
  if (IntToSize(input_data->index_) < input_parameter_data_num_ + device_tensor_store_keys_.size()) {
    input_parameter_data_[sequential_num][input_data->index_].push(input_data->data_);
  } else {
    // The outputs of call nodes are placed directly in the input data.
    input_op_datas_[sequential_num].emplace_back(input_data);
  }
  if (CheckRunningCondition(context)) {
    Run(context);
  }
}

bool StackActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  if (!ControlActor::CheckRunningCondition(context)) {
    return false;
  }

  if (input_parameter_data_num_ != 0) {
    const auto &data_iter = input_parameter_data_.find(context->sequential_num_);
    if (data_iter == input_parameter_data_.end()) {
      return false;
    }
    if (data_iter->second.size() != input_parameter_data_num_) {
      return false;
    }

    auto iter = input_branch_ids_.find(context->sequential_num_);
    if (iter == input_branch_ids_.end() || iter->second.empty()) {
      MS_LOG(ERROR) << "There is no branch id for actor:" << GetAID();
    }
    size_t branch_id_size = iter->second.size();
    if (std::any_of(data_iter->second.begin(), data_iter->second.end(),
                    [branch_id_size](const auto &one_stack) { return one_stack.second.size() != branch_id_size; })) {
      return false;
    }
  }
  return true;
}

void StackActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (input_parameter_data_num_ != 0) {
    const auto &data_iter = input_parameter_data_.find(context->sequential_num_);
    if (data_iter == input_parameter_data_.end()) {
      MS_LOG(ERROR) << "Invalid input for actor:" << GetAID();
    }
    for (const auto &one_stack : data_iter->second) {
      if (one_stack.first >= input_parameter_data_num_ + device_tensor_store_keys_.size()) {
        MS_LOG(ERROR) << "Invalid input index:" << one_stack.first << " need:" << input_parameter_data_num_
                      << " for actor:" << GetAID();
      }
      input_device_tensors_[one_stack.first] = one_stack.second.top();
    }
  }
  ControlActor::FetchInput(context);
}

void StackActor::EraseInput(const OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::EraseInput(context);

  if (input_parameter_data_num_ != 0) {
    const auto &data_iter = input_parameter_data_.find(context->sequential_num_);
    if (data_iter == input_parameter_data_.end()) {
      MS_LOG(ERROR) << "Invalid input for actor:" << GetAID();
    }

    for (auto &one_stack : data_iter->second) {
      if (one_stack.second.empty()) {
        MS_LOG(ERROR) << "Input index:" << one_stack.first << " is null in actor:" << GetAID();
      }
      one_stack.second.pop();
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
