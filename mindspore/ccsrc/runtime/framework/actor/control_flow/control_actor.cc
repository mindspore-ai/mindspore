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

#include "runtime/framework/actor/control_flow/control_actor.h"

namespace mindspore {
namespace runtime {
ControlActor::ControlActor(const std::string &name, KernelTransformType type,
                           const std::vector<KernelWithIndex> &parameters, const AnfNodePtr &node)
    : AbstractActor(name, type, nullptr), formal_parameters_(parameters), node_(node) {
  for (size_t i = 0; i < parameters.size(); ++i) {
    input_partials_.emplace_back(std::make_shared<OpPartial>());
  }
  input_device_tensors_.resize(parameters.size());
}

void ControlActor::Init() {
  output_data_by_output_index_.resize(formal_parameters_.size());
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) >= formal_parameters_.size()) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID();
    }

    auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
    (void)output_data_by_output_index_[data_arrow->from_output_index_].emplace_back(data.get());
    (void)output_data_.emplace_back(std::move(data));
  }
}

size_t ControlActor::FetchNodePosition(const KernelWithIndex &node) const {
  const auto &iter = find(formal_parameters_.begin(), formal_parameters_.end(), node);
  if (iter == formal_parameters_.end()) {
    MS_LOG(EXCEPTION) << "Invalid formal parameter:" << node.first->DebugString() << " index:" << node.second
                      << " for actor:" << GetAID();
  }
  return iter - formal_parameters_.begin();
}

void ControlActor::Run(OpContext<DeviceTensor> *const context) {
  FetchInput(context);
  EraseInput(context);
  SendOutput(context);
}

void ControlActor::RunOpPartial(OpPartialPtr partial, size_t position, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  input_op_partials_[sequential_num].emplace_back(position, partial);
  if (CheckRunningCondition(context)) {
    Run(context);
  }
}

void ControlActor::RunBranchID(int branch_id, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  input_branch_ids_[sequential_num].push(branch_id);

  if (CheckRunningCondition(context)) {
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
    if (partial_iter->second.size() != input_partials_num_) {
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
        MS_LOG(ERROR) << "Invalid index, need:" << input_data->index_ << " current:" << input_device_tensors_.size()
                      << " for actor:" << GetAID();
      }

      input_device_tensors_[input_data->index_] = input_data->data_;
    }
  }

  // Fetch input device tensor from device store.
  for (auto &local_device_tensor : local_device_tensors_) {
    MS_EXCEPTION_IF_NULL(local_device_tensor.second);
    if (local_device_tensor.first >= input_device_tensors_.size()) {
      MS_LOG(ERROR) << "Invalid local index:" << local_device_tensor.first
                    << " current:" << local_device_tensors_.size() << " for actor:" << GetAID();
    }
    input_device_tensors_[local_device_tensor.first] = local_device_tensor.second;
  }

  // Fetch input device tensor from device store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto device_context = device_contexts_[device_tensor_store_key.first];
    MS_EXCEPTION_IF_NULL(device_context);
    auto device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                device_context->GetDeviceAddressType());
    if (device_tensor == nullptr) {
      MS_LOG(ERROR) << GetAID() << " get device tensor store failed: " << device_tensor_store_key.second->DebugString();
    }

    if (device_tensor_store_key.first >= input_device_tensors_.size()) {
      MS_LOG(ERROR) << "The input index is out of range, need:" << device_tensor_store_key.first
                    << " current:" << input_device_tensors_.size() << " for actor:" << GetAID();
    }
    input_device_tensors_[device_tensor_store_key.first] = device_tensor;
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
      MS_EXCEPTION_IF_NULL(input_partial.second->func_graph_);
      if (input_partial.first >= input_partials_.size()) {
        MS_LOG(ERROR) << "Invalid partial index:" << input_partial.first << " vector size:" << input_partials_.size()
                      << " for actor:" << GetAID();
      }
      input_partials_[input_partial.first] = input_partial.second;
    }
  }
  // Fetch input partial from local partial.
  for (const auto &local_partial : local_partials_) {
    if (local_partial.first >= input_partials_.size()) {
      MS_LOG(ERROR) << "Invalid partial index:" << local_partial.first << " vector size:" << input_partials_.size()
                    << " for actor:" << GetAID();
    }
    MS_EXCEPTION_IF_NULL(local_partial.second);
    *(input_partials_[local_partial.first]) = *(local_partial.second);
  }
  // Fetch branch id in stack.
  auto iter = input_branch_ids_.find(context->sequential_num_);
  if (iter != input_branch_ids_.end() && (!iter->second.empty())) {
    output_branch_id_ = iter->second.top();
  }
}

void ControlActor::EraseInput(const OpContext<DeviceTensor> *context) {
  AbstractActor::EraseInput(context);
  const auto &sequential_num = context->sequential_num_;
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
      MS_LOG(ERROR) << "Invalid partial input:" << partial_arrow->from_output_index_
                    << " current:" << input_partials_.size() << " for actor:" << GetAID();
    }
    auto output_partial = input_partials_[partial_arrow->from_output_index_];
    MS_EXCEPTION_IF_NULL(output_partial->func_graph_);
    ActorDispatcher::Send(partial_arrow->to_op_id_, &ControlActor::RunOpPartial, output_partial,
                          IntToSize(partial_arrow->to_input_index_), context);
  }
}
}  // namespace runtime
}  // namespace mindspore
