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

#include "runtime/framework/actor/control_flow/exit_actor.h"
#include "runtime/framework/actor/output_actor.h"

namespace mindspore {
namespace runtime {
void ExitActor::Init() {
  // Init output data in base class.
  ControlActor::Init();

  // Init output data in each output branch.
  for (size_t i = 0; i < output_branch_data_arrows_.size(); ++i) {
    auto &output_branch_data_arrows = output_branch_data_arrows_[i];
    for (auto &data_arrow : output_branch_data_arrows) {
      MS_EXCEPTION_IF_NULL(data_arrow);
      auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
      output_branch_data_[i].emplace_back(data_arrow->from_output_index_, std::move(data));
    }
  }
}

void ExitActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::FetchInput(context);
  CopyDeviceAddress();

  auto data_iter = output_branch_data_.find(output_branch_id_);
  if (data_iter != output_branch_data_.end()) {
    for (auto &output_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(output_data.second);
      MS_EXCEPTION_IF_NULL(input_device_tensors_[output_data.first]);
      output_data.second->data_ = input_device_tensors_[output_data.first];
    }
  }
}

void ExitActor::SendOutput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  // 1.Send output in base class.
  ControlActor::SendOutput(context);

  // 2.Send output control in output branch.
  const auto &control_iter = output_branch_control_arrows_.find(output_branch_id_);
  if (control_iter != output_branch_control_arrows_.end()) {
    auto source_aid = const_cast<AID *>(&GetAID());
    for (const auto &control_arrow : control_iter->second) {
      ActorDispatcher::Send(control_arrow, &OpActor::RunOpControl, source_aid, context);
    }
  }

  // 2.Send output data in output branch.
  const auto &branch_data_iter = output_branch_data_.find(output_branch_id_);
  if (branch_data_iter != output_branch_data_.end()) {
    for (const auto &output_data : branch_data_iter->second) {
      MS_EXCEPTION_IF_NULL(output_data.second);
      ActorDispatcher::Send(output_data.second->op_id_, &OpActor::RunOpData, output_data.second.get(), context);
    }
  }
}

void ExitActor::CopyDeviceAddress() {
  // If node is not empty, it is the exit of funcgraph, no need to create device address.
  if (node_ != nullptr) {
    return;
  }
  std::vector<DeviceTensor *> new_device_tensors;
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    auto input_device_tensor = input_device_tensors_[i];
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    const KernelWithIndex &node_with_index = input_device_tensor->GetNodeIndex();
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    if (!node_with_index.first->isa<CNode>()) {
      continue;
    }

    MS_EXCEPTION_IF_NULL(device_contexts_[i]);
    auto new_device_tensor =
      device_contexts_[i]->CreateDeviceAddress(nullptr, input_device_tensors_[i]->GetSize(),
                                               input_device_tensors_[i]->format(), input_device_tensors_[i]->type_id());
    MS_EXCEPTION_IF_NULL(new_device_tensor);
    new_device_tensor->set_ptr(input_device_tensor->GetMutablePtr());
    new_device_tensor->set_from_mem_pool(input_device_tensor->from_mem_pool());
    new_device_tensor->SetNodeIndex(node_with_index.first, node_with_index.second);
    new_device_tensor->set_original_ref_count(SIZE_MAX);
    new_device_tensor->ResetRefCount();
    new_device_tensors.emplace_back(new_device_tensor.get());
    created_device_tensors_.emplace_back(new_device_tensor);

    input_device_tensor->set_ptr(nullptr);
    input_device_tensor->set_from_mem_pool(false);
  }
  input_device_tensors_.swap(new_device_tensors);

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
}  // namespace runtime
}  // namespace mindspore
