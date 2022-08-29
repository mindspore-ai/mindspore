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

#include "runtime/graph_scheduler/actor/control_flow/gather_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/entrance_actor.h"

namespace mindspore {
namespace runtime {
GatherActor::GatherActor(const std::string &name, const AID &memory_manager_aid,
                         const std::vector<KernelWithIndex> &parameters, const AnfNodePtr &node)
    : ControlActor(name, KernelTransformType::kGatherActor, memory_manager_aid, parameters, node),
      gather_input_(nullptr) {
  device_contexts_.resize(parameters.size());
}

void GatherActor::SendOutput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(gather_input_);

  // 1.Send data with branch id.
  const auto &iter = output_data_with_branch_id_arrows_.find(gather_input_->func_graph_);
  if (iter != output_data_with_branch_id_arrows_.end()) {
    OpRealParameterWithBranchID output;
    BuildOutput(&output, context);
    for (const auto &data_with_branch_id_arrow : iter->second) {
      ActorDispatcher::Send(data_with_branch_id_arrow, &EntranceActor::RunOpRealParameterWithBranchID, output, context);
    }
  }

  // 2.Send branch id.
  for (const auto &branch_id_arrow : output_branch_id_arrows_) {
    ActorDispatcher::Send(branch_id_arrow, &ControlActor::RunBranchID, output_branch_id_, context);
  }

  // 3.Send data and control in base class. Control arrow needs to be sent after the real parameter data and branch id.
  AbstractActor::SendOutput(context);

  // 4.Send Partial.
  for (const auto &partial_arrow : output_partial_arrows_) {
    MS_EXCEPTION_IF_NULL(partial_arrow);
    if (IntToSize(partial_arrow->from_output_index_) >= input_partials_.size()) {
      std::string error_info = "Invalid partial input:" + std::to_string(partial_arrow->from_output_index_) +
                               " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    // The first partial maybe the local partial in the partial node, so the output need use the gather input partial.
    if (partial_arrow->from_output_index_ == 0) {
      ActorDispatcher::Send(partial_arrow->to_op_id_, &ControlActor::RunOpPartial, gather_input_,
                            IntToSize(partial_arrow->to_input_index_), context);
    } else {
      ActorDispatcher::Send(partial_arrow->to_op_id_, &ControlActor::RunOpPartial,
                            input_partials_[IntToSize(partial_arrow->from_output_index_)],
                            IntToSize(partial_arrow->to_input_index_), context);
    }
  }

  // 5.Destroy the gathehr input.
  gather_input_ = nullptr;
}

void GatherActor::IncreaseDynamicRefCounts(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // Build the gather input.
  GatherInput(context);

  // Increase dynamic ref count by the output data with branch id.
  MS_EXCEPTION_IF_NULL(gather_input_);
  const auto &iter = output_data_with_branch_id_arrows_.find(gather_input_->func_graph_);
  if (iter != output_data_with_branch_id_arrows_.end()) {
    for (size_t i = 0; i < iter->second.size(); ++i) {
      // The device tensors of gather input are equivalent to output, so use the gather input directly to improve the
      // performance.
      IncreaseDynamicRefCount(gather_input_);
    }
  }

  // Increase dynamic ref count by the output data.
  for (size_t i = 0; i < output_data_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_data_[i].first);
    std::string error_info = GetAID().Name() + " fetches data null, data index:" + std::to_string(i);
    MS_EXCEPTION_IF_CHECK_FAIL((output_data_[i].first->data_ != nullptr), error_info);
    IncreaseDynamicRefCount(output_data_[i].first.get());
  }

  // Increase dynamic ref count by the output partial.
  for (const auto &partial_arrow : output_partial_arrows_) {
    MS_EXCEPTION_IF_NULL(partial_arrow);
    if (IntToSize(partial_arrow->from_output_index_) >= input_partials_.size()) {
      std::string error_info = "Invalid partial input:" + std::to_string(partial_arrow->from_output_index_) +
                               " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    // The first partial maybe the local partial, so need use the gather input partial.
    if (partial_arrow->from_output_index_ == 0) {
      IncreaseDynamicRefCount(gather_input_);
    } else {
      IncreaseDynamicRefCount(input_partials_[IntToSize(partial_arrow->from_output_index_)]);
    }
  }
}

void GatherActor::GatherInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (input_partials_.empty()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input partials is empty.");
  }

  MS_EXCEPTION_IF_NULL(input_partials_[0]);
  gather_input_ = std::make_shared<OpPartial>();
  MS_EXCEPTION_IF_NULL(gather_input_);
  // The first input partial is the base of gather inputs.
  gather_input_->func_graph_ = input_partials_[0]->func_graph_;
  gather_input_->device_tensors_ = input_partials_[0]->device_tensors_;
  gather_input_->partials_ = input_partials_[0]->partials_;

  // The gather actor needs to put the inputs into the first partial in order. In order to keep the index consistent,
  // the inputs need to be delayed in sequence. The offset indicates the number of delays, that is, the number of
  // inputs in the first partial.
  size_t offset = gather_input_->device_tensors_.size() + gather_input_->partials_.size();

  // Put all the real parameters in the first partial.
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    const auto &device_tensor = input_device_tensors_[i];
    if (device_tensor != nullptr) {
      (void)gather_input_->device_tensors_.emplace_back(i + offset, device_tensor);
    }
  }

  // Put other partials in the first partial.
  for (size_t i = 1; i < input_partials_.size(); ++i) {
    if (input_partials_[i] != nullptr) {
      (void)gather_input_->partials_.emplace_back(i + offset, input_partials_[i]);
    }
  }
}

void GatherActor::BuildOutput(OpRealParameterWithBranchID *const output, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(gather_input_);
  // Copy the data from the gather input to the output.
  output->branch_id_ = output_branch_id_;
  output->device_tensors_ = gather_input_->device_tensors_;
  output->partials_ = gather_input_->partials_;

  // The first input of gather actor is the target funcgraph, which will not be sent to the entrance actor as
  // an real parameter, so the subsequent index needs to be reduced by one.
  for (auto &device_tensor : output->device_tensors_) {
    if (device_tensor.first == 0) {
      std::string error_info =
        "Invalid device tensor index:" + std::to_string(device_tensor.first) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    device_tensor.first--;
  }
  for (auto &partial : output->partials_) {
    if (partial.first == 0) {
      std::string error_info =
        "Invalid partial index:" + std::to_string(partial.first) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    partial.first--;
  }
}
}  // namespace runtime
}  // namespace mindspore
