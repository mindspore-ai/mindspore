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

#include "runtime/framework/actor/control_flow/gather_actor.h"
#include "runtime/framework/actor/control_flow/entrance_actor.h"

namespace mindspore {
namespace runtime {
GatherActor::GatherActor(const std::string &name, const AID &memory_manager_aid,
                         const std::vector<KernelWithIndex> &parameters, const AnfNodePtr &node)
    : ControlActor(name, KernelTransformType::kGatherActor, memory_manager_aid, parameters, node) {
  device_contexts_.resize(parameters.size());
}

void GatherActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::FetchInput(context);
  // The gather actor needs to put the inputs into the first partial in order. In order to keep the index consistent,
  // the inputs need to be delayed in sequence. The offset indicates the number of delays, that is, the number of
  // inputs in the first partial.
  size_t offset = input_partials_[0]->device_tensors_.size() + input_partials_[0]->partials_.size();

  // Put other real parameter in partial.
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    const auto &device_tensor = input_device_tensors_[i];
    if (device_tensor != nullptr) {
      input_partials_[0]->device_tensors_.emplace_back(i + offset, device_tensor);
    }
  }

  for (size_t i = 1; i < input_partials_.size(); ++i) {
    if (input_partials_[i] != nullptr && input_partials_[i]->func_graph_ != nullptr) {
      auto output_partial = std::make_shared<OpPartial>();
      *output_partial = *input_partials_[i];
      input_partials_[0]->partials_.emplace_back(i + offset, output_partial);
    }
  }
}

void GatherActor::FetchOutput(OpRealParameterWithBranchID *const output, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(context);
  output->branch_id_ = output_branch_id_;
  output->device_tensors_ = input_partials_[0]->device_tensors_;
  output->partials_ = input_partials_[0]->partials_;

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

void GatherActor::SendOutput(OpContext<DeviceTensor> *const context) {
  // Send data with branch id.
  const auto &iter = output_data_with_branch_id_arrows_.find(input_partials_[0]->func_graph_);
  if (iter != output_data_with_branch_id_arrows_.end()) {
    // Build the output data struct.
    OpRealParameterWithBranchID output;
    FetchOutput(&output, context);

    for (const auto &data_with_branch_id_arrow : iter->second) {
      ActorDispatcher::Send(data_with_branch_id_arrow, &EntranceActor::RunOpRealParameterWithBranchID, output, context);
    }
  }

  // Control arrow needs to be sent after the real parameter data and branch id.
  ControlActor::SendOutput(context);
}

void GatherActor::IncreaseDynamicRefCounts(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::IncreaseDynamicRefCounts(context);

  // Increase dynamic ref count by the output data with branch id.
  const auto &iter = output_data_with_branch_id_arrows_.find(input_partials_[0]->func_graph_);
  if (iter != output_data_with_branch_id_arrows_.end()) {
    // Build the output data struct.
    OpRealParameterWithBranchID output;
    FetchOutput(&output, context);

    for (size_t i = 0; i < iter->second.size(); ++i) {
      IncreaseDynamicRefCount(output);
    }
  }
}

void GatherActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;

  std::vector<DeviceTensor *> memory_free_list;
  if (input_op_partials_.count(sequential_num) > 0) {
    for (auto &input_partial_pair : input_op_partials_[sequential_num]) {
      // All inputs in gather actor will be gathered to the first input partial. When the ref count is released,
      // only the device tensor in the first input partial needs to be released.
      if (input_partial_pair.first != 0) {
        continue;
      }
      auto partial_device_tensors = GetAllDeviceTensors(input_partial_pair.second);
      (void)std::copy(partial_device_tensors.begin(), partial_device_tensors.end(),
                      std::back_inserter(memory_free_list));
      break;
    }
  }

  if (memory_free_list.size() > 0) {
    memory_free_lists_.emplace_back(memory_free_list);
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                          device_contexts_[0], context);
  }
}
}  // namespace runtime
}  // namespace mindspore
