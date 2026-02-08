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

#include "backend/ms_backend/runtime/actors/base/abstract_actor.h"
#include "backend/ms_backend/runtime/actors/base/output_actor.h"
#include "backend/ms_backend/runtime/graph_executor/kernel_capture/graph_capture_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {

std::atomic<int64_t> gActorId = 0;

AbstractActor::AbstractActor(const std::string &name, KernelTransformType type, const AID *recorder_aid)
    : OpRTActor(name),
      type_(type),
      recorder_aid_(recorder_aid),
      actor_id_(++gActorId),
      input_datas_num_(0),
      input_controls_num_(0),
      running_dependent_msg_num_(0),
      parent_fusion_actor_{nullptr},
      memory_alloc_insert_position_{nullptr},
      memory_free_insert_position_{nullptr},
      enable_input_optimize_(EnableInputOptimize()) {}

void AbstractActor::RunOpData(OpData<KernelTensor> *const input_data, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  // The unused data may be invalid ptr.
  const auto &device_tensor = input_data->data_->device_address().get();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (!ActorDispatcher::enable_async_launch_kernel() && !device_tensor->IsPtrValid() &&
      (!TEST_FLAG(input_data->data_->flag(), device::kDeviceAddressFlagNotUsed) &&
       !TEST_FLAG(input_data->data_->flag(), device::kDeviceAddressFlagNullptr))) {
    std::stringstream error_info;
    error_info << "The input_data does not have a valid ptr of actor:" << GetAID().Name()
               << " with index:" << input_data->index_ << ", kernel tensor:" << input_data->data_->ToString();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info.str());
  }
  auto &sequential_num = context->sequential_num_;
  (void)input_op_datas_[sequential_num].emplace_back(input_data);

  auto is_run = CheckRunningCondition(context);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Actor(" << GetAID().Name() << ") receive the input op data and check running condition:" << is_run
    << ", sequential num:" << sequential_num << " input index:" << input_data->index_
    << " input data:" << input_data->data_->ToString();
  if (is_run) {
    Run(context);
  }
}

void AbstractActor::RunOpControl(AID *const input_control, OpContext<KernelTensor> *const context) {
  auto &sequential_num = context->sequential_num_;
  (void)input_op_controls_[sequential_num].emplace_back(input_control);

  auto is_run = CheckRunningCondition(context);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR_MSG)
    << "Actor(" << GetAID().Name()
    << ") receive the input op control from:" << (input_control == nullptr ? "null" : input_control->Name())
    << " and check running condition:" << is_run << ", sequential num:" << sequential_num;
  if (is_run) {
    Run(context);
  }
}

void AbstractActor::RunBatchOpData(std::vector<OpData<KernelTensor> *> *const batch_input_data,
                                   OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(batch_input_data);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR_MSG)
    << "Actor(" << GetAID().Name() << ") receive the batch input op data, sequential num:" << context->sequential_num_;
  for (auto &input_data : *batch_input_data) {
    RunOpData(input_data, context);
  }
}

bool AbstractActor::CheckRunningCondition(const OpContext<KernelTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    const auto &data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      return false;
    }
    if (data_iter->second.size() < input_datas_num_) {
      return false;
    } else if (data_iter->second.size() > input_datas_num_) {
      MS_LOG(ERROR) << "Invalid input data num:" << data_iter->second.size() << " need:" << input_datas_num_
                    << " for actor:" << GetAID() << ", sequential num:" << context->sequential_num_;
      return false;
    }
  }

  if (input_controls_num_ != 0) {
    const auto &control_iter = input_op_controls_.find(context->sequential_num_);
    if (control_iter == input_op_controls_.end()) {
      return false;
    }
    if (control_iter->second.size() < input_controls_num_) {
      return false;
    } else if (control_iter->second.size() > input_controls_num_) {
      MS_LOG(ERROR) << "Invalid input control num:" << control_iter->second.size() << " need:" << input_controls_num_
                    << " for actor:" << GetAID() << ", sequential num:" << context->sequential_num_;
      return false;
    }
  }
  return true;
}

void AbstractActor::EraseInput(const OpContext<KernelTensor> *context) {
  (void)input_op_datas_.erase(context->sequential_num_);
  (void)input_op_controls_.erase(context->sequential_num_);
}

void AbstractActor::FetchInputByTensorStore(
  std::vector<KernelTensor *> *const input_launch_tensors, std::vector<KernelTensorPtr> *const input_kernel_tensors,
  std::vector<abstract::AbstractBasePtr> *const input_kernel_tensors_for_infer,
  std::vector<KernelTensorPtr> *const memory_free_tensors, OpContext<KernelTensor> *const context) const {
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    const auto &kernel_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                       device_contexts_[0]->GetDeviceType());
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    // Collect the input kernel tensor.
    if (input_launch_tensors && input_kernel_tensors && input_kernel_tensors_for_infer &&
        ((*input_kernel_tensors)[device_tensor_store_key.first] != kernel_tensor)) {
      (*input_launch_tensors)[device_tensor_store_key.first] = kernel_tensor.get();
      (*input_kernel_tensors)[device_tensor_store_key.first] = kernel_tensor;
      (*input_kernel_tensors_for_infer)[device_tensor_store_key.first] = kernel_tensor;
      (*memory_free_tensors)[device_tensor_store_key.first] = kernel_tensor;
    }
  }
}

void AbstractActor::FetchParameterByTensorStore(
  std::vector<KernelTensor *> *const input_launch_tensors, std::vector<KernelTensorPtr> *const input_kernel_tensors,
  std::vector<abstract::AbstractBasePtr> *const input_kernel_tensors_for_infer,
  std::vector<KernelTensorPtr> *const memory_free_tensors, OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kInputProcess, "FetchParameter", true);
  for (const auto &parameter_index : parameter_indexs_) {
    // Collect the input kernel tensor.
    auto kernel_tensor = FetchParameter(parameter_index.second, GetAID());
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    if (input_launch_tensors && input_kernel_tensors && input_kernel_tensors_for_infer &&
        ((*input_kernel_tensors)[parameter_index.first] != kernel_tensor)) {
      (*input_launch_tensors)[parameter_index.first] = kernel_tensor.get();
      (*input_kernel_tensors)[parameter_index.first] = kernel_tensor;
      (*input_kernel_tensors_for_infer)[parameter_index.first] = kernel_tensor;
      (*memory_free_tensors)[parameter_index.first] = kernel_tensor;
    }
  }
}

void AbstractActor::InitOutputData() {
  mindspore::HashMap<std::string, size_t> batch_op_count;
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    auto data = std::make_unique<OpData<KernelTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
    auto &to_op_name = data_arrow->to_op_id_.Name();

    // Identify whether the output data flag is kOutputDataFlagToStack.
    bool is_to_stack = (to_op_name.find(kStackActorNameSuffix) != std::string::npos);
    size_t output_data_flag = is_to_stack ? kOutputDataFlagToStack : kOutputDataFlagInit;

    // Add the batch output data.
    if (TEST_FLAG(data_arrow->flag_, kOutputDataFlagBatch)) {
      if (is_to_stack) {
        MS_LOG(EXCEPTION) << "Not support the batch output data to stack actor.";
      }
      (void)batch_output_data_[to_op_name].emplace_back(data.get());

      SET_FLAG(output_data_flag, kOutputDataFlagBatch);
      // Identify whether the output data flag is kOutputDataFlagLastBatch.
      ++(batch_op_count[to_op_name]);
      if (batch_op_count[to_op_name] == batch_output_data_arrows_[to_op_name].size()) {
        SET_FLAG(output_data_flag, kOutputDataFlagLastBatch);
      }
    }

    // Add the internal fusion flag.
    if (TEST_FLAG(data_arrow->flag_, kOutputDataFlagBetweenFusion)) {
      SET_FLAG(output_data_flag, kOutputDataFlagBetweenFusion);
    }

    // Add the fusion flag.
    if (TEST_FLAG(data_arrow->flag_, kOutputDataFlagToFusion)) {
      SET_FLAG(output_data_flag, kOutputDataFlagToFusion);
    }

    // Add the output data.
    (void)output_data_.emplace_back(std::make_pair(std::move(data), output_data_flag));
  }
}

void AbstractActor::SendOutputData(
  OpContext<KernelTensor> *const context, const std::vector<AnfNodePtr> &output_data_nodes,
  const std::vector<DataArrowPtr> &output_data_arrows,
  const std::vector<std::pair<OpDataUniquePtr<KernelTensor>, size_t>> &output_data_list,
  const mindspore::HashMap<DataArrow *, size_t> &data_arrow_to_fusion_actor_indexs,
  mindspore::HashMap<std::string, std::vector<OpData<KernelTensor> *>> *batch_output_data) {
  for (size_t i = 0; i < output_data_list.size(); ++i) {
    auto &output_data = output_data_list[i];
    auto &to_op_id = output_data.first->op_id_;
    auto &output_data_arrow = output_data_arrows[i];
    UpdateOutputData(output_data.first.get(), output_data_arrow, output_data_nodes[i], context);
    // The index of output data will be modified the real actor input index in the fusion actor, so need recovery the
    // fusion actor index before sending output data to the fusion actor.
    if (TEST_FLAG(output_data.second, kOutputDataFlagToFusion)) {
      output_data.first->index_ = SizeToInt(data_arrow_to_fusion_actor_indexs.at(output_data_arrow.get()));
    }

    if (TEST_FLAG(output_data.second, kOutputDataFlagLastBatch)) {
      // Send batch output data. As the data need update, so all data must be collected completely before sending.
      if (TEST_FLAG(output_data.second, kOutputDataFlagBetweenFusion)) {
        const auto &to_actor = FetchSubActorInFusionActor(to_op_id.Name());
        MS_EXCEPTION_IF_NULL(to_actor);
        ActorDispatcher::SendSync(to_actor, &AbstractActor::RunBatchOpData, &((*batch_output_data)[to_op_id.Name()]),
                                  context);
      } else {
        ActorDispatcher::Send(to_op_id, &AbstractActor::RunBatchOpData, &((*batch_output_data)[to_op_id.Name()]),
                              context);
      }
    } else if (TEST_FLAG(output_data.second, kOutputDataFlagToStack)) {
      // Create a new op data for stack actor.
      auto to_stack_data =
        std::make_unique<OpData<KernelTensor>>(to_op_id, output_data.first->data_, output_data.first->index_);
      (void)to_stack_data_.emplace_back(std::move(to_stack_data));
      if (TEST_FLAG(output_data.second, kOutputDataFlagBetweenFusion)) {
        const auto &to_actor = FetchSubActorInFusionActor(to_op_id.Name());
        MS_EXCEPTION_IF_NULL(to_actor);
        ActorDispatcher::SendSync(to_actor, &OpRTActor::RunOpData, to_stack_data_.back().get(), context);
      } else {
        ActorDispatcher::Send(to_op_id, &OpRTActor::RunOpData, to_stack_data_.back().get(), context);
      }
    } else if (!TEST_FLAG(output_data.second, kOutputDataFlagBatch)) {
      // The batch output data only send when the output flag is kOutputDataFlagLastBatch.
      if (TEST_FLAG(output_data.second, kOutputDataFlagBetweenFusion)) {
        const auto &to_actor = FetchSubActorInFusionActor(to_op_id.Name());
        if (to_actor == nullptr) {
          MS_LOG(EXCEPTION) << "Failed to fetch to actor:" << to_op_id << " in actor:" << GetAID();
        }
        ActorDispatcher::SendSync(to_actor, &OpRTActor::RunOpData, output_data.first.get(), context);
      } else {
        ActorDispatcher::Send(to_op_id, &OpRTActor::RunOpData, output_data.first.get(), context);
      }
    }
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "SendOutputData End";
}

void AbstractActor::SendOutput(OpContext<KernelTensor> *const context) {
  // Must be the execution order: send data --> send control, avoid the illegal timing problem.
  // 1.Send output data.
  SendOutputData(context, output_data_nodes_, output_data_arrows_, output_data_, data_arrow_to_fusion_actor_indexs_,
                 &batch_output_data_);

  // 2.Send output control.
  if (output_control_arrows_.size() > 0) {
    auto from_aid = const_cast<AID *>(&GetAID());
    for (auto &output_control : output_control_arrows_) {
      if (TEST_FLAG(output_control->flag_, kOutputDataFlagBetweenFusion)) {
        const auto &to_actor = FetchSubActorInFusionActor(output_control->to_op_id_.Name());
        ActorDispatcher::SendSync(to_actor, &OpRTActor::RunOpControl, from_aid, context);
      } else {
        ActorDispatcher::Send(output_control->to_op_id_, &OpRTActor::RunOpControl, from_aid, context);
      }
    }
  }

  // 3.Send recorder info.
  SendRecorderInfo(context);
}

void AbstractActor::IncreaseNewRefCounts(OpContext<KernelTensor> *const context) {
  std::for_each(output_data_.begin(), output_data_.end(),
                [this](const auto &pair) { IncreaseNewRefCount(pair.first.get()); });
}

void AbstractActor::IncreaseNewRefCount(const OpData<KernelTensor> *op_data) const {
  MS_EXCEPTION_IF_NULL(op_data);
  MS_EXCEPTION_IF_NULL(op_data->data_);
  op_data->data_->IncreaseNewRefCount(GetAID().Name());
  MS_LOG(DEBUG) << "Actor:" << GetAID() << " increase new ref count for:" << op_data->data_
                << " count:" << op_data->data_->new_ref_count();
}

AbstractActor *AbstractActor::FetchSubActorInFusionActor(const std::string &sub_actor_name) const {
  if (parent_fusion_actor_ == nullptr) {
    return nullptr;
  }
  return (parent_fusion_actor_->sub_actors_[sub_actor_name]).get();
}

void AbstractActor::HandleWaitMessage(OpContext<KernelTensor> *const context, const AID &from_aid) {
  MS_LOG(DEBUG) << "Actor:" << GetAID() << " receive wait message from actor:" << from_aid;
  ActorDispatcher::Send(from_aid, &AbstractActor::HandleNotifyMessage, context, GetAID());
}

bool AbstractActor::IsOutputAddressPersisted(const KernelTensorPtr &output_kernel_tensor,
                                             const KernelWithIndex &output_node) {
  MS_EXCEPTION_IF_NULL(output_node.first);
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Check persist for kernel tensor:" << output_kernel_tensor->ToString()
    << " for node:" << output_node.first->DebugString() << " full name:" << output_node.first->fullname_with_scope()
    << " index:" << output_node.second;
  // The persisted address can't be replaced.
  if (output_kernel_tensor->is_ptr_persisted()) {
    return true;
  }

  if (output_node.first->isa<ValueNode>()) {
    return true;
  }

  // The device address of parameter may come from the device address of input tensor.
  // In order to avoid mistakenly cleaning up the device data of input tensor, return it as persisted address.
  if (output_node.first->isa<Parameter>()) {
    return true;
  }

  // If enable kernel launch capture, the kernel output as graph output will be captured and can not release device
  // memory.
  if (GraphCaptureManager::GetInstance().GetEnableGraphCapture() &&
      GraphCaptureManager::GetInstance().IncrementGraph()) {
    return true;
  }

  // Ref node need check the origin node.
  const auto &graph = AnfAlgo::FetchKernelGraph(output_node.first.get());
  if ((graph != nullptr) && graph->IsInRefOutputMap(output_node)) {
    const auto &origin_node = graph->GetRefNodeRecursive(output_node).first;
    MS_EXCEPTION_IF_NULL(origin_node);
    if (origin_node->isa<ValueNode>() || origin_node->isa<Parameter>()) {
      return true;
    }
  }
  if (output_kernel_tensor->new_ref_count() == SIZE_MAX) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Ref count of kernel tensor:" << output_kernel_tensor->ToString() << " is max, should copy output.";
    return true;
  }
  return false;
}

void AbstractActor::InsertParameterIndexs(size_t to_kernel_idx, ParameterInfo cur_front_node_info) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Actor: " << GetAID().Name()
                                       << ", insert parameter index, to index: " << to_kernel_idx
                                       << ", parameter info.first, kernel with index: "
                                       << cur_front_node_info.first.first->DebugString() << ", "
                                       << cur_front_node_info.first.second
                                       << ", parameter info.second: " << cur_front_node_info.second;
  parameter_indexs_.push_back({to_kernel_idx, cur_front_node_info});
}
}  // namespace runtime
}  // namespace mindspore
