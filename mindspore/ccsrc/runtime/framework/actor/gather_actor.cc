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

#include "runtime/framework/actor/gather_actor.h"
#include "runtime/framework/actor/output_actor.h"
#include "runtime/framework/actor/switch_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/actor/loop_count_actor.h"
#include "mindrt/include/async/async.h"
#include "abstract/utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void GatherActor::Init() {
  input_datas_num_ = data_nodes_.size();
  input_device_tensors_.resize(input_datas_num_);
  output_data_by_output_index_.resize(input_datas_num_);

  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) >= input_datas_num_) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID().Name();
    }

    auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
    (void)output_data_.emplace_back(data.get());
    (void)output_data_by_output_index_[IntToSize(data_arrow->from_output_index_)].emplace_back(std::move(data));
  }
}

size_t GatherActor::FetchDataNodePosition(const KernelWithIndex &data_node) const {
  const auto &iter = find(data_nodes_.begin(), data_nodes_.end(), data_node);
  if (iter == data_nodes_.end()) {
    MS_LOG(EXCEPTION) << "Data node: " << AnfAlgo::GetNodeDebugString(data_node.first) << " index:" << data_node.second
                      << " is not exist in gather actor:" << GetAID();
  }
  return iter - data_nodes_.begin();
}

void GatherActor::RunOpData(OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  input_data_[sequential_num][input_data->index_].push(input_data->data_);

  if (CheckLaunchCondition(context)) {
    FetchInputDeviceTensor(context);
    EraseInput(context);
    SendOutput(context);
  }
}

void GatherActor::RunOpControl(AID *input_control, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_controls_[sequential_num].emplace_back(input_control);

  if (CheckLaunchCondition(context)) {
    FetchInputDeviceTensor(context);
    EraseInput(context);
    SendOutput(context);
  }
}

void GatherActor::CollectBranchId(const int branch_id, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  input_branch_ids_[sequential_num] = branch_id;
  if (CheckLaunchCondition(context)) {
    FetchInputDeviceTensor(context);
    EraseInput(context);
    SendOutput(context);
  }
}

void GatherActor::FetchBackendInputNode(const FuncGraphPtr &func_graph, const ControlNodeParserPtr &parser) {
  for (const auto &input : func_graph->get_inputs()) {
    // Monad input would not send to gather actor.
    if (HasAbstractMonad(input) ||
        (input->isa<Parameter>() && AnfAlgo::IsParameterWeight(input->cast<ParameterPtr>()))) {
      continue;
    }
    front_to_backend_parameter_[input] = parser->GetBackendInputByParameter(input);
  }
}

void GatherActor::SendOutput(OpContext<DeviceTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(context);
  // Must be the execution order: send branch id --> send result --> send data --> send control, avoid the illegal
  // timing problem.
  // 1.Send output branch id.
  if (find(output_branch_arrows_.begin(), output_branch_arrows_.end(), switch_aid_) != output_branch_arrows_.end()) {
    int branch_id = input_branch_id_;
    Async(switch_aid_, &SwitchActor::CollectBranchId, branch_id, context);
  }
  if (find(output_branch_arrows_.begin(), output_branch_arrows_.end(), gather_aid_) != output_branch_arrows_.end()) {
    Async(gather_aid_, &GatherActor::CollectBranchId, local_branch_id_, context);
  }

  // 2.Send output result.
  for (const auto &result_arrow : output_result_arrows_) {
    MS_EXCEPTION_IF_NULL(result_arrow);
    size_t from_index = IntToSize(result_arrow->from_output_index_);
    const auto &front_node = data_nodes_[from_index].first;
    for (const auto &backend_node : front_to_backend_parameter_.at(front_node)) {
      if (AnfAlgo::GetMutableOutputAddr(backend_node.first, backend_node.second, false).get() ==
          input_device_tensors_[from_index]) {
        Async(result_arrow->to_op_id_, &OutputActor::CollectOutput, backend_node.first, backend_node.second,
              result_arrow->to_input_index_, context);
        break;
      }
    }
  }

  // 3.Send output data.
  for (auto &output_data : output_data_) {
    MS_EXCEPTION_IF_NULL(output_data);
    Async(output_data->op_id_, &OpActor::RunOpData, output_data, context);
  }

  // 4.Send output control.
  auto source_aid = const_cast<AID *>(&GetAID());
  for (auto &output_control : output_control_arrows_) {
    Async(output_control, &OpActor::RunOpControl, source_aid, context);
  }
}

void GatherActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto data_iter = input_data_.find(context->sequential_num_);
  if (data_iter != input_data_.end()) {
    for (auto &input_data : data_iter->second) {
      input_device_tensors_[input_data.first] = input_data.second.top();
      input_data.second.pop();
    }
  }

  for (const auto &device_tensor_store_key : device_tensor_store_keys_) {
    const auto &device_context = device_contexts_[device_tensor_store_key.first];
    MS_EXCEPTION_IF_NULL(device_context);
    auto device_tensor =
      DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second, device_context->GetDeviceAddressType());
    if (device_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(device_context->GetDeviceAddressType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[device_tensor_store_key.first] = device_tensor;
  }

  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    const auto &data = input_device_tensors_[i];
    for (auto &output_data : output_data_by_output_index_[i]) {
      MS_EXCEPTION_IF_NULL(output_data);
      output_data->data_ = data;
    }
  }

  if (need_branch_id_input_) {
    input_branch_id_ = input_branch_ids_[context->sequential_num_];
  }
}

bool GatherActor::CheckLaunchCondition(OpContext<DeviceTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(context);

  // Fetch input data.
  if (input_datas_num_ != 0) {
    auto data_iter = input_data_.find(context->sequential_num_);
    if (data_iter == input_data_.end()) {
      return false;
    }
    if (data_iter->second.size() + device_tensor_store_keys_.size() != input_datas_num_) {
      return false;
    }
    if (std::any_of(data_iter->second.begin(), data_iter->second.end(),
                    [](const auto &input_stack) { return input_stack.second.empty(); })) {
      return false;
    }
  }

  // Fetch input control.
  if (input_controls_num_ != 0) {
    auto control_iter = input_op_controls_.find(context->sequential_num_);
    if (control_iter == input_op_controls_.end()) {
      return false;
    }
    if (control_iter->second.size() != input_controls_num_) {
      return false;
    }
  }

  // Fetch input branch id.
  if (need_branch_id_input_) {
    auto branch_id_iter = input_branch_ids_.find(context->sequential_num_);
    if (branch_id_iter == input_branch_ids_.end()) {
      return false;
    }
  }
  return true;
}

void GatherActor::EraseInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  // Erase input data.
  auto data_iter = input_data_.find(context->sequential_num_);
  if (data_iter != input_data_.end() && std::all_of(data_iter->second.begin(), data_iter->second.end(),
                                                    [](const auto &input_data) { return input_data.second.empty(); })) {
    auto ret = input_data_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input data failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }

  // Erase input control.
  if (input_controls_num_ != 0) {
    auto ret = input_op_controls_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input controls failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }

  // Erase input branch id.
  if (need_branch_id_input_) {
    auto ret = input_branch_ids_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input branch id failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
