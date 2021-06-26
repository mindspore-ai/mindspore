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
    output_data_.emplace_back(data.get());
    output_data_by_output_index_[data_arrow->from_output_index_].emplace_back(std::move(data));
  }
}

size_t GatherActor::FetchDataNodePosition(const AnfNodePtr &data_node) const {
  const auto &iter = find(data_nodes_.begin(), data_nodes_.end(), data_node);
  if (iter == data_nodes_.end()) {
    MS_LOG(EXCEPTION) << "Data node: " << data_node->fullname_with_scope()
                      << " is not exist in gather actor:" << GetAID();
  }
  return iter - data_nodes_.begin();
}

void GatherActor::RunOpData(OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);

  auto sequential_num = context->sequential_num_;
  input_op_datas_[sequential_num].emplace_back(input_data);

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

void GatherActor::SendOutput(OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);

  // Branch arrow and result arrow must be executed before the data arrow and control arrow, otherwise the output
  // actor may receive the loop count message first and cause the output to be abnormal.
  if (branch_id_ > kInvalidBranchID) {
    Async(loop_count_aid_, &LoopCountActor::CollectBranchId, branch_id_, context);
    Async(output_aid_, &OutputActor::CollectBranchId, branch_id_, context);
  }

  // Send graph output result.
  for (const auto &result_arrow : output_result_arrows_) {
    MS_EXCEPTION_IF_NULL(result_arrow);
    size_t from_index = result_arrow->from_output_index_;
    const auto &front_node = data_nodes_[from_index];

    for (const auto &backend_node : front_to_backend_parameter_.at(front_node)) {
      if (AnfAlgo::GetMutableOutputAddr(backend_node.first, backend_node.second).get() ==
          input_device_tensors_[from_index]) {
        Async(result_arrow->to_op_id_, &OutputActor::CollectOutput, backend_node.first, backend_node.second,
              result_arrow->to_input_index_, context);
        break;
      }
    }
  }

  // Send output data.
  for (auto &output_data : output_data_) {
    MS_EXCEPTION_IF_NULL(output_data);
    Async(output_data->op_id_, &OpActor::RunOpData, output_data, context);
  }

  // Send output control.
  auto source_aid = const_cast<AID *>(&GetAID());
  for (auto &output_control : output_control_arrows_) {
    Async(output_control, &OpActor::RunOpControl, source_aid, context);
  }
}

void GatherActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);

  auto data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      input_device_tensors_[input_data->index_] = input_data->data_;
    }
  }

  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    const auto &data = input_device_tensors_[i];
    for (auto &output_data : output_data_by_output_index_[i]) {
      MS_EXCEPTION_IF_NULL(output_data);
      output_data->data_ = data;
    }
  }
}

bool GatherActor::CheckLaunchCondition(OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    auto data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      return false;
    }
    if (data_iter->second.size() != input_datas_num_) {
      return false;
    }
  }

  if (input_controls_num_ != 0) {
    auto control_iter = input_op_controls_.find(context->sequential_num_);
    if (control_iter == input_op_controls_.end()) {
      return false;
    }
    if (control_iter->second.size() != input_controls_num_) {
      return false;
    }
  }
  return true;
}

void GatherActor::EraseInput(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    auto ret = input_op_datas_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input data failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }

  if (input_controls_num_ != 0) {
    auto ret = input_op_controls_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input controls failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
