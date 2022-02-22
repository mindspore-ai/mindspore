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

#include "runtime/graph_scheduler/actor/abstract_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void AbstractActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  MS_EXCEPTION_IF_NULL(input_data->data_->GetPtr());
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_datas_[sequential_num].emplace_back(input_data);

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op data and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

void AbstractActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_controls_[sequential_num].emplace_back(input_control);

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input op control and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

bool AbstractActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    const auto &data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      return false;
    }
    if (data_iter->second.size() != input_datas_num_) {
      return false;
    }
  }

  if (input_controls_num_ != 0) {
    const auto &control_iter = input_op_controls_.find(context->sequential_num_);
    if (control_iter == input_op_controls_.end()) {
      return false;
    }
    if (control_iter->second.size() != input_controls_num_) {
      return false;
    }
  }
  return true;
}

void AbstractActor::EraseInput(const OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    auto ret = input_op_datas_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input data failed: " + GetAID().Name();
      // The sequential num may be invalid, can't set the promise value of context.
      MS_LOG(ERROR) << error_info << ", sequential_num: " << context->sequential_num_;
      return;
    }
  }

  if (input_controls_num_ != 0) {
    auto ret = input_op_controls_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input controls failed: " + GetAID().Name();
      // The sequential num may be invalid, can't set the promise value of context.
      MS_LOG(ERROR) << error_info << ", sequential_num: " << context->sequential_num_;
      return;
    }
  }
}

void AbstractActor::SendOutput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // Must be the execution order: send data --> send control, avoid the illegal timing problem.
  // 1.Send output data.
  if (((output_data_arrows_.size() != output_data_.size()) ||
       (output_data_arrows_.size() != output_data_nodes_.size())) &&
      (type_ < KernelTransformType::kSwitchActor)) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The size of output data arrows is not equal to the output data.");
  }
  size_t output_data_arrow_index = 0;
  for (auto &output_data : output_data_) {
    MS_EXCEPTION_IF_NULL(output_data);
    UpdateOutputData(output_data.get(), output_data_arrows_[output_data_arrow_index],
                     output_data_nodes_[output_data_arrow_index], context);
    if (output_data->op_id_.Name().find(kStackActorNameSuffix) != std::string::npos) {
      // Create a new op data for stack actor.
      auto to_stack_data =
        std::make_unique<OpData<DeviceTensor>>(output_data->op_id_, output_data->data_, output_data->index_);
      to_stack_data_.emplace_back(std::move(to_stack_data));
      ActorDispatcher::Send(output_data->op_id_, &OpActor::RunOpData, to_stack_data_.back().get(), context);
    } else {
      ActorDispatcher::Send(output_data->op_id_, &OpActor::RunOpData, output_data.get(), context);
    }
    ++output_data_arrow_index;
  }

  // 2.Send output control.
  if (output_control_arrows_.size() > 0) {
    auto from_aid = const_cast<AID *>(&GetAID());
    for (auto &output_control : output_control_arrows_) {
      ActorDispatcher::Send(output_control, &OpActor::RunOpControl, from_aid, context);
    }
  }

  // 3.Send recorder info.
  SendRecorderInfo(context);

  // No output.
  if ((output_data_arrows_.size() == 0) && (output_control_arrows_.size() == 0) &&
      (type_ < KernelTransformType::kSwitchActor)) {
    SET_OPCONTEXT_SUCCESS_RET((*context));
  }
}
}  // namespace runtime
}  // namespace mindspore
