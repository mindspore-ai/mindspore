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

#include "runtime/framework/actor/loop_count_actor.h"
#include "runtime/framework/actor/data_source_actor.h"
#include "runtime/framework/actor/kernel_actor.h"
#include "runtime/framework/actor/output_actor.h"
#include "runtime/framework/actor/recorder_actor.h"
#include "runtime/framework/actor/debug_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void LoopCountActor::RunOpControl(AID *input_control, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  input_op_controls_[sequential_num].emplace_back(input_control);

  if (CheckLoopCountIncreaseCondition(context)) {
    IncreaseLoopCount(context);
  }
}

void LoopCountActor::CollectBranchId(const int branch_id, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  branch_id_ = branch_id;

  if (CheckLoopCountIncreaseCondition(context)) {
    IncreaseLoopCount(context);
  }
}

void LoopCountActor::SendDebugReq(OpContext<DeviceTensor> *context) {
  Async(*debug_aid_, &DebugActor::DebugOnStepEnd, context, &GetAID());
}

void LoopCountActor::OnDebugFinish(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  SendOutput(context);
}

void LoopCountActor::IncreaseLoopCount(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  auto ret = input_op_controls_.erase(sequential_num);
  if (ret == 0) {
    std::string error_info = "Erase input controls failed: " + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  total_running_count_++;
  current_count_++;
  MS_LOG(INFO) << "Loop count actor(" << GetAID().Name() << ") running, loop count: " << loop_count_
               << ", current count: " << current_count_ << ", total running count: " << total_running_count_;

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
    return;
  }

  SendOutput(context);
}

void LoopCountActor::SendOutput(OpContext<DeviceTensor> *context) {
  // Send recorder info.
  if (recorder_aid_ != nullptr) {
    Async(*recorder_aid_, &RecorderActor::RecordOnStepEnd, context);
  }

  // Send loop count to output actor.
  Async(output_aid_, &OutputActor::CollectLoopCount, current_count_, context);

  if (current_count_ == loop_count_) {
    current_count_ = 0;
    return;
  }

  // Send output control.
  for (auto &data_source_aid : data_source_aids_) {
    Async(data_source_aid, &DataSourceActor::FetchData, context);
  }
  auto source_aid = const_cast<AID *>(&GetAID());
  for (auto &kernel_aid : no_input_kernel_aids_) {
    Async(kernel_aid, &KernelActor::RunOpControl, source_aid, context);
  }
}

bool LoopCountActor::CheckLoopCountIncreaseCondition(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  if (branch_id_ == kInvalidBranchID) {
    return false;
  }

  if (branch_id_ >= SizeToInt(branch_id_to_input_controls_num_.size())) {
    MS_LOG(ERROR) << "Branch id is invalid, id:" << branch_id_;
  }
  return input_op_controls_[sequential_num].size() == branch_id_to_input_controls_num_[branch_id_];
}
}  // namespace runtime
}  // namespace mindspore
