/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/loop_count_actor.h"
#include "runtime/graph_scheduler/actor/data_prepare_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/recorder_actor.h"
#include "runtime/graph_scheduler/actor/debug_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/entrance_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void LoopCountActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // Need wait MemoryManagerActor running finished to avoid the illegal memory timing problem before
  // LoopCountActor exits, because other processors which are not in actor also will process device tensor.
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::Wait, context, GetAID());
}

void LoopCountActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  IncreaseLoopCount(context);
}

void LoopCountActor::IncreaseLoopCount(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  total_running_count_++;
  current_count_++;
  MS_LOG(INFO) << "Loop count actor(" << GetAID().Name() << ") running, loop count: " << loop_count_
               << ", current count: " << current_count_ << ", total running count: " << total_running_count_;

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
    return;
  }

  PostRun(context);
}

void LoopCountActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  ActorDispatcher::Send(*debug_aid_, &DebugActor::DebugOnStepEnd, context, &GetAID());
}

void LoopCountActor::SendOutput(OpContext<DeviceTensor> *const context) {
  // Send recorder info.
  if (recorder_aid_ != nullptr) {
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordOnStepEnd, context);
  }

  // Send output control.
  auto from_aid = const_cast<AID *>(&GetAID());
  for (auto &output_control : output_control_arrows_) {
    ActorDispatcher::Send(output_control, &OpActor::RunOpControl, from_aid, context);
  }

  // Send to EntranceActor to clear the data which are generated in the loop body execution.
  for (auto &entrance_aid : entrance_aids_) {
    ActorDispatcher::Send(entrance_aid, &EntranceActor::ClearDataOnStepEnd, from_aid, context);
  }

  // The LoopCountActor exits.
  if (current_count_ == loop_count_) {
    current_count_ = 0;
    return;
  }

  // Send to DataPrepareActor to trigger next step running.
  ActorDispatcher::Send(data_prepare_aid_, &OpActor::RunOpControl, from_aid, context);
}
}  // namespace runtime
}  // namespace mindspore
