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
#include "runtime/framework/actor/data_prepare_actor.h"
#include "runtime/framework/actor/output_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/actor/recorder_actor.h"
#include "runtime/framework/actor/debug_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void LoopCountActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  (void)input_op_controls_[sequential_num].emplace_back(input_control);
  if (CheckRunningCondition(context)) {
    // Need wait MemoryManagerActor running finished to avoid the illegal memory timing problem before
    // LoopCountActor exits, because other processors which are not in actor also will process device tensor.
    Async(memory_manager_aid_, &MemoryManagerActor::Wait, context, GetAID());
  }
}

void LoopCountActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  IncreaseLoopCount(context);
}

void LoopCountActor::IncreaseLoopCount(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  EraseInput(context);

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

void LoopCountActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  Async(*debug_aid_, &DebugActor::DebugOnStepEnd, context, &GetAID());
}

void LoopCountActor::OnDebugFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  SendOutput(context);
}

void LoopCountActor::SendOutput(OpContext<DeviceTensor> *const context) {
  // Send recorder info.
  if (recorder_aid_ != nullptr) {
    Async(*recorder_aid_, &RecorderActor::RecordOnStepEnd, context);
  }

  // Send loop count to output actor.
  Async(output_aid_, &OutputActor::CollectLoopCount, current_count_, context);

  // The LoopCountActor exits.
  if (current_count_ == loop_count_) {
    current_count_ = 0;
    return;
  }

  // Send to DataPrepareActor to trigger next step running.
  std::vector<std::vector<TensorPtr>> input_tensors;
  Async(data_prepare_aid_, &DataPrepareActor::PrepareData, input_tensors, context);
}
}  // namespace runtime
}  // namespace mindspore
