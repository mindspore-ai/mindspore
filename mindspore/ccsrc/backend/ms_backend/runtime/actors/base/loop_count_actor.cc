/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/actors/base/loop_count_actor.h"
#include <set>
#include "backend/ms_backend/runtime/actors/base/data_prepare_actor.h"
#include "backend/ms_backend/runtime/actors/base/output_actor.h"
#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"
#include "backend/ms_backend/runtime/actors/base/recorder_actor.h"
#include "backend/ms_backend/runtime/actors/base/debug_actor.h"
#include "backend/ms_backend/runtime/actors/base/profiler_actor.h"
#include "backend/ms_backend/runtime/actors/control_flow/entrance_actor.h"
#include "include/backend/common/exec_order/online_execution_order_check.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"

namespace mindspore {
namespace runtime {
using distributed::collective::CollectiveManager;

void LoopCountActor::Run(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // Need wait MemoryManagerActor running finished to avoid the illegal memory timing problem before
  // LoopCountActor exits, because other processors which are not in actor also will process device tensor.
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::Wait, context, GetAID());
}

void LoopCountActor::OnMemoryAllocFinish(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  IncreaseLoopCount(context);
}

void LoopCountActor::IncreaseLoopCount(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  total_running_count_++;
  current_count_++;
  if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
    MS_LOG(INFO) << "Run graph failed and please check error log.";
    return;
  }
  MS_LOG(INFO) << "Loop count actor(" << GetAID().Name() << ") running, loop count: " << loop_count_
               << ", current count: " << current_count_ << ", total running count: " << total_running_count_;

  static auto &process = Process::GetInstance();
  process.CheckCommOrderIteration(total_running_count_);

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
  }

  if (profiler_aid_ != nullptr) {
    MS_LOG(INFO) << "Sync stream in the step end by profiler.";
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kStreamSync, GetAID().Name());
    SendProfilerReq(context);
  }

  if (first_control_aids_.empty() && entrance_aids_.empty()) {
    RealRun(context);
    return;
  }
  HandleNotifyOnePhase(context);
}

void LoopCountActor::RealRun(OpContext<KernelTensor> *const context) {
  notify_messages_.clear();
  // Sync device stream.
  if ((strategy_ == GraphExecutionStrategy::kPipeline) && is_need_sync_stream_) {
    MS_LOG(INFO) << "Sync stream in the step end.";
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kStreamSync, GetAID().Name());
    std::set<const DeviceContext *> sync_stream_device_contexts;
    for (auto &device_context : device_contexts_) {
      MS_EXCEPTION_IF_NULL(device_context);
      if ((sync_stream_device_contexts.count(device_context) == 0) &&
          (!device_context->device_res_manager_->SyncAllStreams())) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context),
                                          ("Sync stream failed:" + device_context->device_context_key().ToString()));
      }
      (void)sync_stream_device_contexts.insert(device_context);
    }
    MS_LOG(INFO) << "Sync stream success.";
  }

  PostRun(context);
}

void LoopCountActor::SendDebugReq(OpContext<KernelTensor> *const context) {
  ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugOnStepEnd, context, &GetAID(), total_running_count_,
                            device_contexts_);
}

void LoopCountActor::SendProfilerReq(OpContext<KernelTensor> *const context) {
  ActorDispatcher::SendSync(*profiler_aid_, &ProfilerActor::ProfilerOnStepEnd, context, &GetAID(),
                            total_running_count_);
}

void LoopCountActor::HandleNotifyOnePhase(OpContext<KernelTensor> *const context) {
  if (first_control_aids_.empty()) {
    HandleNotifyTwoPhase(context);
    return;
  }
  for (auto &first_control_aid : first_control_aids_) {
    ActorDispatcher::Send(first_control_aid, &AbstractActor::HandleWaitMessage, context, GetAID());
  }
}

void LoopCountActor::HandleNotifyTwoPhase(OpContext<KernelTensor> *const context) {
  if (entrance_aids_.empty()) {
    RealRun(context);
    return;
  }
  // Send to EntranceActor to clear the data which are generated in the loop body execution.
  for (auto &entrance_aid : entrance_aids_) {
    ActorDispatcher::Send(entrance_aid, &AbstractActor::HandleWaitMessage, context, GetAID());
  }
  return;
}

void LoopCountActor::HandleNotifyMessage(OpContext<KernelTensor> *const context, const AID &from_aid) {
  notify_messages_.emplace_back(from_aid);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR_MSG)
    << "Actor:" << GetAID() << " receive signal message from actor:" << from_aid
    << " current size:" << notify_messages_.size() << " need size:" << first_control_aids_.size() << " and"
    << entrance_aids_.size() << " for actor:" << GetAID();
  if (notify_messages_.size() < first_control_aids_.size() ||
      (notify_messages_.size() < first_control_aids_.size() + entrance_aids_.size() &&
       notify_messages_.size() > first_control_aids_.size())) {
    return;
  }

  if (notify_messages_.size() == first_control_aids_.size()) {
    MS_LOG(DEBUG) << "Handle first control aid finish for actor:" << GetAID();
    HandleNotifyTwoPhase(context);
    return;
  }

  if (notify_messages_.size() == first_control_aids_.size() + entrance_aids_.size()) {
    MS_LOG(DEBUG) << "Handle first control aid and entrance aid finish for actor:" << GetAID();
    RealRun(context);
    return;
  }

  std::stringstream ofs;
  ofs << "Invalid input signals size:" << notify_messages_.size()
      << " first control aid size:" << first_control_aids_.size() << " entrance aid size:" << entrance_aids_.size()
      << " for actor:" << GetAID();
  SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), ofs.str());
}

void LoopCountActor::SendOutput(OpContext<KernelTensor> *const context) {
  // Send recorder info.
  if (recorder_aid_ != nullptr) {
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordOnStepEnd, context);
  }

  // Send output control.
  auto from_aid = const_cast<AID *>(&GetAID());
  for (auto &output_control : output_control_arrows_) {
    MS_EXCEPTION_IF_NULL(output_control);
    ActorDispatcher::Send(output_control->to_op_id_, &OpRTActor::RunOpControl, from_aid, context);
  }

  // The LoopCountActor exits.
  if (current_count_ == loop_count_) {
    current_count_ = 0;
    return;
  }

  // Send to DataPrepareActor to trigger next step running.
  ActorDispatcher::Send(data_prepare_aid_, &OpRTActor::RunOpControl, from_aid, context);
}
}  // namespace runtime
}  // namespace mindspore
