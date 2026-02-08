/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/runtime/actor/loop_count_actor.h"
#include <set>
#include "backend/ge_backend/runtime/actor/data_prepare_actor.h"
#include "backend/ge_backend/runtime/actor/output_actor.h"
#include "backend/ge_backend/runtime/actor/memory_manager_actor.h"
#include "backend/ge_backend/runtime/actor/recorder_actor.h"
#include "backend/ge_backend/runtime/actor/debug_actor.h"
#include "backend/ge_backend/runtime/actor/profiler_actor.h"
#include "backend/ge_backend/runtime/actor/control_flow/entrance_actor.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
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
  MS_LOG(INFO) << "Loop count actor(" << GetAID().Name() << ") running, loop count: " << loop_count_
               << ", current count: " << current_count_ << ", total running count: " << total_running_count_;

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
    return;
  }

  if (profiler_aid_ != nullptr) {
    MS_LOG(INFO) << "Sync stream in the step end by profiler.";
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kStreamSync, GetAID().Name());
    SendProfilerReq(context);
    return;
  }

  // Sync device stream.
  if ((strategy_ == GraphExecutionStrategy::kPipeline) && is_need_sync_stream_) {
    MS_LOG(INFO) << "Sync stream in the step end.";
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kStreamSync, GetAID().Name());
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    host_context->device_res_manager_->SyncAllStreams(false);
    MS_LOG(INFO) << "Sync stream success.";
  }

  PostRun(context);
}

void LoopCountActor::SendDebugReq(OpContext<KernelTensor> *const context) {
  ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugOnStepEnd, context, &GetAID(), total_running_count_,
                            sink_size_);
  OnDebugFinish(context);
}

void LoopCountActor::SendProfilerReq(OpContext<KernelTensor> *const context) {
  ActorDispatcher::SendSync(*profiler_aid_, &ProfilerActor::ProfilerOnStepEnd, context, &GetAID(),
                            total_running_count_);
  OnDebugFinish(context);
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
  ActorDispatcher::Send(data_prepare_aid_, &OpRTActor::RunOpControl, from_aid, context);
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
