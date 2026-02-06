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

#include "backend/ge_backend/runtime/actor/profiler_actor.h"
#include <vector>
#include <memory>
#include <string>
#include "async/async.h"
#include "utils/log_adapter.h"
#include "utils/file_utils.h"
#include "tools/profiler/profiling.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
/*
 * Feature group: ascend step start timestamp
 * Target device group: Ascend.
 * Description: Add step start timestamp when profiler is started.
 */
void ProfilerActor::AscendStepStart(const std::vector<KernelGraphPtr> &graphs) {
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profiler == nullptr || !profiler->IsInitialized() || graphs.empty()) {
    return;
  }
  if (profiler->GetEnableFlag() && !graphs[0]->IsDatasetGraph()) {
    profile_started_ = false;
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    for (size_t i = 0; i < graphs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(graphs[i]);
      if (!profile_started_) {
        host_context->device_res_manager_->BindDeviceToCurrentThread(false);
        MS_LOG(INFO) << "Dot step start timestamp.";
        profiler->StepStart(current_step++, host_context->device_res_manager_->GetStream());
        profile_started_ = true;
      }
    }
  }
}

/*
 * Feature group: ascend step end timestamp
 * Target device group: Ascend.
 * Description: Add step end timestamp when profiler is end.
 */
void ProfilerActor::AscendStepEnd() {
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profile_started_ && profiler != nullptr && profiler->GetEnableFlag()) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    host_context->device_res_manager_->BindDeviceToCurrentThread(false);
    host_context->device_res_manager_->SyncAllStreams(false);
    MS_LOG(INFO) << "Dot step end timestamp.";
    profiler->StepStop();
    profile_started_ = false;
  }
}

/*
 * Feature group: Dump, Online Profilerger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 */
void ProfilerActor::ProfilerOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                                        const std::vector<AnfNodePtr> &origin_parameters_order,
                                        OpContext<KernelTensor> *const op_context, const AID *) {
  MS_LOG(INFO) << "Profiler on step begin.";
  AscendStepStart(graphs);
  MS_LOG(INFO) << "Profiler_actor ProfilerOnStepBegin.";
  return;
}

/*
 * Feature group: Dump, Online Profilerger.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: MindRT.
 */
void ProfilerActor::ProfilerOnStepEnd(OpContext<KernelTensor> *const op_context, const AID *,
                                      int total_running_count_) {
  MS_LOG(INFO) << "Profiler on step begin.";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  step_count = total_running_count_;
  if (backend == "ge") {
    AscendStepEnd();
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
    MS_LOG(INFO) << "Profiler_actor ProfilerOnStepEnd.";
    return;
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
