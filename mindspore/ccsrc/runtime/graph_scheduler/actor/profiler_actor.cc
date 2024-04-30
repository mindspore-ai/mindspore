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

#include "runtime/graph_scheduler/actor/profiler_actor.h"
#include <vector>
#include <memory>
#include <string>
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"
#include "utils/file_utils.h"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace runtime {
/*
 * Feature group: ascend step start timestamp
 * Target device group: Ascend.
 * Description: Add step start timestamp when profiler is started.
 */
void ProfilerActor::AscendStepStart(const std::vector<KernelGraphPtr> &graphs,
                                    std::vector<DeviceContext *> device_contexts) {
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profiler == nullptr || !profiler->IsInitialized() || graphs.empty()) {
    return;
  }
  if (profiler->GetEnableFlag() && !graphs[0]->IsDatasetGraph()) {
    profile_started_ = false;
    for (size_t i = 0; i < graphs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(graphs[i]);
      MS_EXCEPTION_IF_NULL(device_contexts[i]);
      if (device_contexts[i]->GetDeviceType() == device::DeviceType::kAscend && !profile_started_) {
        device_ctx_ = device_contexts[i];
        device_ctx_->device_res_manager_->BindDeviceToCurrentThread(false);
        MS_LOG(INFO) << "Dot step start timestamp.";
        profiler->StepStart(current_step++, device_contexts[i]->device_res_manager_->GetStream());
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
    MS_EXCEPTION_IF_NULL(device_ctx_);
    device_ctx_->device_res_manager_->BindDeviceToCurrentThread(false);
    device_ctx_->device_res_manager_->SyncAllStreams();
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
                                        std::vector<DeviceContext *> device_contexts,
                                        OpContext<DeviceTensor> *const op_context, const AID *) {
  MS_LOG(INFO) << "Profiler on step begin.";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  device_ctx_ = device_contexts[0];
  if (backend == "ge") {
    AscendStepStart(graphs, device_contexts);
    MS_LOG(INFO) << "Profiler_actor ProfilerOnStepBegin.";
    return;
  }
}

/*
 * Feature group: Dump, Online Profilerger.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: MindRT.
 */
void ProfilerActor::ProfilerOnStepEnd(OpContext<DeviceTensor> *const op_context, const AID *,
                                      int total_running_count_) {
  MS_LOG(INFO) << "Profiler on step begin.";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  step_count = total_running_count_;
  if (backend == "ge") {
    AscendStepEnd();
    device_ctx_->device_res_manager_->SyncAllStreams();
    MS_LOG(INFO) << "Profiler_actor ProfilerOnStepEnd.";
    return;
  }
}
}  // namespace runtime
}  // namespace mindspore
