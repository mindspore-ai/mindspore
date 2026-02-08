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

#include "backend/ms_backend/runtime/actors/base/recorder_actor.h"
#include <string>
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void RecorderActor::RecordInfo(const std::string op_name, const KernelLaunchAddr *launch_info,
                               const DeviceContext *device_context, OpContext<KernelTensor> *const op_context) {}

void RecorderActor::RecordOnStepEnd(OpContext<KernelTensor> *const op_context) {
  MS_EXCEPTION_IF_NULL(op_context);
  // Record iter_start, fp_start and iter_end op name and timestamp at the step end. (GPU)
  if (profiler::ProfilerManager::GetInstance()->GetProfilingEnableFlag()) {
    profiler::ProfilerManager::GetInstance()->RecordOneStepStartEndInfo();
  }
}
}  // namespace runtime
}  // namespace mindspore
