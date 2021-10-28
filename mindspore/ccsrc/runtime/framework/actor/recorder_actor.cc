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

#include "runtime/framework/actor/recorder_actor.h"
#include <string>
#include <utility>
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/recorder_manager.h"
#include "debug/rdr/running_data_recorder.h"
#endif
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void RecorderActor::RecordInfo(const std::string op_name, const KernelLaunchInfo *launch_info_,
                               const DeviceContext *device_context, OpContext<DeviceTensor> *const op_context) {
  MS_EXCEPTION_IF_NULL(launch_info_);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);

#ifdef ENABLE_DUMP_IR
  if (op_name.empty()) {
    MS_LOG(WARNING) << "GPU kernel's op_name is empty, do not record its memory address in RDR.";
    return;
  }
  std::string name = "mem_address_list";
  if (!RecorderManager::Instance().CheckRdrMemIsRecord()) {
    (void)RDR::RecordMemAddressInfo(SUBMODULE_ID, name);
    RecorderManager::Instance().SetRdrMemIsRecord(true);
  } else {
    (void)RDR::UpdateMemAddress(SUBMODULE_ID, name, op_name, *launch_info_);
  }
#endif
}

void RecorderActor::RecordOnStepEnd(OpContext<DeviceTensor> *const op_context) {
  MS_EXCEPTION_IF_NULL(op_context);
#ifndef ENABLE_SECURITY
  // Record iter_start, fp_start and iter_end op name and timestamp at the step end. (GPU)
  if (profiler::ProfilerManager::GetInstance()->GetProfilingEnableFlag()) {
    profiler::ProfilerManager::GetInstance()->RecordOneStepStartEndInfo();
  }
#endif
}
}  // namespace runtime
}  // namespace mindspore
