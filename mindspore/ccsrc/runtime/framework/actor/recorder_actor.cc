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
#include "debug/rdr/recorder_manager.h"
#include "debug/rdr/mem_address_recorder.h"
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
  // record GPU memory address info
  if (!RecorderManager::Instance().RdrEnable()) {
    return;
  }
  std::string name = "mem_address_list";
  if (!RecorderManager::Instance().CheckRdrGPUMemIsRecord()) {
    std::string submodule_name = "KERNEL";
    auto mem_info_recorder = std::make_shared<GPUMemAddressRecorder>(submodule_name, name);
    if (mem_info_recorder == nullptr) {
      MS_LOG(ERROR) << "Make GPUMemAddressRecorder shared pointer failed.";
      return;
    }
    mem_info_recorder->SaveMemInfo(op_name, launch_info_);
    bool result = RecorderManager::Instance().RecordObject(std::move(mem_info_recorder));
    if (result) {
      RecorderManager::Instance().SetRdrGPUMemIsRecord(true);
    }
  } else {
    std::string submodule_name = "KERNEL";
    auto recorder = RecorderManager::Instance().GetRecorder(submodule_name, name);
    if (recorder != nullptr) {
      auto mem_recorder = std::dynamic_pointer_cast<GPUMemAddressRecorder>(recorder);
      mem_recorder->SaveMemInfo(op_name, launch_info_);
    }
  }
#endif
}

void RecorderActor::RecordOnStepEnd(OpContext<DeviceTensor> *const op_context) {
  MS_EXCEPTION_IF_NULL(op_context);
  // todo clear
  // Record iter_start, fp_start and iter_end op name and timestamp at the step end. (GPU)
  if (profiler::ProfilerManager::GetInstance()->GetEnableRecorderActorFlag()) {
    profiler::ProfilerManager::GetInstance()->RecordOneStepStartEndInfo();
  }
}
}  // namespace runtime
}  // namespace mindspore
