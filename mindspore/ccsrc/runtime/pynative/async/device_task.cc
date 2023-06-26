/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <string>
#include "runtime/pynative/async/device_task.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace pynative {
DeviceOpBuildTask::~DeviceOpBuildTask() {
  if (!has_set_value_) {
    promise_.set_value(false);
  }
}

void DeviceOpBuildTask::SetBuildReady(bool build_success) {
  promise_.set_value(build_success);
  has_set_value_ = true;
}

void DeviceOpRunTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeDeviceTask,
                                     runtime::ProfilerRecorder::kNoName, false);
  MS_LOG(DEBUG) << "Wait for build";
  auto build_status = future_.get();
  if (!build_status) {
    MS_LOG(WARNING) << "Op build failed, no need to launch.";
    return;
  }
  MS_EXCEPTION_IF_NULL(run_func_);
  run_func_(context_);
}

void AllocViewMemDeviceTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeDeviceTask,
                                     std::string("AllocView"), false);
  run_func_(device_context_, tensor_);
}

void KernelDeviceTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeDeviceTask,
                                     std::string("KernelTask"), false);
  run_func_(task_type_, input_addr_list_, input_storage_list_, output_addr_list_, device_context_);
}
}  // namespace pynative
}  // namespace mindspore
