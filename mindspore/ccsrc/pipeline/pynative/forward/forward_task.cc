/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/forward/forward_task.h"

#include <memory>
#include "include/common/utils/tensor_future.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace pynative {
void FrontendTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     std::string(), false);
  run_func_(op_run_info_);
  op_run_info_ = nullptr;
}

void FrontendTask::SetException(const std::exception_ptr &e) {
  if (op_run_info_ == nullptr) {
    return;
  }
  op_run_info_->stub_output->SetException(e);
}

void BackendTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBackendTask,
                                     std::string(), false);
  run_func_(op_run_info_, backend_op_run_info_);
  op_run_info_ = nullptr;
}

void BackendTask::SetException(const std::exception_ptr &e) {
  if (backend_op_run_info_ != nullptr) {
    for (auto &promise : backend_op_run_info_->device_sync_promises) {
      promise->SetValue(std::make_shared<pynative::DeviceAddressFutureData>(nullptr, e));
    }
  }
}
}  // namespace pynative
}  // namespace mindspore
