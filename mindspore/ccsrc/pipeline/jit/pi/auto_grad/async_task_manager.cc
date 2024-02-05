/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/pi/auto_grad/async_task_manager.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace pijit {
void RecordTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBpropTask,
                                     runtime::ProfilerRecorder::kNoName, false);
  MS_LOG(DEBUG) << "Gradient record task start...";
  run_task_(prim_, out_, inputs_);
  run_task_ = nullptr;
  MS_LOG(DEBUG) << "Gradient record task finished.";
}

void RunBpropTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBpropTask,
                                     runtime::ProfilerRecorder::kNoName, false);
  MS_LOG(DEBUG) << "Gradient bprop graph task start...";
  run_task_();
  run_task_ = nullptr;
  MS_LOG(DEBUG) << "Gradient bprop graph task finished.";
}
}  // namespace pijit
}  // namespace mindspore
