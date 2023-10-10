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

#include <string>
#include <memory>
#include "include/common/utils/tensor_future.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace pynative {
void FrontendTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     runtime::ProfilerRecorder::kNoName, false);
  run_func_(op_run_info_);
  op_run_info_ = nullptr;
}

void FrontendTask::SetException(const std::exception_ptr &e) {
  if (op_run_info_ == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(op_run_info_->stub_output);
  op_run_info_->stub_output->SetException(e);
}

void SliceOpFrontendTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     "Slice Op", false);
  run_func_(input_values_, slice_op_infos_, requires_grad_, stub_output_);
  input_values_.clear();
  slice_op_infos_.clear();
  stub_output_ = nullptr;
}

void SliceOpFrontendTask::SetException(const std::exception_ptr &e) {
  if (stub_output_ == nullptr) {
    return;
  }
  stub_output_->SetException(e);
}

void BackendTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBackendTask,
                                     runtime::ProfilerRecorder::kNoName, false);
  run_func_(op_run_info_, backend_op_run_info_);
  op_run_info_ = nullptr;
}

void BackendTask::SetException(const std::exception_ptr &e) {
  if (backend_op_run_info_ != nullptr) {
    for (auto &promise : backend_op_run_info_->device_sync_promises) {
      MS_EXCEPTION_IF_NULL(promise);
      promise->SetValue(std::make_shared<pynative::DeviceAddressFutureData>(nullptr, e));
    }
  }
}

void AllocViewMemBackendTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBackendTask,
                                     std::string("AllocView"), false);
  run_func_(op_run_info_, input_tensor_, input_idx_, need_wait_);
  op_run_info_ = nullptr;
  input_tensor_ = nullptr;
}

void AllocViewMemBackendTask::SetException(const std::exception_ptr &e) {
  if (op_run_info_ == nullptr || op_run_info_->stub_output == nullptr) {
    return;
  }
  op_run_info_->stub_output->SetException(e);
}

void ContiguousBackendTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBackendTask,
                                     std::string("Contiguous"), false);
  run_func_(tensor_);
  tensor_ = nullptr;
}

void ViewKernelBackendTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBackendTask,
                                     std::string("ViewKernel"), false);
  run_func_(op_run_info_, task_type_);
  op_run_info_ = nullptr;
}
}  // namespace pynative
}  // namespace mindspore
