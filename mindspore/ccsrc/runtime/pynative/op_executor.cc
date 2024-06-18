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

#include "runtime/pynative/op_executor.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore::runtime {
OpExecutor &OpExecutor::GetInstance() {
  static OpExecutor instance;
  return instance;
}

OpExecutor::OpExecutor() = default;

OpExecutor::~OpExecutor() = default;

void OpExecutor::Reset() {
  runtime::Pipeline::Get().backend_stage()->Reset();
  runtime::Pipeline::Get().launch_stage()->Reset();
}

void OpExecutor::PushOpRunTask(const std::shared_ptr<DeviceOpRunTask> &op_run_task) {
  MS_EXCEPTION_IF_NULL(op_run_task);
  MS_EXCEPTION_IF_NULL(op_run_task->context());
  runtime::Pipeline::Get().backend_stage()->Push(op_run_task);
}

void OpExecutor::PushOpRunTask(const std::shared_ptr<PyBoostDeviceTask> &op_run_task) {
  MS_EXCEPTION_IF_NULL(op_run_task);
  runtime::Pipeline::Get().backend_stage()->Push(op_run_task);
}

void OpExecutor::PushSimpleOpRunTask(const std::shared_ptr<AsyncTask> &op_run_task) {
  runtime::Pipeline::Get().backend_stage()->Push(op_run_task);
}

bool OpExecutor::RunQueueEmpty() { return runtime::Pipeline::Get().backend_stage()->Empty(); }

void OpExecutor::WorkerJoin() {
  GilReleaseWithCheck release_gil;
  runtime::Pipeline::Get().backend_stage()->WorkerJoin();
  runtime::Pipeline::Get().launch_stage()->WorkerJoin();
}

void OpExecutor::DispatchLaunchTask(const std::function<void()> &func) {
  if (NeedSync()) {
    runtime::Pipeline::Get().WaitForward();
    func();
  } else {
    auto task = std::make_shared<runtime::DeviceLaunchTask>([=]() { func(); });
    runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(task->task_id());
    runtime::Pipeline::Get().launch_stage()->Push(task);
  }
}

bool OpExecutor::NeedSync() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) ||
         (context->get_param<int>(MS_CTX_EXECUTION_MODE) == mindspore::kGraphMode && !async_for_graph_);
}

void OpExecutor::ChildAfterFork() {
  MS_LOG(DEBUG) << "OpExecutor reinitialize after fork";
  MS_LOG(DEBUG) << "Reinitialize async_queue_.";
  runtime::Pipeline::Get().backend_stage()->ChildAfterFork();
  runtime::Pipeline::Get().launch_stage()->ChildAfterFork();
  // Refresh the lazy callback in Tensor.
  tensor::Tensor::RegisterLazyCallback([]() { runtime::Pipeline::Get().WaitAll(); });
  MS_LOG(DEBUG) << "OpExecutor reinitialize after fork done.";
}
}  // namespace mindspore::runtime
