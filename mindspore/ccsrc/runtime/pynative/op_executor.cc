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

void OpExecutor::RegisterForwardCallback(const std::function<void()> &callback) { forward_callback_ = callback; }

void OpExecutor::Reset() { runtime::Pipeline::Get().backend_stage()->Reset(); }

void OpExecutor::WaitForRun() {
  MS_LOG(DEBUG) << "Start";
  runtime::Pipeline::Get().backend_stage()->Wait();
  MS_LOG(DEBUG) << "All task finish";
}

void OpExecutor::Wait() {
  GilReleaseWithCheck gil_release;
  WaitForRun();
}

void OpExecutor::WaitAll() {
  GilReleaseWithCheck gil_release;
  if (forward_callback_ != nullptr) {
    forward_callback_();
  }
  WaitForRun();
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

bool OpExecutor::ActorInQueue(GraphId graph_id) {
  return runtime::Pipeline::Get().backend_stage()->TaskInQueue(graph_id);
}

void OpExecutor::WorkerJoin() {
  GilReleaseWithCheck release_gil;
  runtime::Pipeline::Get().backend_stage()->WorkerJoin();
}

bool OpExecutor::NeedSync() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) ||
         context->get_param<int>(MS_CTX_EXECUTION_MODE) == mindspore::kGraphMode;
}

void OpExecutor::ChildAfterFork() {
  MS_LOG(DEBUG) << "OpExecutor reinitialize after fork";
  MS_LOG(DEBUG) << "Reinitialize async_queue_.";
  runtime::Pipeline::Get().backend_stage()->ChildAfterFork();
  MS_LOG(DEBUG) << "OpExecutor reinitialize after fork done.";
}
}  // namespace mindspore::runtime
