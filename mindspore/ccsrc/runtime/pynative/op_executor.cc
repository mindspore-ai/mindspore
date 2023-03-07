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

namespace mindspore::runtime {
OpExecutor &OpExecutor::GetInstance() {
  static OpExecutor instance;
  return instance;
}

OpExecutor::OpExecutor() = default;

OpExecutor::~OpExecutor() = default;

void OpExecutor::RegisterForwardCallback(const std::function<void()> &callback) { forward_callback_ = callback; }

void OpExecutor::Register(const std::function<void()> &callback) { batch_build_callback_ = callback; }

void OpExecutor::Reset() {
  ClearResources();
  batch_build_callback_ = nullptr;
  async_queue_.Reset();
}

void OpExecutor::ClearResources() {
  MS_LOG(DEBUG) << "Start clear tasks";
  std::unique_lock<std::mutex> lock(build_mutex_);
  // Set the build task failed, and no need to run op_run_tasks.
  for (auto &build_task : op_build_tasks_) {
    build_task->SetBuildReady(false);
  }
  op_build_tasks_.clear();
  MS_LOG(DEBUG) << "End clear tasks";
}

void OpExecutor::WaitForBuild() {
  if (!executing_) {
    ExecuteGuard guard;
    if (batch_build_callback_ != nullptr) {
      batch_build_callback_();
    }
  }
}

void OpExecutor::WaitForRun() {
  MS_LOG(DEBUG) << "Start";
  async_queue_.Wait();
  MS_LOG(DEBUG) << "All task finish";
}

void OpExecutor::Wait() {
  GilReleaseWithCheck gil_release;
  WaitForBuild();
  WaitForRun();
}

void OpExecutor::WaitAll() {
  GilReleaseWithCheck gil_release;
  forward_callback_();
  WaitForBuild();
  WaitForRun();
}

void OpExecutor::PushOpBuildTask(const std::shared_ptr<pynative::BackendOpBuildTask> &op_build_task) {
  std::unique_lock<std::mutex> lock(build_mutex_);
  op_build_tasks_.push_back(op_build_task);
}

void OpExecutor::PushOpRunTask(const std::shared_ptr<pynative::BackendOpRunTask> &op_run_task) {
  async_queue_.Push(op_run_task);
  (void)actor_in_queue_.insert(op_run_task->context()->graph_id());
}

void OpExecutor::ClearOpBuildTasks() {
  std::unique_lock<std::mutex> lock(build_mutex_);
  for (auto &task : op_build_tasks_) {
    task->SetBuildReady(true);
  }
  op_build_tasks_.clear();
  MS_LOG(DEBUG) << "Clear build task";
}

std::vector<std::shared_ptr<pynative::BackendOpBuildTask>> OpExecutor::PopOpBuildTasks() {
  std::unique_lock<std::mutex> lock(build_mutex_);
  auto build_tasks = op_build_tasks_;
  op_build_tasks_.clear();
  return build_tasks;
}

bool OpExecutor::BuildQueueEmpty() {
  std::unique_lock<std::mutex> lock(build_mutex_);
  return op_build_tasks_.empty();
}

bool OpExecutor::RunQueueEmpty() { return async_queue_.Empty(); }

bool OpExecutor::BuildQueueFull() {
  std::unique_lock<std::mutex> lock(build_mutex_);
  return op_build_tasks_.size() > kMaxQueueSize;
}

bool OpExecutor::ActorInQueue(GraphId graph_id) {
  auto iter = actor_in_queue_.find(graph_id);
  return iter != actor_in_queue_.end();
}

void OpExecutor::WorkerJoin() {
  Wait();
  async_queue_.WorkerJoin();
}
}  // namespace mindspore::runtime
