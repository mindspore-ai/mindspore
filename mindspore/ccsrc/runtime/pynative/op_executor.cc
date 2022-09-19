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
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "include/common/utils/signal_util.h"
#endif

namespace mindspore::runtime {
OpExecutor &OpExecutor::GetInstance() {
  static OpExecutor instance;
  return instance;
}

OpExecutor::OpExecutor() { worker_ = std::make_shared<std::thread>(&OpExecutor::WorkerLoop, this); }

OpExecutor::~OpExecutor() { WorkerJoin(); }

void OpExecutor::Register(const std::function<void()> &callback) {
  batch_build_callback_ = callback;
  registered_ = true;
}

void OpExecutor::Reset() {
  ClearResources();
  batch_build_callback_ = nullptr;
  registered_ = false;

  // There is still one task in progress
  try {
    WaitForRun();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Wait failed, error message:" << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "Wait failed";
  }
}

void OpExecutor::ClearResources() {
  MS_LOG(DEBUG) << "Start clear tasks";
  std::lock_guard<std::mutex> lock(task_mutex_);
  ClearRunOpTasks();

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
  std::unique_lock<std::mutex> lock(task_mutex_);
  task_cond_var_.wait(lock, [this]() { return op_run_tasks_.empty(); });
  MsException::Instance().CheckException();
  MS_LOG(DEBUG) << "All task finish";
}

void OpExecutor::Wait() {
  WaitForBuild();
  WaitForRun();
}

void OpExecutor::PushOpBuildTask(const std::shared_ptr<OpBuildTask> &op_build_task) {
  std::lock_guard<std::mutex> lock(task_mutex_);
  op_build_tasks_.push_back(op_build_task);
}

void OpExecutor::PushOpRunTask(const std::shared_ptr<OpTask> &op_run_task) {
  std::lock_guard<std::mutex> lock(task_mutex_);
  op_run_tasks_.push(op_run_task);
  actor_in_queue_.insert(op_run_task->context()->graph_id());
  task_cond_var_.notify_all();
}

void OpExecutor::ClearOpBuildTasks() {
  std::lock_guard<std::mutex> lock(task_mutex_);
  for (auto &task : op_build_tasks_) {
    task->SetBuildReady(true);
  }
  op_build_tasks_.clear();
  MS_LOG(DEBUG) << "Clear build task";
}

bool OpExecutor::BuildQueueEmpty() {
  std::lock_guard<std::mutex> lock(task_mutex_);
  return op_build_tasks_.empty();
}

bool OpExecutor::RunQueueEmpty() {
  std::lock_guard<std::mutex> lock(task_mutex_);
  return op_run_tasks_.empty();
}

bool OpExecutor::BuildQueueFull() {
  std::lock_guard<std::mutex> lock(task_mutex_);
  return op_build_tasks_.size() > kMaxQueueSize;
}

bool OpExecutor::ActorInQueue(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(task_mutex_);
  auto iter = actor_in_queue_.find(graph_id);
  return iter != actor_in_queue_.end();
}

void OpExecutor::ClearRunOpTasks() {
  actor_in_queue_.clear();
  std::queue<std::shared_ptr<OpTask>> empty;
  // No need to worry about ExitOpTask.
  // ClearRunOpTasks is executed before ~OpExecutor
  std::swap(op_run_tasks_, empty);
}

void OpExecutor::WorkerLoop() {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  SignalGuard sig([](int, siginfo_t *, void *) {
    int this_pid = getpid();
    MS_LOG(WARNING) << "Process " << this_pid << " receive KeyboardInterrupt signal.";
    (void)kill(this_pid, SIGTERM);
  });
#endif

  while (true) {
    std::shared_ptr<OpTask> task;
    {
      MS_LOG(DEBUG) << "Wait task in queue";
      std::unique_lock<std::mutex> lock(task_mutex_);
      task_cond_var_.wait(lock, [this]() { return !op_run_tasks_.empty(); });
      task = op_run_tasks_.front();
    }

    MS_LOG(DEBUG) << "Get task";
    MS_EXCEPTION_IF_NULL(task);
    if (task->task_type() == kExitTask) {
      MS_LOG(DEBUG) << "Thread exit";
      return;
    }
    try {
      task->Run();
      std::unique_lock<std::mutex> lock(task_mutex_);
      if (!op_run_tasks_.empty()) {
        op_run_tasks_.pop();
        actor_in_queue_.erase(task->context()->graph_id());
      }

      if (op_run_tasks_.empty()) {
        MS_LOG(DEBUG) << "Task queue empty";
        task_cond_var_.notify_all();
      }
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Run lazy task failed, error message:" << e.what();
      {
        std::unique_lock<std::mutex> lock(task_mutex_);
        ClearRunOpTasks();
        MsException::Instance().SetException();
        task_cond_var_.notify_all();
      }
    }
  }
}

void OpExecutor::WorkerJoin() {
  try {
    // Avoid worker thread join itself which will cause deadlock
    if (worker_->joinable() && worker_->get_id() != std::this_thread::get_id()) {
      {
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto task = std::make_shared<ExitOpTask>();
        op_run_tasks_.push(task);
        task_cond_var_.notify_all();
        MS_LOG(DEBUG) << "Push exit task and notify all";
      }
      worker_->join();
      MS_LOG(DEBUG) << "Worker join finish";
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "WorkerJoin failed: " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "WorkerJoin failed";
  }
}
}  // namespace mindspore::runtime
