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
#include <string>
#include <vector>
#include "include/common/profiler.h"

namespace mindspore {
namespace pijit {

void AsyncTaskMultiWorker::Depend(std::shared_ptr<AsyncTaskMultiWorker> task) {
  depends_.push_back(task);
  task->notifies_.push_back(shared_from_this());
  if (task->Done()) {
    comp_count_++;
  }
}

void AsyncTaskMultiWorker::DependOn(std::vector<std::shared_ptr<AsyncTaskMultiWorker>> *tasks) {
  if (tasks != nullptr) {
    tasks->clear();
    tasks->insert(tasks->begin(), depends_.begin(), depends_.end());
  }
}

void AsyncTaskMultiWorker::Notify() {
  for (auto task : notifies_) {
    task->comp_count_++;
  }
}

void AsyncTaskMultiWorker::NotifyTo(std::vector<std::shared_ptr<AsyncTaskMultiWorker>> *tasks) {
  if (tasks != nullptr) {
    tasks->clear();
    tasks->insert(tasks->begin(), notifies_.begin(), notifies_.end());
  }
}

bool AsyncTaskMultiWorker::Available() { return comp_count_ == depends_.size(); }

void AsyncTaskMultiWorker::Reset() {
  comp_count_ = 0;
  done_ = false;
}

void AsyncTaskMultiWorker::RunWrapper() {
  Run();
  done_ = true;
  Notify();
}

AsyncQueueMultiWorker::AsyncQueueMultiWorker(std::string name, runtime::kThreadWaitLevel wait_level,
                                             size_t worker_count)
    : name_(name), wait_level_(wait_level), worker_cnt_(worker_count), ready_cnt_(0), terminate_(false) {}

AsyncQueueMultiWorker::~AsyncQueueMultiWorker() { WorkerJoin(); }

void AsyncQueueMultiWorker::Push(const AsyncTaskPtr &task) {
  while (workers_.size() < worker_cnt_) {
    workers_.emplace_back(std::make_unique<std::thread>(&AsyncQueueMultiWorker::WorkerLoop, this));
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (task->Available()) {
    tasks_queue_.push_back(task);
  } else {
    wait_queue_.push_back(task);
  }
  lock.unlock();
  task_cv_.notify_one();
}

void AsyncQueueMultiWorker::Wait() {
  if (workers_.size() == 0) {
    return;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  ready_cv_.wait(lock, [this] { return tasks_queue_.size() == 0 && ready_cnt_ == worker_cnt_; });
}

bool AsyncQueueMultiWorker::Empty() { return tasks_queue_.size() == 0; }

void AsyncQueueMultiWorker::Clear() {
  std::unique_lock<std::mutex> lock(mutex_);
  tasks_queue_.clear();
}

void AsyncQueueMultiWorker::WorkerJoin() {
  std::unique_lock<std::mutex> lock(mutex_);
  tasks_queue_.clear();
  terminate_ = true;
  lock.unlock();
  task_cv_.notify_all();
  for (size_t w = 0; w < workers_.size(); ++w) {
    if (workers_[w]->joinable()) {
      workers_[w]->join();
    }
  }
}

bool AsyncQueueMultiWorker::Available() { return tasks_queue_.size() > 0; }

AsyncTaskPtr AsyncQueueMultiWorker::PopAvailable() {
  auto iter = std::find_if(tasks_queue_.begin(), tasks_queue_.end(), [](auto &task) { return task->Available(); });
  if (iter != tasks_queue_.end()) {
    AsyncTaskPtr ret = *iter;
    tasks_queue_.erase(iter);
    return ret;
  } else {
    return nullptr;
  }
}

AsyncTaskPtr AsyncQueueMultiWorker::Pop() {
  std::unique_lock<std::mutex> lock(mutex_);
  auto task = PopAvailable();
  if (task != nullptr) {
    return task;
  } else {
    ready_cnt_++;
    if (ready_cnt_ == worker_cnt_) {
      ready_cv_.notify_one();
    }
    task_cv_.wait(lock, [this] { return Available() || terminate_; });
    AsyncTaskPtr ret = PopAvailable();
    if (ret != nullptr) {
      ready_cnt_--;
    } else {
      if (ready_cnt_ == worker_cnt_) {
        lock.unlock();
        ready_cv_.notify_one();
      }
    }
    return ret;
  }
}

void AsyncQueueMultiWorker::WorkerLoop() {
  while (!terminate_) {
    auto task = Pop();
    if (task != nullptr) {
      task->RunWrapper();
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (tasks_queue_.size() != 0) {
      return;
    }
    for (auto iter = wait_queue_.begin(); iter != wait_queue_.end();) {
      if (!(*iter)->Available()) {
        iter++;
      } else {
        tasks_queue_.push_back((*iter));
        iter = wait_queue_.erase(iter);
      }
    }
  }
}

void RecordTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBpropTask,
                                     runtime::ProfilerRecorder::kNoName, false);
  MS_LOG(DEBUG) << "Gradient record task start...";
  run_task_(prim_, out_, inputs_);
  run_task_ = nullptr;
  MS_LOG(DEBUG) << "Gradient record task finished.";
}

void RunGenerateBpropTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBpropTask,
                                     runtime::ProfilerRecorder::kNoName, false);
  MS_LOG(DEBUG) << "Generate bprop graph task start...";
  run_task_();
  run_task_ = nullptr;
  MS_LOG(DEBUG) << "Generate bprop graph task finished.";
}

void RunBpropTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeBpropTask,
                                     runtime::ProfilerRecorder::kNoName, false);
  MS_LOG(DEBUG) << "Run gradient bprop graph task start...";
  run_task_(value_);
  run_task_ = nullptr;
  MS_LOG(DEBUG) << "Run gradient bprop graph task finished.";
}
}  // namespace pijit
}  // namespace mindspore
