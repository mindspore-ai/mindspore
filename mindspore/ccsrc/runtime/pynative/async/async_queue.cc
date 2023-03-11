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

#include "runtime/pynative/async/async_queue.h"

#include <utility>
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "include/common/utils/signal_util.h"
#endif
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace pynative {
AsyncQueue::~AsyncQueue() { WorkerJoin(); }

void AsyncQueue::WorkerLoop() {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  SignalGuard sig([](int, siginfo_t *, void *) {
    int this_pid = getpid();
    MS_LOG(WARNING) << "Process " << this_pid << " receive KeyboardInterrupt signal.";
    (void)kill(this_pid, SIGTERM);
  });
#endif

  while (true) {
    std::shared_ptr<AsyncTask> task;
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      task_cond_var_.wait(lock, [this]() { return !tasks_.empty(); });
      task = tasks_.front();
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
      if (!tasks_.empty()) {
        tasks_.pop();
      }

      if (tasks_.empty()) {
        MS_LOG(DEBUG) << "Task queue empty";
        task_cond_var_.notify_all();
      }
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Run task failed, error msg:" << e.what();
      {
        std::unique_lock<std::mutex> lock(task_mutex_);

        MsException::Instance().SetException();
        // MsException is unreliable because it gets modified everywhere.
        auto e_ptr = std::current_exception();
        task->SetException(e_ptr);
        while (!tasks_.empty()) {
          auto &t = tasks_.front();
          if (t->task_type() == kExitTask) {
            break;
          }
          t->SetException(e_ptr);
          tasks_.pop();
        }

        task_cond_var_.notify_all();
      }
    }
  }
}

void AsyncQueue::Push(const std::shared_ptr<AsyncTask> &task) {
  if (worker_ == nullptr) {
    worker_ = std::make_shared<std::thread>(&AsyncQueue::WorkerLoop, this);
  }
  std::lock_guard<std::mutex> lock(task_mutex_);
  tasks_.push(task);
  task_cond_var_.notify_all();
}

void AsyncQueue::Wait() {
  if (worker_ == nullptr) {
    return;
  }
  // Avoid deadlock.
  if (worker_->get_id() == std::this_thread::get_id()) {
    return;
  }
  std::unique_lock<std::mutex> lock(task_mutex_);
  task_cond_var_.wait(lock, [this]() { return tasks_.empty(); });
  MsException::Instance().CheckException();
}

bool AsyncQueue::Empty() {
  std::lock_guard<std::mutex> lock(task_mutex_);
  return tasks_.empty();
}

void AsyncQueue::Clear() {
  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    if (tasks_.empty()) {
      return;
    }
    std::queue<std::shared_ptr<AsyncTask>> empty;
    std::swap(tasks_, empty);

    // Avoid to push task after WorkerJoin.
    if (worker_ != nullptr && worker_->joinable()) {
      auto task = std::make_shared<WaitTask>();
      tasks_.push(task);
    }

    task_cond_var_.notify_all();
  }
  // There is still one task in progress
  Wait();
}

void AsyncQueue::Reset() {
  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    if (tasks_.empty()) {
      return;
    }
    std::queue<std::shared_ptr<AsyncTask>> empty;
    std::swap(tasks_, empty);
    MS_LOG(DEBUG) << "Reset AsyncQueue";
  }
}

void AsyncQueue::WorkerJoin() {
  try {
    if (worker_ == nullptr) {
      return;
    }
    // Avoid worker thread join itself which will cause deadlock
    if (worker_->joinable() && worker_->get_id() != std::this_thread::get_id()) {
      {
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto task = std::make_shared<ExitTask>();
        tasks_.push(task);
        task_cond_var_.notify_all();
        MS_LOG(DEBUG) << "Push exit task and notify all";
      }
      worker_->join();
      MS_LOG(DEBUG) << "Worker join finish";
      MsException::Instance().CheckException();
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "WorkerJoin failed: " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "WorkerJoin failed";
  }
}
}  // namespace pynative
}  // namespace mindspore
