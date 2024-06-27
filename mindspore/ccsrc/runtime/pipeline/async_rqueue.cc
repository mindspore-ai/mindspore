/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "runtime/pipeline/async_rqueue.h"

#include <utility>
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "include/common/utils/signal_util.h"
#endif
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"
#include "mindrt/include/fork_utils.h"
#include "include/common/profiler.h"

#include "utils/profile.h"

namespace mindspore {
namespace runtime {
constexpr size_t kThreadNameThreshold = 15;
thread_local kThreadWaitLevel current_level_{kThreadWaitLevel::kLevelUnknown};

AsyncRQueue::~AsyncRQueue() {
  try {
    WorkerJoin();
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "WorkerJoin failed, error msg:" << e.what();
  }
}

void AsyncRQueue::SetThreadName() const {
// Set thread name for gdb debug
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  (void)pthread_setname_np(pthread_self(), name_.substr(0, kThreadNameThreshold).c_str());
#endif
}

void AsyncRQueue::WorkerLoop() {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  // cppcheck-suppress unreadVariable
  SignalGuard sig([](int, siginfo_t *, void *) {
    int this_pid = getpid();
    MS_LOG(WARNING) << "Process " << this_pid << " receive KeyboardInterrupt signal.";
    (void)kill(this_pid, SIGTERM);
  });
#endif

  // Thread init.
  SetThreadName();
  runtime::ProfilerAnalyzer::GetInstance().SetThreadIdToName(std::this_thread::get_id(), name_);
  {
    // cppcheck-suppress unreadVariable
    std::unique_lock<std::mutex> lock(level_mutex_);
    thread_id_to_wait_level_[std::this_thread::get_id()] = wait_level_;
  }

  while (true) {
    std::shared_ptr<AsyncTask> task = tasks_queue_.Head();

    MS_LOG(DEBUG) << "Get task";
    MS_EXCEPTION_IF_NULL(task);
    if (task->task_type() == kExitTask) {
      tasks_queue_.Dequeue();
      MS_LOG(DEBUG) << "Thread exit";
      return;
    }

    try {
      task->Run();
      tasks_queue_.Dequeue();
    } catch (const std::exception &e) {
      MS_LOG(INFO) << "Run task failed, error msg:" << e.what();
      {
        MsException::Instance().SetException();
        // MsException is unreliable because it gets modified everywhere.
        auto e_ptr = std::current_exception();
        while (!tasks_queue_.IsEmpty()) {
          auto &t = tasks_queue_.Head();
          if (t->task_type() == kExitTask) {
            break;
          }
          t->SetException(e_ptr);
          tasks_queue_.Dequeue();
        }
      }
    }
  }
}

void AsyncRQueue::Push(const AsyncTaskPtr &task) {
  if (worker_ == nullptr) {
    worker_ = std::make_unique<std::thread>(&AsyncRQueue::WorkerLoop, this);
  }

  if (current_level_ == kThreadWaitLevel::kLevelUnknown) {
    // cppcheck-suppress unreadVariable
    std::unique_lock<std::mutex> lock(level_mutex_);
    current_level_ = thread_id_to_wait_level_[std::this_thread::get_id()];
  }

  if (current_level_ >= wait_level_) {
    MS_LOG(EXCEPTION) << "Cannot push task from thread " << current_level_ << " to queue " << wait_level_;
  }
  tasks_queue_.Enqueue(task);
}

bool AsyncRQueue::CanPush() const {
  if (current_level_ == kThreadWaitLevel::kLevelUnknown) {
    // cppcheck-suppress unreadVariable
    std::unique_lock<std::mutex> lock(level_mutex_);
    current_level_ = thread_id_to_wait_level_[std::this_thread::get_id()];
  }
  return current_level_ < wait_level_;
}

void AsyncRQueue::Wait() {
  if (worker_ == nullptr) {
    return;
  }
  if (current_level_ == kThreadWaitLevel::kLevelUnknown) {
    // cppcheck-suppress unreadVariable
    std::unique_lock<std::mutex> lock(level_mutex_);
    current_level_ = thread_id_to_wait_level_[std::this_thread::get_id()];
  }

  if (current_level_ >= wait_level_) {
    MS_LOG(DEBUG) << "No need to wait, current level " << current_level_ << " AsyncQueue name " << name_;
    // Only need to wait the low level thread.
    return;
  }

  MS_LOG(DEBUG) << "Start to wait thread " << name_;
  while (!tasks_queue_.IsEmpty()) {
  }
  MsException::Instance().CheckException();
  MS_LOG(DEBUG) << "End to wait thread " << name_;
}

bool AsyncRQueue::Empty() { return tasks_queue_.IsEmpty(); }

void AsyncRQueue::Clear() {
  {
    if (tasks_queue_.IsEmpty()) {
      return;
    }

    ClearTaskWithException();

    // Avoid to push task after WorkerJoin.
    if (worker_ != nullptr && worker_->joinable()) {
      auto task = std::make_shared<WaitTask>();
      tasks_queue_.Enqueue(task);
    }
  }
  // There is still one task in progress
  Wait();
}

void AsyncRQueue::Reset() {
  {
    if (tasks_queue_.IsEmpty()) {
      return;
    }

    ClearTaskWithException();
    MS_LOG(DEBUG) << "Reset AsyncQueue";
  }
}

void AsyncRQueue::ClearTaskWithException() {
  while (!tasks_queue_.IsEmpty()) {
    auto &t = tasks_queue_.Head();
    t->SetException(std::make_exception_ptr(std::runtime_error("Clean up tasks that are not yet running")));
    tasks_queue_.Dequeue();
  }
}

void AsyncRQueue::WorkerJoin() {
  try {
    if (worker_ == nullptr) {
      return;
    }
    // Avoid worker thread join itself which will cause deadlock
    if (worker_->joinable() && worker_->get_id() != std::this_thread::get_id()) {
      {
        auto task = std::make_shared<ExitTask>();
        tasks_queue_.Enqueue(task);
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

void AsyncRQueue::ChildAfterFork() {
  MS_LOG(DEBUG) << "AsyncQueue reinitialize after fork";
  if (worker_ != nullptr) {
    MS_LOG(DEBUG) << "Release and recreate worker_.";
    (void)worker_.release();
    worker_ = std::make_unique<std::thread>(&AsyncRQueue::WorkerLoop, this);
  }
  MS_LOG(DEBUG) << "AsyncQueue reinitialize after fork done.";
}
}  // namespace runtime
}  // namespace mindspore
