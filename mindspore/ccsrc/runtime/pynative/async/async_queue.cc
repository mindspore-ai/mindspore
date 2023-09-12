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
#include "mindrt/include/fork_utils.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace pynative {
constexpr int32_t kTaskQueueSize = 8192;
constexpr size_t kMaxSpinCount = 300000;
constexpr size_t kThreadNameThreshold = 15;
thread_local kThreadWaitLevel current_level_{kThreadWaitLevel::kLevelUnknown};

AsyncQueue::AsyncQueue(std::string name, kThreadWaitLevel wait_level)
    : name_(std::move(name)), wait_level_(wait_level) {
  // If the fork occurs, thread resources are not forked to child processes, so
  // we need to reinitialize threads in child processes.
  ForkUtils::GetInstance().RegisterCallbacks(this, static_cast<void (AsyncQueue::*)()>(nullptr),
                                             static_cast<void (AsyncQueue::*)()>(nullptr),
                                             &AsyncQueue::ReinitAfterFork);
}

AsyncQueue::~AsyncQueue() {
  try {
    WorkerJoin();
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "WorkerJoin failed, error msg:" << e.what();
  }
}

void AsyncQueue::SetThreadName() const {
// Set thread name for gdb debug
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  (void)pthread_setname_np(pthread_self(), name_.substr(0, kThreadNameThreshold).c_str());
#endif
}

bool AsyncQueue::TaskInQueue(uint32_t task_id) {
  std::unique_lock<std::mutex> lock(task_mutex_);
  return task_in_queue_.find(task_id) != task_in_queue_.end();
}

void AsyncQueue::WorkerLoop() {
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
    std::shared_ptr<AsyncTask> task;
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      task_cond_var_->wait(lock, [this]() { return !tasks_queque_.empty(); });
      task = tasks_queque_.front();
    }

    MS_LOG(DEBUG) << "Get task";
    MS_EXCEPTION_IF_NULL(task);
    if (task->task_type() == kExitTask) {
      std::unique_lock<std::mutex> lock(task_mutex_);
      tasks_queque_.pop();
      MS_LOG(DEBUG) << "Thread exit";
      return;
    }

    try {
      task->Run();
      std::unique_lock<std::mutex> lock(task_mutex_);
      if (!tasks_queque_.empty()) {
        tasks_queque_.pop();
        task_in_queue_.erase(task->task_id());
      }

      if (tasks_queque_.empty()) {
        MS_LOG(DEBUG) << "Task queue empty";
        task_cond_var_->notify_all();
      }
    } catch (const std::exception &e) {
      MS_LOG(INFO) << "Run task failed, error msg:" << e.what();
      {
        // cppcheck-suppress unreadVariable
        std::unique_lock<std::mutex> lock(task_mutex_);

        MsException::Instance().SetException();
        // MsException is unreliable because it gets modified everywhere.
        auto e_ptr = std::current_exception();
        while (!tasks_queque_.empty()) {
          auto &t = tasks_queque_.front();
          if (t->task_type() == kExitTask) {
            break;
          }
          t->SetException(e_ptr);
          tasks_queque_.pop();
          task_in_queue_.erase(task->task_id());
        }

        task_cond_var_->notify_all();
      }
    }
  }
}

void AsyncQueue::Push(const std::shared_ptr<AsyncTask> &task) {
  if (task_cond_var_ == nullptr) {
    task_cond_var_ = std::make_unique<std::condition_variable>();
  }
  if (worker_ == nullptr) {
    worker_ = std::make_unique<std::thread>(&AsyncQueue::WorkerLoop, this);
  }
  // cppcheck-suppress unreadVariable
  std::lock_guard<std::mutex> lock(task_mutex_);
  tasks_queque_.push(task);
  if (task->task_id() != UINT32_MAX) {
    task_in_queue_.insert(task->task_id());
  }
  task_cond_var_->notify_all();
}

void AsyncQueue::Wait() {
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
  std::unique_lock<std::mutex> lock(task_mutex_);
  task_cond_var_->wait(lock, [this]() { return tasks_queque_.empty(); });
  MsException::Instance().CheckException();
  MS_LOG(DEBUG) << "End to wait thread " << name_;
}

bool AsyncQueue::Empty() {
  // cppcheck-suppress unreadVariable
  std::lock_guard<std::mutex> lock(task_mutex_);
  return tasks_queque_.empty();
}

void AsyncQueue::Clear() {
  {
    // cppcheck-suppress unreadVariable
    std::lock_guard<std::mutex> lock(task_mutex_);
    if (tasks_queque_.empty()) {
      return;
    }

    ClearTaskWithException();

    // Avoid to push task after WorkerJoin.
    if (worker_ != nullptr && worker_->joinable()) {
      auto task = std::make_shared<WaitTask>();
      tasks_queque_.push(task);
    }

    task_cond_var_->notify_all();
  }
  // There is still one task in progress
  Wait();
  ForkUtils::GetInstance().DeregCallbacks(this);
}

void AsyncQueue::Reset() {
  {
    // cppcheck-suppress unreadVariable
    std::lock_guard<std::mutex> lock(task_mutex_);
    if (tasks_queque_.empty()) {
      return;
    }

    ClearTaskWithException();
    MS_LOG(DEBUG) << "Reset AsyncQueue";
  }
}

void AsyncQueue::ClearTaskWithException() {
  while (!tasks_queque_.empty()) {
    auto &t = tasks_queque_.front();
    t->SetException(std::make_exception_ptr(std::runtime_error("Clean up tasks that are not yet running")));
    tasks_queque_.pop();
    task_in_queue_.erase(t->task_id());
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
        // cppcheck-suppress unreadVariable
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto task = std::make_shared<ExitTask>();
        tasks_queque_.push(task);
        task_cond_var_->notify_all();
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

void AsyncQueue::ReinitAfterFork() {
  MS_LOG(INFO) << "fork event detected in child process, worker thread will be recreated.";
  if (task_cond_var_ != nullptr) {
    (void)task_cond_var_.release();
    task_cond_var_ = std::make_unique<std::condition_variable>();
  }
  if (worker_ != nullptr) {
    (void)worker_.release();
    worker_ = std::make_unique<std::thread>(&AsyncQueue::WorkerLoop, this);
  }
}
}  // namespace pynative
}  // namespace mindspore
