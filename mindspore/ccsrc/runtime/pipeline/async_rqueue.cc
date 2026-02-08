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

#include "include/runtime/pipeline/async_rqueue.h"

#include <utility>
#include <map>
#include <memory>
#include <vector>
#include <string>
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "include/utils/signal_util.h"
#endif
#include "include/runtime/utils/runtime_conf/thread_bind_core.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"
#include "tools/profiler/profiler.h"
#include "runtime/pipeline/task/task.h"

namespace mindspore {
namespace runtime {
constexpr size_t kThreadNameThreshold = 15;
thread_local kThreadWaitLevel current_level_{kThreadWaitLevel::kLevelUnknown};
constexpr auto kMsDevHostBlockingRun = "MS_DEV_HOST_BLOCKING_RUN";

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

void AsyncRQueue::BindCoreForThread() {
  auto &bind_core_manager = ThreadBindCore::GetInstance();
  if (bind_core_manager.is_enable_thread_bind_core_) {
    // Bind core for pynative pipeline thread.
    const auto &core_list = bind_core_manager.get_thread_bind_core_list(kBindCoreModule::kPYNATIVE);
    if (core_list.empty()) {
      MS_LOG(WARNING) << "Skip to bind thread core for 'pynative' as no available core assigned.";
    } else {
      if (const auto it = thread_to_core_idx.find(name_); it != thread_to_core_idx.end()) {
        bind_core_manager.bind_thread_core({core_list[it->second]});
        return;
      }
    }

    // Bind core for runtime batch launch thread.
    auto runtime_conf_instance = RuntimeConf::GetInstance();
    MS_EXCEPTION_IF_NULL(runtime_conf_instance);
    bool enable_batch_launch_kernel = runtime_conf_instance->IsKernelLaunchGroupConfigured();
    if (!enable_batch_launch_kernel) {
      return;
    }

    uint32_t group_launch_thread_num = runtime_conf_instance->group_launch_thread_num();
    std::map<std::string, int> batch_launch_thread_to_core_idx;
    const std::string kBatchLaunch = "batch_launch_";
    for (uint32_t i = 0; i < group_launch_thread_num; i++) {
      batch_launch_thread_to_core_idx.emplace((kBatchLaunch + std::to_string(i)), i);
    }

    auto iter = batch_launch_thread_to_core_idx.find(name_);
    if (iter == batch_launch_thread_to_core_idx.end()) {
      return;
    }

    const auto &batch_launch_core_list = bind_core_manager.get_thread_bind_core_list(kBindCoreModule::kBATCHLAUNCH);
    if (batch_launch_core_list.empty()) {
      MS_LOG(WARNING) << "Skip to bind thread core for 'batch launch' as no available core assigned.";
    } else {
      auto bind_code_index = iter->second;
      std::vector<int> cpu_list = {batch_launch_core_list[bind_code_index]};
      bind_core_manager.bind_thread_core(cpu_list);
      MS_LOG(INFO) << "The thread: " << name_ << " bind core : " << cpu_list;
    }
  }
}

void AsyncRQueue::WorkerLoop() {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  // cppcheck-suppress unreadVariable
  SignalGuard sig([](int, siginfo_t *, void *) {
    int this_pid = getpid();
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

  BindCoreForThread();

  while (true) {
    std::shared_ptr<AsyncTask> task = tasks_queue_->Head();

    MS_LOG(DEBUG) << "Get task";
    MS_EXCEPTION_IF_NULL(task);
    if (task->task_type() == kExitTask) {
      tasks_queue_->Dequeue();
      MS_LOG(DEBUG) << "Thread exit";
      return;
    }

    try {
      task->Run();
      tasks_queue_->Dequeue();
    } catch (const std::exception &e) {
      MS_LOG(INFO) << "Run task failed, error msg:" << e.what();
      {
        MsException::Instance().SetException();
        // MsException is unreliable because it gets modified everywhere.
        auto e_ptr = std::current_exception();
        while (!tasks_queue_->IsEmpty()) {
          auto &t = tasks_queue_->Head();
          if (t->task_type() == kExitTask) {
            break;
          }
          t->SetException(e_ptr);
          tasks_queue_->Dequeue();
        }
      }
    }
  }
}

void AsyncRQueue::Push(const AsyncTaskPtr &task) {
  static bool blocking_run = common::GetEnv(kMsDevHostBlockingRun) == "1";
  if (blocking_run || disable_multi_thread_) {
    task->Run();
    return;
  }
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
  tasks_queue_->Enqueue(task);
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

  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWait, name_, false,
                                     false);

  MS_LOG(DEBUG) << "Start to wait thread " << name_;
  while (!tasks_queue_->IsEmpty()) {
  }
  MsException::Instance().CheckException();
  MS_LOG(DEBUG) << "End to wait thread " << name_;
}

bool AsyncRQueue::Empty() { return tasks_queue_->IsEmpty(); }

void AsyncRQueue::Clear() {
  {
    if (tasks_queue_->IsEmpty()) {
      return;
    }

    ClearTaskWithException();

    // Avoid to push task after WorkerJoin.
    if (worker_ != nullptr && worker_->joinable()) {
      auto task = std::make_shared<WaitTask>();
      tasks_queue_->Enqueue(task);
    }
  }
  // There is still one task in progress
  Wait();
}

void AsyncRQueue::Reset() {
  {
    if (tasks_queue_->IsEmpty()) {
      return;
    }

    ClearTaskWithException();
    MS_LOG(DEBUG) << "Reset AsyncQueue";
  }
}

void AsyncRQueue::ClearTaskWithException() {
  while (!tasks_queue_->IsEmpty()) {
    auto &t = tasks_queue_->Head();
    t->SetException(std::make_exception_ptr(std::runtime_error("Clean up tasks that are not yet running")));
    tasks_queue_->Dequeue();
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
        tasks_queue_->Enqueue(task);
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

void AsyncRQueue::ChildAfterForkPre() {
  thread_id_to_wait_level_.clear();
  current_level_ = kThreadWaitLevel::kLevelUnknown;
}

void AsyncRQueue::ChildAfterFork() {
  MS_LOG(DEBUG) << "AsyncQueue " << name_ << " reinitialize after fork";
  if (worker_ != nullptr) {
    MS_LOG(DEBUG) << "Release and recreate worker_.";
    (void)worker_.release();
    worker_ = nullptr;
  }
  tasks_queue_ = std::make_unique<RingQueue<AsyncTaskPtr, kQueueCapacity>>();
  MS_LOG(DEBUG) << "AsyncQueue " << name_ << " reinitialize after fork done.";
}

void AsyncRQueue::ParentBeforeFork() {
  WorkerJoin();
  if (worker_ != nullptr) {
    (void)worker_.release();
    worker_ = nullptr;
  }
  tasks_queue_ = std::make_unique<RingQueue<AsyncTaskPtr, kQueueCapacity>>();
  thread_id_to_wait_level_.clear();
  current_level_ = kThreadWaitLevel::kLevelUnknown;
}

void AsyncRQueue::SetSpin(bool spin) {
  tasks_queue_->set_spin(spin);
  MS_LOG(INFO) << "Thread " << name_ << " is set spin to " << spin;
}
}  // namespace runtime
}  // namespace mindspore
