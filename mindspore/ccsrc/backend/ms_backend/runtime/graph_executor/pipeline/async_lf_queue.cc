/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/graph_executor/pipeline/async_lf_queue.h"
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "include/utils/signal_util.h"
#endif
#include "utils/ms_exception.h"
#include "tools/profiler/profiler.h"

namespace mindspore {
namespace runtime {
constexpr size_t kThreadNameThreshold = 15;

void AsyncLFQueue::SetThreadName() const {
// Set thread name to monitor thread status or gdb debug.
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  (void)pthread_setname_np(pthread_self(), name_.substr(0, kThreadNameThreshold).c_str());
#endif
}

AsyncLFQueue::AsyncLFQueue(std::string name) : name_(std::move(name)) {
  worker_ = std::make_unique<std::thread>(&AsyncLFQueue::WorkerLoop, this);
}

AsyncLFQueue::~AsyncLFQueue() {
  try {
    WorkerJoin();
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "WorkerJoin failed, error msg:" << e.what();
  }
}

void AsyncLFQueue::WorkerLoop() {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  // cppcheck-suppress unreadVariable
  SignalGuard sig([](int, siginfo_t *, void *) {
    int this_pid = getpid();
    (void)kill(this_pid, SIGTERM);
  });
#endif

  SetThreadName();

  while (alive_) {
    auto *task = tasks_queue_.Front();
    if (task == nullptr) {
      return;
    }

    try {
      (*task)();
      tasks_queue_.Pop();
    } catch (const std::exception &e) {
      MsException::Instance().SetException();
      while (!tasks_queue_.Empty()) {
        tasks_queue_.Pop();
      }
    }
  }
}

void AsyncLFQueue::Init() {
  if (init_) {
    return;
  }

  if (worker_ == nullptr) {
    worker_ = std::make_unique<std::thread>(&AsyncLFQueue::WorkerLoop, this);
  }
  init_ = true;
}

void AsyncLFQueue::BindDevice(const std::set<const device::DeviceContext *> &device_contexts) {
  auto bind_device_task = [&device_contexts]() {
    std::for_each(device_contexts.begin(), device_contexts.end(), [](const device::DeviceContext *item) {
      item->device_res_manager_->BindDeviceToCurrentThread(false);
    });
  };
  Push(std::move(bind_device_task));
  Wait();
}

void AsyncLFQueue::Wait() {
  if (!init_ || worker_ == nullptr) {
    return;
  }
  if (worker_->get_id() == std::this_thread::get_id()) {
    return;
  }

  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kWaitTaskFinish, name_);
  std::atomic<bool> atomic_wait_flag = false;
  auto wait_task = [&atomic_wait_flag]() { atomic_wait_flag = true; };
  Push(std::move(wait_task));

  while (atomic_wait_flag == false) {
  }
}

bool AsyncLFQueue::Empty() const { return tasks_queue_.Empty(); }

void AsyncLFQueue::Pause() {
  if (!init_) {
    return;
  }

  if (tasks_queue_.IsPaused()) {
    // Has beed paused already.
    return;
  }

  Wait();
  tasks_queue_.Pause();
}

void AsyncLFQueue::Continue() {
  if (!init_) {
    return;
  }
  tasks_queue_.Continue();
}

std::thread::id AsyncLFQueue::thread_id() const {
  MS_EXCEPTION_IF_NULL(worker_);
  return worker_->get_id();
}

void AsyncLFQueue::WorkerJoin() {
  if (worker_ == nullptr) {
    return;
  }
  if (init_) {
    while (!Empty()) {
    }
  }

  alive_ = false;
  tasks_queue_.Finalize();

  if (worker_->joinable()) {
    worker_->join();
  }
  worker_ = nullptr;
}
}  // namespace runtime
}  // namespace mindspore
