/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "include/common/thread_pool.h"
#include <exception>
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace common {
constexpr size_t kYieldThreshold = 1000;

ThreadPool::ThreadPool() : max_thread_num_(std::thread::hardware_concurrency()) {}

void ThreadPool::SyncRunLoop(const std::shared_ptr<ThreadContext> &context) {
  if (context == nullptr) {
    return;
  }
  size_t yield_count = 0;
  while (true) {
    if (exit_run_) {
      return;
    }

    if (!context->task) {
      ++yield_count;
      if (yield_count > kYieldThreshold) {
        yield_count = 0;
        std::unique_lock<std::mutex> lock(context->mutex);
        context->cond_var.wait(lock, [&context, this] { return context->task != nullptr || exit_run_; });
      } else {
        std::this_thread::yield();
        continue;
      }
    }

    if (exit_run_) {
      return;
    }

    try {
      auto &task = *(context->task);
      task();
    } catch (std::exception &e) {
      MsException::Instance().SetException();
    }
    yield_count = 0;
    context->task = nullptr;
  }
}

bool ThreadPool::SyncRun(const std::vector<Task> &tasks) {
  if (tasks.empty()) {
    return true;
  }
  if (tasks.size() == 1) {
    auto ret = tasks[0]();
    return ret == SUCCESS;
  }
  std::unique_lock<std::mutex> lock(pool_mtx_);
  exit_run_ = false;
  size_t task_num = tasks.size();
  size_t thread_num = sync_run_threads_.size();
  if (thread_num < max_thread_num_ && thread_num < task_num) {
    auto new_thread_num = max_thread_num_;
    if (task_num < max_thread_num_) {
      new_thread_num = task_num;
    }
    contexts_.resize(new_thread_num);
    for (size_t i = thread_num; i < new_thread_num; ++i) {
      contexts_[i] = std::make_shared<ThreadContext>();
      sync_run_threads_.emplace_back(std::thread(&ThreadPool::SyncRunLoop, this, contexts_[i]));
    }
  }
  if (contexts_.empty()) {
    return true;
  }
  size_t used_thread_num = contexts_.size();
  if (task_num < used_thread_num) {
    used_thread_num = task_num;
  }
  bool running = true;
  size_t task_index = 0;
  while (running) {
    running = false;
    for (size_t i = 0; i < used_thread_num; ++i) {
      MS_EXCEPTION_IF_NULL(contexts_[i]);
      auto &task_run = contexts_[i]->task;
      if (task_run) {
        running = true;
      } else if (task_index < task_num) {
        std::lock_guard<std::mutex> task_lock(contexts_[i]->mutex);
        contexts_[i]->task = &(tasks[task_index]);
        contexts_[i]->cond_var.notify_one();
        running = true;
        ++task_index;
      }
    }
    if (running) {
      std::this_thread::yield();
    }
  }
  return true;
}

ThreadPool &ThreadPool::GetInstance() {
  static ThreadPool instance{};
  return instance;
}

void ThreadPool::ClearThreadPool() {
  std::lock_guard<std::mutex> sync_run_lock(pool_mtx_);
  if (exit_run_) {
    return;
  }
  exit_run_ = true;
  for (auto &context : contexts_) {
    MS_EXCEPTION_IF_NULL(context);
    context->cond_var.notify_one();
  }
  for (auto &it : sync_run_threads_) {
    if (it.joinable()) {
      it.join();
    }
  }
  sync_run_threads_.clear();
}

ThreadPool::~ThreadPool() {
  try {
    ClearThreadPool();
  } catch (...) {
    // exit
  }
}
}  // namespace common
}  // namespace mindspore
