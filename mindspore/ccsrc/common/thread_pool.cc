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

#include "common/thread_pool.h"
#include <algorithm>
#include <exception>
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace common {
#if ENABLE_D || ENABLE_GPU
const size_t kDeviceNum = 8;
#endif
const size_t kMaxThreadNum = 23;

ThreadPool::ThreadPool() {
  size_t process_core_num = std::thread::hardware_concurrency() - 1;
  if (process_core_num < 1) {
    process_core_num = 1;
  }
#if ENABLE_D || ENABLE_GPU
  max_thread_num_ = process_core_num / kDeviceNum;
#else
  max_thread_num_ = process_core_num;
#endif
  if (max_thread_num_ < 1) {
    max_thread_num_ = 1;
  }
  if (max_thread_num_ > kMaxThreadNum) {
    max_thread_num_ = kMaxThreadNum;
  }
}

void ThreadPool::SyncRunLoop() {
  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      task_cond_var_.wait(lock, [this] { return !task_queue_.empty() || exit_run_; });
      if (exit_run_) {
        return;
      }
      task = task_queue_.front();
      task_queue_.pop();
    }
    try {
      task();
    } catch (std::exception &e) {
      MsException::Instance().SetException();
    }
    {
      std::unique_lock<std::mutex> task_lock(task_mutex_);
      task_finished_count_ = task_finished_count_ + 1;
    }
    finished_cond_var_.notify_one();
  }
}

bool ThreadPool::SyncRun(const std::vector<Task> &tasks) {
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
    for (size_t i = thread_num; i < new_thread_num; ++i) {
      sync_run_threads_.emplace_back(std::thread(&ThreadPool::SyncRunLoop, this));
    }
  }

  for (auto &task : tasks) {
    std::lock_guard<std::mutex> task_lock(task_mutex_);
    task_queue_.push(task);
    task_cond_var_.notify_one();
  }
  {
    std::unique_lock<std::mutex> task_lock(task_mutex_);
    finished_cond_var_.wait(task_lock, [this, task_num] { return task_num == task_finished_count_; });
    task_finished_count_ = 0;
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
  task_cond_var_.notify_all();
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
