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

#include "include/utils/thread_pool.h"
#include <exception>
#include "thread/threadlog.h"
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace common {
constexpr size_t kYieldThreshold = 1000;

ThreadPool::ThreadPool() {
  auto env_thread_num = common::GetEnv("MS_DEV_PARAM_INIT_WORKER_NUM");
  if (!env_thread_num.empty()) {
    MS_LOG(WARNING) << "Manually set max_thread_num_ to " << this->max_thread_num_;
    this->max_thread_num_ = static_cast<size_t>(std::stoi(env_thread_num));
    return;
  }
  constexpr size_t LOWER_BOUND = 1;
  constexpr size_t UPPER_BOUND = 4;
  size_t cnt = std::thread::hardware_concurrency() / 8;
  cnt = std::min(cnt, UPPER_BOUND);
  cnt = (cnt > 1 && cnt % 2 != 0) ? cnt - 1 : cnt;
  cnt = std::max(cnt, LOWER_BOUND);
  this->max_thread_num_ = cnt;
  MS_LOG(INFO) << "Set max_thread_num_ to " << this->max_thread_num_;
}

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
      MS_EXCEPTION_IF_NULL(context->cond_var);
      ++yield_count;
      if (yield_count > kYieldThreshold) {
        yield_count = 0;
        std::unique_lock<std::mutex> lock(context->mutex);
        context->cond_var->wait(lock, [&context, this] { return context->task != nullptr || exit_run_; });
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
#ifdef _WIN32
bool ThreadPool::SetAffinity() const { return false; }
#elif defined(BIND_CORE)
bool ThreadPool::SetAffinity(const pthread_t &thread_id, cpu_set_t *cpu_set) {
  if (cpu_set == nullptr) {
    return false;
  }
#ifdef __ANDROID__
#if __ANDROID_API__ >= 21
  THREAD_INFO("thread: %d, mask: %lu", pthread_gettid_np(thread_id), cpu_set->__bits[0]);
  int ret = sched_setaffinity(pthread_gettid_np(thread_id), sizeof(cpu_set_t), cpu_set);
  if (ret != THREAD_OK) {
    THREAD_ERROR("bind thread %d to cpu failed. ERROR %d", pthread_gettid_np(thread_id), ret);
    return false;
  }
  return true;
#endif
#else
#if defined(__APPLE__)
  THREAD_ERROR("not bind thread to apple's cpu.");
  return false;
#else
  int ret = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), cpu_set);
  if (ret != THREAD_OK) {
    THREAD_ERROR("set thread: %lu to cpu failed", thread_id);
    return false;
  }
  return true;
#endif  // __APPLE__
#endif  // __ANDROID__
  return false;
}
#endif  // __BIND_CORE__

bool ThreadPool::FreeScheduleThreads(const std::vector<int> &core_list) {
  if (core_list.empty()) {
    return false;
  }
#ifdef _WIN32
  return false;
#elif defined(BIND_CORE)
  for (const auto &sync_run_thread : sync_run_threads_) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (auto core_id : core_list) {
      CPU_SET(core_id, &mask);
    }
    if (!SetAffinity(sync_run_thread->native_handle(), &mask)) {
      return false;
    }
  }
  return true;
#endif  // BIND_CORE
  return false;
}

bool ThreadPool::SetCpuAffinity(const std::vector<int> &core_list) {
  if (core_list.empty()) {
    return false;
  }
#ifdef _WIN32
  return false;
#elif defined(BIND_CORE)
  for (size_t i = 0; i < sync_run_threads_.size(); i++) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_list[i % core_list.size()], &mask);
    if (!SetAffinity(sync_run_threads_[i]->native_handle(), &mask)) {
      return false;
    }
  }
  return true;
#endif  // BIND_CORE
  return false;
}

bool ThreadPool::SyncRun(const std::vector<Task> &tasks, const std::vector<int> &core_list) {
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
      sync_run_threads_.emplace_back(std::make_unique<std::thread>(&ThreadPool::SyncRunLoop, this, contexts_[i]));
    }
  }
  if (contexts_.empty()) {
    return true;
  }
  auto set_affinity_ret = SetCpuAffinity(core_list);
  if (set_affinity_ret) {
    MS_LOG(INFO) << "Set cpu affinity success.";
  } else {
    MS_LOG(DEBUG) << "Set cpu affinity failed.";
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
      MS_EXCEPTION_IF_NULL(contexts_[i]->cond_var);
      auto &task_run = contexts_[i]->task;
      if (task_run) {
        running = true;
      } else if (task_index < task_num) {
        std::lock_guard<std::mutex> task_lock(contexts_[i]->mutex);
        contexts_[i]->task = &(tasks[task_index]);
        contexts_[i]->cond_var->notify_one();
        running = true;
        ++task_index;
      }
    }
    if (running) {
      std::this_thread::yield();
    }
  }
  auto free_schedule_threads_ret = FreeScheduleThreads(core_list);
  if (free_schedule_threads_ret) {
    MS_LOG(INFO) << "Free schedule threads success.";
  } else {
    MS_LOG(DEBUG) << "Free schedule threads failed.";
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
    context->cond_var->notify_one();
  }
  for (auto &it : sync_run_threads_) {
    MS_EXCEPTION_IF_NULL(it);
    if (it->joinable()) {
      it->join();
    }
  }
  sync_run_threads_.clear();
}

void ThreadPool::ChildAfterFork() {
  THREAD_INFO("common thread pool clear thread after fork in child process");
  for (auto &context : contexts_) {
    MS_EXCEPTION_IF_NULL(context);
    if (context->cond_var != nullptr) {
      (void)context->cond_var.release();
      context->task = nullptr;
    }
  }
  contexts_.clear();
  for (auto &it : sync_run_threads_) {
    if (it != nullptr) {
      (void)it.release();
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
