/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_THREAD_POOL_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_THREAD_POOL_H_

#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <queue>
#include <string>
#include <atomic>
#include <memory>
#include <utility>
#include <functional>
#include <iostream>
#include "utils/log_adapter.h"
#include "include/common/visible.h"

#ifdef PARALLEL_INFERENCE
#define BIND_CORE
#endif
#ifdef __ANDROID__
#define BIND_CORE
#include <sched.h>
#endif
#ifdef __linux__
#define BIND_CORE
#endif

namespace mindspore {
namespace common {
enum Status { FAIL = -1, SUCCESS = 0 };
using Task = std::function<Status()>;

struct ThreadContext {
  std::mutex mutex;
  std::unique_ptr<std::condition_variable> cond_var{nullptr};
  const Task *task{nullptr};

  ThreadContext() {
    cond_var = std::make_unique<std::condition_variable>();
    MS_EXCEPTION_IF_NULL(cond_var);
  }
};

class COMMON_EXPORT ThreadPool {
 public:
  ~ThreadPool();
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  static ThreadPool &GetInstance();
  bool SyncRun(const std::vector<Task> &tasks, const std::vector<int> &core_list = {});
  size_t GetSyncRunThreadNum() const { return max_thread_num_; }
  void ClearThreadPool();
  void ChildAfterFork();

 private:
  ThreadPool();
  void SyncRunLoop(const std::shared_ptr<ThreadContext> &context);
  bool SetCpuAffinity(const std::vector<int> &core_list);
  bool FreeScheduleThreads(const std::vector<int> &core_list);
#ifdef _WIN32
  bool ThreadPool::SetAffinity() const;
#elif defined(BIND_CORE)
  bool SetAffinity(const pthread_t &thread_id, cpu_set_t *cpu_set);
#endif

  size_t max_thread_num_{1};
  std::mutex pool_mtx_;
  std::atomic_bool exit_run_ = {false};
  std::vector<std::unique_ptr<std::thread>> sync_run_threads_{};
  std::vector<std::shared_ptr<ThreadContext>> contexts_;
};
}  // namespace common
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_THREAD_POOL_H_
