/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_H_

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <new>
#include "thread/core_affinity.h"

namespace mindspore {

#ifdef THREAD_POOL_DEBUG
#include <stdio.h>
#define THREAD_INFO(content, args...) \
  { printf("[INFO] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }
#define THREAD_ERROR(content, args...) \
  { printf("[ERROR] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }
#else
#define THREAD_INFO(content, args...)
#define THREAD_ERROR(content, args...)
#endif

#define THREAD_ERROR_IF_NULL(ptr) \
  do {                            \
    if ((ptr) == nullptr) {       \
      return THREAD_ERROR;        \
    }                             \
  } while (0)

#define THREAD_RETURN_IF_NULL(ptr) \
  do {                             \
    if ((ptr) == nullptr) {        \
      return;                      \
    }                              \
  } while (0)

enum ThreadRet { THREAD_OK = 0, THREAD_ERROR = 1 };
enum ThreadType { kActorThread = 0, kKernelThread = 1 };

using Func = int (*)(void *arg, int);
using Contend = void *;

typedef struct Task {
  Task(Func f, Contend c) : func(f), content(c) {}
  Func func;
  Contend content;
  std::atomic_int task_id{0};
  std::atomic_int finished{0};
  std::atomic_int status{THREAD_OK};  // return status, RET_OK
} Task;

typedef struct Worker {
  std::thread thread;
  std::atomic_int type{kActorThread};
  std::atomic_bool active{false};
  Task *task{nullptr};
  std::mutex mutex;
  std::condition_variable cond_var;
  int spin{0};
} Worker;

class ThreadPool {
 public:
  static ThreadPool *CreateThreadPool(size_t thread_num);
  virtual ~ThreadPool();

  size_t thread_num() const { return thread_num_; }

  int SetCpuAffinity(const std::vector<int> &core_list);
  int SetCpuAffinity(BindMode bind_mode);

  int SetProcessAffinity(BindMode bind_mode) const;

  int ParallelLaunch(const Func &func, Contend contend, int task_num);

 protected:
  ThreadPool() = default;

  int CreateThreads(size_t thread_num);
  void DestructThreads();

  int InitAffinityInfo();

  virtual void ThreadAsyncRun(Worker *worker);
  void KernelThreadRun(Worker *worker);

  void DistributeTask(Task *task, int task_num);

  std::mutex pool_mutex_;

  std::vector<Worker *> workers_;
  std::vector<Worker *> freelist_;
  std::atomic_bool alive_{true};

  size_t inter_thread_num_{0};
  size_t thread_num_{1};

  CoreAffinity *affinity_{nullptr};
};

}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_H_
