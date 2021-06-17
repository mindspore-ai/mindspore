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
#include "thread/threadlog.h"
#include "thread/core_affinity.h"

namespace mindspore {
constexpr int kDefaultSpinCount = 300000;
constexpr int kDefaultFrequency = 1;
constexpr float kMaxScale = 1.;

// used in scenarios with unequal division of task
// the parameters indicate the start and end coefficients
using Func = int (*)(void *, int, float, float);
using Content = void *;

typedef struct Task {
  Task(Func f, Content c) : func(f), content(c) {}
  Func func;
  Content content;
  std::atomic_int finished{0};
  std::atomic_int status{THREAD_OK};  // return status, RET_OK
} Task;

// busy, the thread is running task
// held, the thread has been marked as occupied
// idle, the thread is waiting
enum ThreadStatus { kThreadBusy = 0, kThreadHeld = 1, kThreadIdle = 2 };

typedef struct Worker {
  std::thread thread;
  std::atomic_int status{kThreadBusy};
  std::mutex mutex;
  std::condition_variable cond_var;
  std::atomic<Task *> task{nullptr};
  std::atomic_int task_id{0};
  float lhs_scale{0.};
  float rhs_scale{kMaxScale};
  int frequency{kDefaultFrequency};
  int spin{0};
} Worker;

class ThreadPool {
 public:
  static ThreadPool *CreateThreadPool(size_t thread_num);
  virtual ~ThreadPool();

  size_t thread_num() const { return workers_.size(); }

  int SetCpuAffinity(const std::vector<int> &core_list);
  int SetCpuAffinity(BindMode bind_mode);
  int SetProcessAffinity(BindMode bind_mode) const;

  int ParallelLaunch(const Func &func, Content content, int task_num) const;

 protected:
  ThreadPool() = default;

  int CreateThreads(size_t thread_num);
  void DestructThreads();

  int InitAffinityInfo();

  void AsyncRunTask(Worker *worker) const;
  void SyncRunTask(Task *task, int task_num) const;

  void DistributeTask(Task *task, int task_num) const;
  void CalculateScales(const std::vector<Worker *> &workers, int sum_frequency) const;
  void ActiveWorkers(const std::vector<Worker *> &workers, Task *task, int task_num, const Worker *curr) const;
  void YieldAndDeactive(Worker *worker) const;

  bool RunLocalKernelTask(Worker *worker) const;

  Worker *CurrentWorker() const;

  std::mutex pool_mutex_;
  std::atomic_bool alive_{true};

  std::vector<Worker *> workers_;

  CoreAffinity *affinity_{nullptr};
};

}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_H_
