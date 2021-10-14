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

#include <new>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>
#include "thread/threadlog.h"
#include "thread/core_affinity.h"

namespace mindspore {
constexpr int kDefaultSpinCount = 300000;
constexpr int kMaxCount = 30000;
constexpr int kMinSpinCount = 1;
constexpr int kDefaultFrequency = 1;
constexpr float kMaxScale = 1.;

enum ThreadStatus {
  kThreadBusy = 0,  // busy, the thread is running task
  kThreadHeld = 1,  // held, the thread has been marked as occupied
  kThreadIdle = 2   // idle, the thread is waiting
};

// used in scenarios with unequal division of task
// the parameters indicate the start and end coefficients
using Func = std::function<int(void *, int, float, float)>;
using Content = void *;

typedef struct Task {
  Task(Func f, Content c) : func(f), content(c) {}
  Func func;
  Content content;
  std::atomic_int finished{0};
  std::atomic_int status{THREAD_OK};  // return status, RET_OK
} Task;

class Worker {
 public:
  Worker() = default;
  virtual ~Worker();
  // create thread and start running at the same time
  void CreateThread();
  // assign task and then activate thread
  void Active(Task *task, int task_id);
  // activate thread
  void Active();
  // whether or not it is idle and marked as held
  bool available();
  // assigns task first before running
  bool RunLocalKernelTask();
  // set max spin count before running
  void SetMaxSpinCount(int max_spin_count) { max_spin_count_ = max_spin_count; }

  void set_frequency(int frequency) { frequency_ = frequency; }
  int frequency() const { return frequency_; }

  void set_scale(float lhs_scale, float rhs_scale);
  float lhs_scale() const { return lhs_scale_; }
  float rhs_scale() const { return rhs_scale_; }

  std::thread::id thread_id() const { return thread_.get_id(); }
#ifdef BIND_CORE
  void set_mask(const cpu_set_t &mask) { mask_ = mask; }
  pthread_t handle() { return thread_.native_handle(); }
#endif

 protected:
  void SetAffinity();
  void Run();
  void YieldAndDeactive();
  void WaitUntilActive();

  bool alive_{true};
  std::thread thread_;
#ifdef BIND_CORE
  cpu_set_t mask_;
#endif
  std::atomic_int status_{kThreadBusy};
  std::atomic_int active_num_{0};

  std::mutex mutex_;
  std::condition_variable cond_var_;

  std::atomic<Task *> task_{nullptr};
  std::atomic_int task_id_{0};
  float lhs_scale_{0.};
  float rhs_scale_{kMaxScale};
  int frequency_{kDefaultFrequency};
  int spin_count_{0};
  int max_spin_count_{kMinSpinCount};
};

class ThreadPool {
 public:
  static ThreadPool *CreateThreadPool(size_t thread_num, const std::vector<int> &core_list = {});
  virtual ~ThreadPool();

  size_t thread_num() const { return workers_.size(); }

  int SetCpuAffinity(const std::vector<int> &core_list);
  int SetCpuAffinity(BindMode bind_mode);
  int SetProcessAffinity(BindMode bind_mode) const;

  int ParallelLaunch(const Func &func, Content content, int task_num) const;
  void DisableOccupiedActorThread() { occupied_actor_thread_ = false; }
  void SetActorThreadNum(size_t actor_thread_num) { actor_thread_num_ = actor_thread_num; }
  void SetKernelThreadNum(size_t kernel_thread_num) { kernel_thread_num_ = kernel_thread_num; }
  size_t GetKernelThreadNum() const { return kernel_thread_num_; }
  void SetSpinCountMaxValue();
  void SetSpinCountMinValue();
  void SetMaxSpinCount(int spin_count);
  void SetMinSpinCount(int spin_count);
  void ActiveWorkers() const;

 protected:
  ThreadPool() = default;

  int CreateThreads(size_t thread_num, const std::vector<int> &core_list);

  int InitAffinityInfo();

  void SyncRunTask(Task *task, int start_num, int task_num) const;

  void DistributeTask(Task *task, int task_num) const;
  void CalculateScales(const std::vector<Worker *> &workers, int sum_frequency) const;
  void ActiveWorkers(const std::vector<Worker *> &workers, Task *task, int task_num, const Worker *curr) const;

  Worker *CurrentWorker() const;

  std::mutex pool_mutex_;
  std::vector<Worker *> workers_;
  CoreAffinity *affinity_{nullptr};
  size_t actor_thread_num_{0};
  size_t kernel_thread_num_{0};
  bool occupied_actor_thread_{true};
  int max_spin_count_{kDefaultSpinCount};
  int min_spin_count_{kMinSpinCount};
};

}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_H_
