/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <queue>
#include <string>
#include <new>
#include <vector>
#include <unordered_map>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>
#include "thread/threadlog.h"
#include "thread/core_affinity.h"
#ifndef _WIN32
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) || defined(_M_X64)
#define PLATFORM_86
#include <pmmintrin.h>
#endif
#endif
#include "mindapi/base/macros.h"
#include "thread/hqueue.h"

#define USE_HQUEUE
namespace mindspore {
constexpr int kDefaultSpinCount = 300000;
constexpr int kMaxCount = 30000;
constexpr int kDefaultKernelSpinCount = 3000;
constexpr int kMinSpinCount = 1;
constexpr int kDefaultFrequency = 1;
constexpr float kMaxScale = 1.;
constexpr size_t kMaxHqueueSize = 8192;
constexpr size_t kMinActorRunOther = 2;
/* Thread status */
constexpr int kThreadBusy = 0;  // busy, the thread is running task
constexpr int kThreadHeld = 1;  // held, the thread has been marked as occupied
constexpr int kThreadIdle = 2;  // idle, the thread is waiting

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

typedef struct TaskSplit {
  TaskSplit(Task *task, int task_id) : task_(task), task_id_(task_id) {}
  Task *task_;
  int task_id_;
} TaskSplit;

class ThreadPool;
class Worker {
 public:
  explicit Worker(ThreadPool *pool, size_t index) : pool_(pool), worker_id_(index) {}
  virtual ~Worker();
  // create thread and start running at the same time
  virtual void CreateThread();
  // assign task and then activate thread
  void Active(std::vector<TaskSplit> *task_list, int task_id_start, int task_id_end);
  // activate thread
  void Active();

  // whether or not it is idle and marked as held
  bool available();
  // assigns task first before running
  virtual bool RunLocalKernelTask();
  virtual void RunOtherKernelTask();
  // try to run a single task
  bool TryRunTask(TaskSplit *task_split);
  // set max spin count before running
  void SetMaxSpinCount(int max_spin_count) { max_spin_count_ = max_spin_count; }
  void InitWorkerMask(const std::vector<int> &core_list, const size_t workers_size);
  void InitLocalTaskQueue(HQueue<TaskSplit> *task_queue) { local_task_queue_ = task_queue; }

  void set_frequency(int frequency) { frequency_ = frequency; }
  int frequency() const { return frequency_; }

  void set_scale(float lhs_scale, float rhs_scale);
  float lhs_scale() const { return lhs_scale_; }
  float rhs_scale() const { return rhs_scale_; }
  HQueue<TaskSplit> *local_task_queue() { return local_task_queue_; }

  std::thread::id thread_id() const { return thread_.get_id(); }

#ifdef _WIN32
  uint64_t core_id() { return core_id_; }
#elif defined(BIND_CORE)
  void set_mask(const cpu_set_t &mask) { mask_ = mask; }
  pthread_t handle() { return thread_.native_handle(); }
#endif
  inline void set_alive(bool flag) { alive_ = flag; }
  inline bool alive() const { return alive_; }

 protected:
  void SetAffinity();
  void YieldAndDeactive();
  virtual void WaitUntilActive();

  bool alive_{true};
  std::thread thread_;
#ifdef _WIN32
  uint64_t core_id_;
#elif defined(BIND_CORE)
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
  ThreadPool *pool_{nullptr};
  HQueue<TaskSplit> *local_task_queue_{nullptr};
  size_t worker_id_{0};
  std::vector<int> core_list_;

 private:
  void Run();
};

class MS_CORE_API ThreadPool {
 public:
  static ThreadPool *CreateThreadPool(size_t thread_num, const std::vector<int> &core_list = {});
  virtual ~ThreadPool();

  size_t thread_num() const { return workers_.size(); }
  const std::vector<std::unique_ptr<HQueue<TaskSplit>>> &task_queues() { return task_queues_; }

  int SetCpuAffinity(const std::vector<int> &core_list);
  int SetCpuAffinity(BindMode bind_mode);
  int SetProcessAffinity(BindMode bind_mode) const;
  void SyncRunTask(Task *task, int start_num, int task_num) const;
  int SyncRunFunc(const Func &func, Content content, int start, int end) const;

  virtual int ParallelLaunch(const Func &func, Content content, int task_num);

  void DisableOccupiedActorThread() { occupied_actor_thread_ = false; }
  void SetActorThreadNum(size_t actor_thread_num) { actor_thread_num_ = actor_thread_num; }
  void SetKernelThreadNum(size_t kernel_thread_num) { kernel_thread_num_ = kernel_thread_num; }
  size_t GetKernelThreadNum() const { return kernel_thread_num_ + actor_thread_num_; }
  size_t GetActorThreadNum() const { return actor_thread_num_; }
  void SetKernelThreadMaxSpinCount(int spin_count);
  void SetSpinCountMaxValue();
  void SetSpinCountMinValue();
  void SetMaxSpinCount(int spin_count);
  void SetMinSpinCount(int spin_count);
  void ActiveWorkers();
  void SetWorkerIdMap();
  // init task queues
  int TaskQueuesInit(size_t thread_num);
  const std::unordered_map<std::thread::id, size_t> &GetWorkerIdMap() const { return worker_ids_; }
  float GetServerCpuFrequence() const { return server_cpu_frequence; }
  inline size_t actor_thread_num() const { return actor_thread_num_; }
  virtual bool SetRunnerID(const std::string &runner_id) { return false; }
  template <typename T = Worker>
  int CreateThreads(size_t thread_num, const std::vector<int> &core_list) {
    size_t core_num = std::thread::hardware_concurrency();
    thread_num = thread_num < core_num ? thread_num : core_num;
    THREAD_INFO("ThreadInfo, Num: [%zu], CoreNum: [%zu]", thread_num, core_num);
    if (thread_num == 0) {
      THREAD_INFO("Current thread as working thread.");
      return THREAD_OK;
    }
    std::lock_guard<std::mutex> _l(pool_mutex_);
    size_t start = workers_.size();
    for (size_t i = 0; i < thread_num; ++i) {
      auto worker = new (std::nothrow) T(this, workers_.size());
      THREAD_ERROR_IF_NULL(worker);
      worker->InitWorkerMask(core_list, workers_.size());
      size_t queues_idx = start + i;
      if (queues_idx >= task_queues_.size()) {
        THREAD_ERROR("task_queues out of range.");
        return THREAD_ERROR;
      }
      worker->InitLocalTaskQueue(task_queues_[queues_idx].get());
      workers_.push_back(worker);
    }
    for (size_t i = 0; i < thread_num; ++i) {
      workers_[start + i]->CreateThread();
      THREAD_INFO("create kernel thread[%zu]", i);
    }
    return THREAD_OK;
  }

 protected:
  ThreadPool() = default;

  int InitAffinityInfo();

  void DistributeTask(std::vector<TaskSplit> *task_list, Task *task, int task_num, Worker *curr) const;
  void CalculateScales(const std::vector<Worker *> &workers, int sum_frequency) const;
  void ActiveWorkers(const std::vector<Worker *> &workers, std::vector<TaskSplit> *task_list, int task_num,
                     const Worker *curr) const;

  Worker *CurrentWorker(size_t *index) const;
  Worker *CurrentWorker() const;

  std::mutex pool_mutex_;
  std::vector<Worker *> workers_;
  std::vector<std::unique_ptr<HQueue<TaskSplit>>> task_queues_;
  std::unordered_map<std::thread::id, size_t> worker_ids_;
  CoreAffinity *affinity_{nullptr};
  size_t actor_thread_num_{0};
  size_t kernel_thread_num_{0};
  bool occupied_actor_thread_{true};
  int max_spin_count_{kDefaultSpinCount};
  int min_spin_count_{kMinSpinCount};
  float server_cpu_frequence = -1.0f;  // Unit : GHz
  static std::mutex create_thread_pool_muntex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_H_
