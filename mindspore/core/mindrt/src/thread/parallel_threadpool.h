/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_PARALLEL_THREADPOOL_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_PARALLEL_THREADPOOL_H_

#include <queue>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "thread/actor_threadpool.h"

namespace mindspore {
typedef struct Distributor {
  int started = 0;
  int task_num = 0;
} Distributor;
typedef struct ParallelTask : public Task {
  ParallelTask() : Task(nullptr, nullptr) {}
  std::atomic<Distributor> distributor;
  std::atomic_bool valid = false;
  std::atomic_bool occupied = false;
} ParallelTask;
class ParallelThreadPool;
class ParallelWorker : public Worker {
 public:
  explicit ParallelWorker(ThreadPool *pool, size_t index) : Worker(pool, index) {
    parallel_pool_ = reinterpret_cast<ParallelThreadPool *>(pool_);
  }
  void CreateThread() override;
  bool RunLocalKernelTask() override;
  ~ParallelWorker() override {
    {
      std::lock_guard<std::mutex> _l(mutex_);
      alive_ = false;
    }

    Active();
    if (thread_.joinable()) {
      thread_.join();
    }
    pool_ = nullptr;
    parallel_pool_ = nullptr;
  }

 protected:
  void WaitUntilActive() override;

 private:
  void Run() override;
  bool RunQueueActorTask();
  ParallelThreadPool *parallel_pool_{nullptr};
};

class ParallelThreadPool : public ActorThreadPool {
 public:
  static ParallelThreadPool *CreateThreadPool(size_t actor_thread_num, size_t all_thread_num,
                                              const std::vector<int> &core_list, BindMode bind_mode);
  ~ParallelThreadPool() override {
    // wait until actor queue is empty
    bool terminate = false;
    int count = 0;
    do {
      {
#ifdef USE_HQUEUE
        terminate = actor_queue_.Empty();
#else
        std::lock_guard<std::mutex> _l(actor_mutex_);
        terminate = actor_queue_.empty();
#endif
      }
      if (!terminate) {
        ActiveWorkers();
        std::this_thread::yield();
      }
    } while (!terminate && count++ < kMaxCount);
    for (auto &worker : workers_) {
      worker->set_alive(false);
    }
    ActiveWorkers();
    for (auto &worker : workers_) {
      delete worker;
      worker = nullptr;
    }
    workers_.clear();
    tasks_size_ = 0;
    if (tasks_) {
      delete[] tasks_;
    }
  }

  int ParallelLaunch(const Func &func, Content content, int task_num) override;

  void PushActorToQueue(ActorBase *actor) override {
    if (!actor) {
      return;
    }
    {
#ifdef USE_HQUEUE
      while (!actor_queue_.Enqueue(actor)) {
      }
#else
      std::lock_guard<std::mutex> _l(actor_mutex_);
      actor_queue_.push(actor);
#endif
    }
    THREAD_DEBUG("actor[%s] enqueue success", actor->GetAID().Name().c_str());
    size_t size = workers_.size() > tasks_size_ ? tasks_size_ : workers_.size();
    for (size_t i = 0; i < size; i++) {
      workers_[i]->Active();
    }
  }

  inline bool RunTaskOnce(int start, int end);

  bool RunParallel();

  size_t tasks_size() const { return tasks_size_; }

 private:
  ParallelThreadPool() {}
  int CreateParallelThreads(size_t actor_thread_num, size_t all_thread_num, const std::vector<int> &core_list);

  std::atomic_int tasks_start_ = 0;
  std::atomic_int tasks_end_ = 0;
  ParallelTask *tasks_;
  size_t tasks_size_ = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_PARALLEL_THREADPOOL_H_
