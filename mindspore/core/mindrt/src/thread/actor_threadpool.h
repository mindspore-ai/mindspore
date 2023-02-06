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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_ACTOR_THREADPOOL_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_ACTOR_THREADPOOL_H_

#include <queue>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "thread/threadpool.h"
#include "thread/core_affinity.h"
#include "actor/actor.h"
#include "thread/hqueue.h"
#ifndef USE_HQUEUE
#define USE_HQUEUE
#endif
namespace mindspore {
class ActorThreadPool;
class ActorWorker : public Worker {
 public:
  explicit ActorWorker(ThreadPool *pool, size_t index) : Worker(pool, index) {}
  void CreateThread() override;
  bool ActorActive();
  ~ActorWorker() override {
    {
      std::lock_guard<std::mutex> _l(mutex_);
      alive_ = false;
    }
    cond_var_.notify_one();

    bool terminate = false;
    int count = 0;
    while (local_task_queue_ && !terminate && count++ < kMaxCount) {
      terminate = local_task_queue_->Empty();
      if (!terminate) {
        auto task_split = local_task_queue_->Dequeue();
        (void)TryRunTask(task_split);
      }
    }

    if (thread_.joinable()) {
      thread_.join();
    }
    local_task_queue_ = nullptr;
  };

 private:
  void RunWithSpin();
  bool RunQueueActorTask();
};

class ActorThreadPool : public ThreadPool {
 public:
  // create ThreadPool that contains actor thread and kernel thread
  static ActorThreadPool *CreateThreadPool(size_t actor_thread_num, size_t all_thread_num, BindMode bind_mode) {
    std::vector<int> core_list;
    return ActorThreadPool::CreateThreadPool(actor_thread_num, all_thread_num, core_list, bind_mode);
  }

  static ActorThreadPool *CreateThreadPool(size_t actor_thread_num, size_t all_thread_num,
                                           const std::vector<int> &core_list, BindMode bind_mode);
  // create ThreadPool that contains only actor thread
  static ActorThreadPool *CreateThreadPool(size_t thread_num);
  ~ActorThreadPool() override;

  static void set_actor_queue_size(size_t actor_queue_size) { actor_queue_size_ = actor_queue_size; }

  virtual int ActorQueueInit();
  virtual void PushActorToQueue(ActorBase *actor);
  virtual ActorBase *PopActorFromQueue();

 protected:
  ActorThreadPool() = default;

  std::mutex actor_mutex_;
  std::condition_variable actor_cond_;
#ifdef USE_HQUEUE
  HQueue<ActorBase> actor_queue_;
#else
  std::queue<ActorBase *> actor_queue_;
#endif

 private:
  int CreateThreads(size_t actor_thread_num, size_t all_thread_num, const std::vector<int> &core_list);

  // Support to set the size of actor queue.
  static size_t actor_queue_size_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_ACTOR_THREADPOOL_H_
