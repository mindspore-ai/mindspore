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
#define USE_HQUEUE
namespace mindspore {
class ActorThreadPool;

class ActorWorker : public Worker {
 public:
  void CreateThread(ActorThreadPool *pool);
  bool ActorActive();

 private:
  void RunWithSpin();
  bool RunQueueActorTask();

  ActorThreadPool *pool_{nullptr};
};

class ActorThreadPool : public ThreadPool {
 public:
  // create ThreadPool that contains actor thread and kernel thread
  static ActorThreadPool *CreateThreadPool(size_t actor_thread_num, size_t all_thread_num, BindMode bind_mode);

  static ActorThreadPool *CreateThreadPool(size_t actor_thread_num, size_t all_thread_num,
                                           const std::vector<int> &core_list);
  // create ThreadPool that contains only actor thread
  static ActorThreadPool *CreateThreadPool(size_t thread_num);
  ~ActorThreadPool() override;

  void PushActorToQueue(ActorBase *actor);
  ActorBase *PopActorFromQueue();

 private:
  ActorThreadPool() {}
  int CreateThreads(size_t actor_thread_num, size_t all_thread_num, const std::vector<int> &core_list);
  size_t actor_thread_num_{0};

  std::mutex actor_mutex_;
  std::condition_variable actor_cond_;
#ifdef USE_HQUEUE
  HQueue<ActorBase> actor_queue_;
#else
  std::queue<ActorBase *> actor_queue_;
#endif
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_ACTOR_THREADPOOL_H_
