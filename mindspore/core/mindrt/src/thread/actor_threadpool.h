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
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "thread/threadpool.h"
#include "actor/actor.h"
#include "thread/hqueue.h"

namespace mindspore {
enum ThreadPolicy {
  kThreadSpin = 0,  // thread run in spin
  kThreadWait = 1   // synchronous and wait
};

class ActorThreadPool;

class ActorWorker : public Worker {
 public:
  void CreateThread(ActorThreadPool *pool, ThreadPolicy policy);
  bool Active();

 private:
  void RunWithWait();
  void RunWithSpin();
  bool RunQueueActorTask();

  ActorThreadPool *pool_{nullptr};
};

class ActorThreadPool : public ThreadPool {
 public:
  // create ThreadPool that contains actor thread and kernel thread
  static ActorThreadPool *CreateThreadPool(size_t actor_thread_num, size_t all_thread_num, ThreadPolicy policy);
  // create ThreadPool that contains only actor thread
  static ActorThreadPool *CreateThreadPool(size_t thread_num, ThreadPolicy policy);
  ~ActorThreadPool() override;

  void PushActorToQueue(const ActorReference &actor);
  ActorReference PopActorFromQueue();
  void WaitUntilNotify();

 private:
  ActorThreadPool() {}
  int CreateThreads(size_t actor_thread_num, size_t all_thread_num, ThreadPolicy policy);

  size_t actor_thread_num_{0};

  bool exit_{false};
  std::mutex actor_mutex_;
  std::condition_variable actor_cond_;
  std::queue<ActorReference> actor_queue_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_ACTOR_THREADPOOL_H_
