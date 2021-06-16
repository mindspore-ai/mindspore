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

#include "thread/actor_threadpool.h"
#include "thread/core_affinity.h"

namespace mindspore {
ActorThreadPool::~ActorThreadPool() {
  // wait until actor queue is empty
  bool terminate = false;
  do {
    std::lock_guard<std::mutex> _l(actor_mutex_);
    if (actor_queue_.empty()) {
      terminate = true;
    } else {
      std::this_thread::yield();
    }
  } while (!terminate);
  alive_ = false;
  DestructThreads();
}

void ActorThreadPool::AsyncRunMultiTask(Worker *worker) {
  THREAD_RETURN_IF_NULL(worker);
  while (alive_) {
    // only run either local KernelTask or PoolQueue ActorTask
    bool busy = RunLocalKernelTask(worker) || RunPoolQueueActorTask();
    if (!busy) {
      // wait until Actor enqueue or distribute KernelTask
      std::unique_lock<std::mutex> _l(worker->mutex);
      worker->status = kThreadIdle;
      worker->cond_var.wait(_l, [&] { return worker->status == kThreadBusy || !alive_; });
    }
  }
}

bool ActorThreadPool::RunPoolQueueActorTask() {
  ActorBase *actor = nullptr;
  if (!PopActorFromQueue(&actor)) {
    return false;
  }
  if (actor != nullptr) {
    actor->Run();
  }
  return true;
}

bool ActorThreadPool::PopActorFromQueue(ActorBase **actor) {
  std::lock_guard<std::mutex> _l(actor_mutex_);
  if (actor_queue_.empty()) {
    return false;
  }
  *actor = actor_queue_.front().get();
  actor_queue_.pop();
  return true;
}

void ActorThreadPool::EnqueReadyActor(const ActorReference &actor) {
  {
    std::lock_guard<std::mutex> _l(actor_mutex_);
    actor_queue_.push(actor);
  }
  THREAD_INFO("actor[%s] enqueue success", actor->GetAID().Name().c_str());
  // active one idle actor thread if exist
  for (size_t i = 0; i < actor_thread_num_; ++i) {
    std::lock_guard<std::mutex> _l(workers_[i]->mutex);
    if (workers_[i]->status == kThreadIdle) {
      workers_[i]->status = kThreadBusy;
      workers_[i]->cond_var.notify_one();
      break;
    }
  }
}

int ActorThreadPool::CreateThreads(size_t actor_thread_num, size_t all_thread_num) {
  size_t core_num = std::thread::hardware_concurrency();
  THREAD_INFO("ThreadInfo, Actor: [%zu], All: [%zu], CoreNum: [%zu]", actor_thread_num, all_thread_num, core_num);
  actor_thread_num_ = actor_thread_num < core_num ? actor_thread_num : core_num;
  if (actor_thread_num_ <= 0 || actor_thread_num > all_thread_num) {
    THREAD_ERROR("thread num is invalid");
    return THREAD_ERROR;
  }
  for (size_t i = 0; i < actor_thread_num_; ++i) {
    std::lock_guard<std::mutex> _l(pool_mutex_);
    auto worker = new (std::nothrow) Worker();
    THREAD_ERROR_IF_NULL(worker);
    worker->thread = std::thread(&ActorThreadPool::AsyncRunMultiTask, this, worker);
    workers_.push_back(worker);
    THREAD_INFO("create actor thread[%zu]", i);
  }
  size_t kernel_thread_num = all_thread_num - actor_thread_num_;
  if (kernel_thread_num > 0) {
    return ThreadPool::CreateThreads(kernel_thread_num);
  }
  return THREAD_OK;
}

ActorThreadPool *ActorThreadPool::CreateThreadPool(size_t actor_thread_num, size_t all_thread_num) {
  ActorThreadPool *pool = new (std::nothrow) ActorThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
  int ret = pool->CreateThreads(actor_thread_num, all_thread_num);
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
#ifdef BIND_CORE
  ret = pool->InitAffinityInfo();
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
#endif  // BIND_CORE
  return pool;
}

ActorThreadPool *ActorThreadPool::CreateThreadPool(size_t thread_num) {
  ActorThreadPool *pool = new (std::nothrow) ActorThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
  int ret = pool->CreateThreads(thread_num, thread_num);
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  return pool;
}
}  // namespace mindspore
