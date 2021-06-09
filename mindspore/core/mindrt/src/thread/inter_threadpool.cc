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

#include "thread/inter_threadpool.h"
#include "thread/core_affinity.h"

namespace mindspore {
InterThreadPool::~InterThreadPool() {
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

void InterThreadPool::AsyncRunMultiTask(Worker *worker) {
  THREAD_RETURN_IF_NULL(worker);
  while (alive_) {
    if (RunLocalKernelTask(worker) || RunPoolQueueActorTask(worker)) {
      worker->spin = 0;
    } else {
      // deactivate this worker only on the first entry
      if (worker->spin == 0) {
        worker->running = false;
      }
      worker->spin++;
      std::this_thread::yield();
    }
    if (worker->spin >= kDefaultSpinCount) {
      WaitUntilActivate(worker);
    }
  }
}

bool InterThreadPool::RunPoolQueueActorTask(Worker *worker) {
  ActorBase *actor = nullptr;
  if (!PopActorFromQueue(&actor)) {
    return false;
  }
  if (actor != nullptr) {
    actor->Run();
  }
  return true;
}

bool InterThreadPool::PopActorFromQueue(ActorBase **actor) {
  std::lock_guard<std::mutex> _l(actor_mutex_);
  if (actor_queue_.empty()) {
    return false;
  }
  *actor = actor_queue_.front().get();
  actor_queue_.pop();
  return true;
}

void InterThreadPool::EnqueReadyActor(const ActorReference &actor) {
  {
    std::lock_guard<std::mutex> _l(actor_mutex_);
    actor_queue_.push(actor);
  }
  THREAD_INFO("actor[%s] enqueue success", actor->GetAID().Name().c_str());
  // active one free actor thread
  for (size_t i = 0; i < actor_thread_num_; ++i) {
    bool expected = false;
    if (workers_[i]->running.compare_exchange_strong(expected, true)) {
      workers_[i]->cond_var.notify_one();
      break;
    }
  }
}

int InterThreadPool::CreateThreads(size_t actor_thread_num, size_t all_thread_num) {
  size_t core_num = std::thread::hardware_concurrency();
  THREAD_INFO("actor_thread_num: %zu, actor_thread_num: [%zu], core_num: [%zu]", actor_thread_num, all_thread_num,
              core_num);
  actor_thread_num_ = actor_thread_num < core_num ? actor_thread_num : core_num;
  if (actor_thread_num_ <= 0) {
    THREAD_ERROR("thread num is invalid");
    return THREAD_ERROR;
  }
  for (size_t i = 0; i < actor_thread_num_; ++i) {
    std::lock_guard<std::mutex> _l(pool_mutex_);
    auto worker = new (std::nothrow) Worker();
    THREAD_ERROR_IF_NULL(worker);
    worker->thread = std::thread(&InterThreadPool::AsyncRunMultiTask, this, worker);
    workers_.push_back(worker);
    THREAD_INFO("create actor thread[%zu]", i);
  }
  size_t kernel_thread_num = all_thread_num - actor_thread_num_;
  if (kernel_thread_num > 0) {
    return ThreadPool::CreateThreads(kernel_thread_num);
  }
  return THREAD_OK;
}

InterThreadPool *InterThreadPool::CreateThreadPool(size_t actor_thread_num, size_t all_thread_num) {
  InterThreadPool *pool = new (std::nothrow) InterThreadPool();
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

InterThreadPool *InterThreadPool::CreateThreadPool(size_t thread_num) {
  InterThreadPool *pool = new (std::nothrow) InterThreadPool();
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
