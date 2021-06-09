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
  {
    THREAD_INFO("wait util actor queue is empty");
    std::unique_lock<std::mutex> _l(actor_mutex_);
    finish_cond_var_.wait(_l, [this]() { return actor_queue_.empty(); });
  }
  exit_ = true;
  alive_ = false;
  actor_cond_var_.notify_all();
  DestructThreads();
}

void InterThreadPool::ThreadAsyncRun(Worker *worker) {
  THREAD_RETURN_IF_NULL(worker);
  while (alive_) {
    if (worker->type == kKernelThread) {
      KernelThreadRun(worker);
    } else if (worker->type == kActorThread) {
      ActorThreadRun();
    }
  }
}

void InterThreadPool::ActorThreadRun() {
#ifndef SUPPORT_NNIE
  ActorReference actor;
  {
    std::unique_lock<std::mutex> _l(actor_mutex_);
    actor_cond_var_.wait(_l, [&]() { return !actor_queue_.empty() || exit_; });
    if (exit_ && actor_queue_.empty()) {
      return;
    }
    actor = actor_queue_.front();
    actor_queue_.pop();
  }
  actor->Run();
  finish_cond_var_.notify_one();
#endif
}

void InterThreadPool::EnqueReadyActor(const ActorReference &actor) {
  {
    std::lock_guard<std::mutex> _l(actor_mutex_);
    actor_queue_.push(actor);
  }
  actor_cond_var_.notify_one();
  THREAD_INFO("actor enqueue success");
}

InterThreadPool *InterThreadPool::CreateThreadPool(size_t inter_thread_num, size_t intra_thread_num) {
  InterThreadPool *pool = new (std::nothrow) InterThreadPool(inter_thread_num);
  if (pool == nullptr) {
    return nullptr;
  }
  size_t thread_num = inter_thread_num * intra_thread_num;
  int ret = pool->CreateThreads(thread_num);
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
  InterThreadPool *pool = new (std::nothrow) InterThreadPool(thread_num);
  if (pool == nullptr) {
    return nullptr;
  }
  int ret = pool->CreateThreads(thread_num);
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  return pool;
}
}  // namespace mindspore
