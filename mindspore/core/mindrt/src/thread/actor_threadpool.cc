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
#ifndef _MSC_VER
#include <sched.h>
#include <unistd.h>
#endif
#include "thread/actor_threadpool.h"
#include "thread/core_affinity.h"

namespace mindspore {
constexpr size_t MAX_READY_ACTOR_NR = 4096;
void ActorWorker::CreateThread(ActorThreadPool *pool) {
  THREAD_RETURN_IF_NULL(pool);
  pool_ = pool;
  thread_ = std::thread(&ActorWorker::RunWithSpin, this);
}

void ActorWorker::RunWithSpin() {
  SetAffinity();
#if !defined(__APPLE__) && !defined(SUPPORT_MSVC)
  static std::atomic_int index = {0};
  (void)pthread_setname_np(pthread_self(), ("ActorThread_" + std::to_string(index++)).c_str());
#endif
  while (alive_) {
    // only run either local KernelTask or PoolQueue ActorTask
    if (RunLocalKernelTask() || RunQueueActorTask()) {
      spin_count_ = 0;
    } else {
      YieldAndDeactive();
    }
    if (spin_count_ > max_spin_count_) {
      WaitUntilActive();
      spin_count_ = 0;
    }
  }
}

bool ActorWorker::RunQueueActorTask() {
  THREAD_ERROR_IF_NULL(pool_);
  auto actor = pool_->PopActorFromQueue();
  if (actor == nullptr) {
    return false;
  }
  actor->Run();
  return true;
}

bool ActorWorker::ActorActive() {
  if (status_ != kThreadIdle) {
    return false;
  }
  {
    std::lock_guard<std::mutex> _l(mutex_);
    active_num_++;
    status_ = kThreadBusy;
  }
  cond_var_.notify_one();
  return true;
}

ActorThreadPool::~ActorThreadPool() {
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
      for (auto &worker : workers_) {
        worker->Active();
      }
      std::this_thread::yield();
    }
  } while (!terminate && count++ < kMaxCount);
  for (auto &worker : workers_) {
    delete worker;
    worker = nullptr;
  }
  workers_.clear();
#ifdef USE_HQUEUE
  actor_queue_.Clean();
#endif
}

ActorBase *ActorThreadPool::PopActorFromQueue() {
#ifdef USE_HQUEUE
  return actor_queue_.Dequeue();
#else
  std::lock_guard<std::mutex> _l(actor_mutex_);
  if (actor_queue_.empty()) {
    return nullptr;
  }
  auto actor = actor_queue_.front();
  actor_queue_.pop();
  return actor;
#endif
}

void ActorThreadPool::PushActorToQueue(ActorBase *actor) {
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
  // active one idle actor thread if exist
  for (size_t i = 0; i < actor_thread_num_; ++i) {
    auto worker = reinterpret_cast<ActorWorker *>(workers_[i]);
    if (worker->ActorActive()) {
      break;
    }
  }
}

int ActorThreadPool::CreateThreads(size_t actor_thread_num, size_t all_thread_num, const std::vector<int> &core_list) {
#ifdef USE_HQUEUE
  if (actor_queue_.Init(MAX_READY_ACTOR_NR) != true) {
    THREAD_ERROR("init actor queue failed.");
    return THREAD_ERROR;
  }
#endif
#ifdef BIND_CORE
  affinity_->SetCoreId(core_list);
#endif
  size_t core_num = std::thread::hardware_concurrency();
  THREAD_INFO("ThreadInfo, Actor: [%zu], All: [%zu], CoreNum: [%zu]", actor_thread_num, all_thread_num, core_num);
  actor_thread_num_ = actor_thread_num < core_num ? actor_thread_num : core_num;
  if (actor_thread_num > all_thread_num) {
    THREAD_ERROR("thread num is invalid");
    return THREAD_ERROR;
  }
  for (size_t i = 0; i < actor_thread_num_; ++i) {
    std::lock_guard<std::mutex> _l(pool_mutex_);
    auto worker = new (std::nothrow) ActorWorker();
    THREAD_ERROR_IF_NULL(worker);
#ifdef BIND_CORE
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (core_list.size() > 0) {
      CPU_SET(core_list[workers_.size() % core_list.size()], &mask);
    }
    worker->set_mask(mask);
#endif
    worker->CreateThread(this);
    workers_.push_back(worker);
    THREAD_INFO("create actor thread[%zu]", i);
  }
  size_t kernel_thread_num = all_thread_num - actor_thread_num_;
  if (kernel_thread_num > 0) {
    return ThreadPool::CreateThreads(kernel_thread_num, core_list);
  }
  return THREAD_OK;
}

ActorThreadPool *ActorThreadPool::CreateThreadPool(size_t actor_thread_num, size_t all_thread_num, BindMode bind_mode) {
  ActorThreadPool *pool = new (std::nothrow) ActorThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
  int ret;
  std::vector<int> core_list;
#ifdef BIND_CORE
  ret = pool->InitAffinityInfo();
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  core_list = pool->affinity_->GetCoreId(all_thread_num, bind_mode);
#endif  // BIND_CORE
  ret = pool->CreateThreads(actor_thread_num, all_thread_num, core_list);
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }

  return pool;
}

ActorThreadPool *ActorThreadPool::CreateThreadPool(size_t actor_thread_num, size_t all_thread_num,
                                                   const std::vector<int> &core_list) {
  ActorThreadPool *pool = new (std::nothrow) ActorThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
  int ret;
#ifdef BIND_CORE
  ret = pool->InitAffinityInfo();
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
#endif  // BIND_CORE
  ret = pool->CreateThreads(actor_thread_num, all_thread_num, core_list);
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }

  return pool;
}

ActorThreadPool *ActorThreadPool::CreateThreadPool(size_t thread_num) {
  ActorThreadPool *pool = new (std::nothrow) ActorThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
  int ret = pool->CreateThreads(thread_num, thread_num, {});
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  return pool;
}
}  // namespace mindspore
