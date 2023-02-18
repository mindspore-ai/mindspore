/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
size_t ActorThreadPool::actor_queue_size_ = kMaxHqueueSize;

void ActorWorker::CreateThread() { thread_ = std::thread(&ActorWorker::RunWithSpin, this); }

void ActorWorker::RunWithSpin() {
  if (!core_list_.empty()) {
    SetAffinity();
  }
#if !defined(__APPLE__) && !defined(_MSC_VER)
  static std::atomic_int index{0};
  (void)pthread_setname_np(pthread_self(), ("ActorThread_" + std::to_string(index++)).c_str());
#endif
#ifdef PLATFORM_86
  // Some CPU kernels need set the flush zero mode to improve performance.
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
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
  if (pool_ == nullptr) {
    return false;
  }
  auto actor = reinterpret_cast<ActorThreadPool *>(pool_)->PopActorFromQueue();
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

int ActorThreadPool::ActorQueueInit() {
#ifdef USE_HQUEUE
  if (actor_queue_.Init(static_cast<int32_t>(actor_queue_size_)) != true) {
    THREAD_ERROR("init actor queue failed.");
    return THREAD_ERROR;
  }
#endif
  return THREAD_OK;
}

int ActorThreadPool::CreateThreads(size_t actor_thread_num, size_t all_thread_num, const std::vector<int> &core_list) {
  if (actor_thread_num > all_thread_num) {
    THREAD_ERROR("thread num is invalid");
    return THREAD_ERROR;
  }
  if (ActorQueueInit() != THREAD_OK) {
    return THREAD_ERROR;
  }
  if (affinity_ != nullptr) {
    affinity_->SetCoreId(core_list);
  }
  size_t core_num = std::thread::hardware_concurrency();
  THREAD_INFO("ThreadInfo, Actor: [%zu], All: [%zu], CoreNum: [%zu]", actor_thread_num, all_thread_num, core_num);
  actor_thread_num_ = actor_thread_num < core_num ? actor_thread_num : core_num;
  core_num -= actor_thread_num_;
  size_t kernel_thread_num =
    (all_thread_num - actor_thread_num_) < core_num ? (all_thread_num - actor_thread_num_) : core_num;
  size_t total_thread_num = actor_thread_num_ + kernel_thread_num;
  if (TaskQueuesInit(total_thread_num) != THREAD_OK) {
    return THREAD_ERROR;
  }

  if (ThreadPool::CreateThreads<ActorWorker>(actor_thread_num_, core_list) != THREAD_OK) {
    return THREAD_ERROR;
  }

  if (kernel_thread_num > 0) {
    return ThreadPool::CreateThreads<Worker>(kernel_thread_num, core_list);
  }
  return THREAD_OK;
}

ActorThreadPool *ActorThreadPool::CreateThreadPool(size_t actor_thread_num, size_t all_thread_num,
                                                   const std::vector<int> &core_list, BindMode bind_mode) {
  std::lock_guard<std::mutex> lock(create_thread_pool_muntex_);
  ActorThreadPool *pool = new (std::nothrow) ActorThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
  int ret = pool->InitAffinityInfo();
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  if (core_list.empty()) {
    ret = pool->CreateThreads(actor_thread_num, all_thread_num, pool->affinity_->GetCoreId(all_thread_num, bind_mode));
  } else {
    ret = pool->CreateThreads(actor_thread_num, all_thread_num, core_list);
  }

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
