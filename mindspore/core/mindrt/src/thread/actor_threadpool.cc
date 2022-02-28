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
void ActorWorker::CreateThread() { thread_ = std::thread(&ActorWorker::RunWithSpin, this); }

void ActorWorker::RunWithSpin() {
  SetAffinity();
#if !defined(__APPLE__) && !defined(SUPPORT_MSVC)
  static std::atomic_int index = {0};
  (void)pthread_setname_np(pthread_self(), ("ActorThread_" + std::to_string(index++)).c_str());
#endif
#ifdef PLATFORM_86
  // Some CPU kernels need set the flush zero mode to improve performance.
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  while (alive_) {
    // only run either local KernelTask or PoolQueue ActorTask
    if (RunLocalKernelTask()) {
      spin_count_ = 0;
    } else {
      YieldAndDeactive();
    }
#ifdef OPERATOR_PARALLELISM
    if (RunQueueActorTask() || RunQueueWorkTask()) {
#else
    if (RunQueueActorTask()) {
#endif
      if (spin_count_ > 0) {
        spin_count_ = 1;
      }
    }
    if (spin_count_ > max_spin_count_) {
      WaitUntilActive();
      spin_count_ = 1;
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
#ifndef OPERATOR_PARALLELISM
  if (available() || check_task_nullptr()) {
    status_ = kThreadBusy;
    set_task_free(true);
  } else {
    set_task_free(false);
  }
#endif
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
#ifdef OPERATOR_PARALLELISM
  if (task_queue_.Init(MAX_READY_TASK_NR) != true) {
    THREAD_ERROR("Init task queue failed");
    return THREAD_ERROR;
  }
#endif
#endif
  if (affinity_ != nullptr) {
    affinity_->SetCoreId(core_list);
  }
  size_t core_num = std::thread::hardware_concurrency();
  THREAD_INFO("ThreadInfo, Actor: [%zu], All: [%zu], CoreNum: [%zu]", actor_thread_num, all_thread_num, core_num);
  actor_thread_num_ = actor_thread_num < core_num ? actor_thread_num : core_num;
  if (actor_thread_num > all_thread_num) {
    THREAD_ERROR("thread num is invalid");
    return THREAD_ERROR;
  }
  for (size_t i = 0; i < actor_thread_num_; ++i) {
    std::lock_guard<std::mutex> _l(pool_mutex_);
    auto worker = new (std::nothrow) ActorWorker(this);
    THREAD_ERROR_IF_NULL(worker);
#ifdef OPERATOR_PARALLELISM
    auto task_messages = reinterpret_cast<TaskMessage *>(malloc(sizeof(TaskMessage) * all_thread_num));
    if (task_messages == nullptr) {
      delete worker;
      THREAD_ERROR("malloc TaskMessages failed.");
      return THREAD_ERROR;
    }
    for (size_t j = 0; j < all_thread_num; j++) {
      task_messages[j].task_id = j;
    }
    worker->SetTaskMessages(task_messages);
#endif
    worker->InitWorkerMask(core_list, workers_.size());
    worker->CreateThread();
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

  auto ret = pool->InitAffinityInfo();
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  auto core_list = pool->affinity_->GetCoreId(all_thread_num, bind_mode);

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
  int ret = pool->InitAffinityInfo();
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
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
