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

#include "thread/threadpool.h"
#include <unistd.h>
#include <algorithm>
#include "thread/core_affinity.h"

namespace mindspore {

constexpr int kDefaultSpinCount = 30000;

ThreadPool::~ThreadPool() {
  alive_.store(false);
  DestructThreads();
}

void ThreadPool::DestructThreads() {
  for (auto &worker : workers_) {
    sem_post(&worker->sem);
    if (worker->thread.joinable()) {
      worker->thread.join();
    }
    sem_destroy(&worker->sem);
    sem_destroy(&worker->init);
    delete worker;
    worker = nullptr;
  }
  workers_.clear();
  if (affinity_ != nullptr) {
    delete affinity_;
    affinity_ = nullptr;
  }
  THREAD_INFO("deconstruct threads success");
}

int ThreadPool::CreateThreads(size_t thread_num) {
  size_t core_num = std::thread::hardware_concurrency();
  thread_num_ = std::min(thread_num, core_num);
  if (thread_num_ <= 0) {
    THREAD_ERROR("thread num is invalid");
    return THREAD_ERROR;
  }
  for (size_t i = 0; i < thread_num_; ++i) {
    Worker *worker = new (std::nothrow) Worker();
    THREAD_ERROR_IF_NULL(worker);
    sem_init(&worker->sem, 0, 0);
    sem_init(&worker->init, 0, 0);
    worker->type = i < inter_thread_num_ ? kActorThread : kKernelThread;
    if (worker->type == kKernelThread) {
      freelist_.push_back(worker);
    }
    worker->thread = std::thread(&ThreadPool::ThreadAsyncRun, this, worker);
    sem_wait(&worker->init);
    workers_.push_back(worker);
    THREAD_INFO("create thread[%zu]", i);
  }
  return THREAD_OK;
}

void ThreadPool::KernelThreadRun(Worker *worker) {
  if (sem_trywait(&worker->sem) == THREAD_OK) {
    Task *task = worker->task;
    if (task == nullptr) {
      return;
    }
    task->status |= task->func(task->content, ++task->task_id);
    ++task->finished;
    worker->task = nullptr;
    {
      std::lock_guard<std::mutex> _l(pool_mutex_);
      freelist_.push_back(worker);
    }
  } else {
    std::this_thread::yield();
    worker->spin++;
    if (worker->spin >= kDefaultSpinCount) {
      worker->spin = 0;
      sem_wait(&worker->sem);
      sem_post(&worker->sem);
    }
  }
}

void ThreadPool::ThreadAsyncRun(Worker *worker) {
  THREAD_RETURN_IF_NULL(worker);
  sem_post(&worker->init);
  while (alive_) {
    KernelThreadRun(worker);
  }
}

int ThreadPool::ParallelLaunch(const Func &func, Contend contend, int task_num) {
  THREAD_INFO("parallel launch, task num: %d", task_num);
  // distribute task to the KernelThread and the free ActorThread,
  // if the task num is greater than the KernelThread num
  Task task = Task(func, contend);
  DistributeTask(&task, task_num);

  task.status |= task.func(task.content, 0);
  ++task.finished;
  // synchronization
  // wait until the finished is equal to task_num
  while (task.finished != task_num) {
    std::this_thread::yield();
  }
  // check the return value of task
  if (task.status != THREAD_OK) {
    return THREAD_ERROR;
  }
  return THREAD_OK;
}

void ThreadPool::DistributeTask(Task *task, int task_num) {
  int count = 0;
  while (count < task_num - 1) {
    std::lock_guard<std::mutex> _l(pool_mutex_);
    if (!freelist_.empty()) {
      Worker *worker = freelist_.back();
      freelist_.pop_back();
      worker->task = task;
      sem_post(&worker->sem);
      count++;
    }
  }
}

int ThreadPool::InitAffinityInfo() {
  affinity_ = new (std::nothrow) CoreAffinity();
  THREAD_ERROR_IF_NULL(affinity_);
  int ret = affinity_->InitHardwareCoreInfo();
  if (ret != THREAD_OK) {
    delete affinity_;
    affinity_ = nullptr;
    return THREAD_ERROR;
  }
  return THREAD_OK;
}

int ThreadPool::SetCpuAffinity(BindMode bind_mode) {
  if (workers_.empty()) {
    return THREAD_ERROR;
  }
#ifdef BIND_CORE
  THREAD_ERROR_IF_NULL(affinity_);
  return affinity_->BindThreads(workers_, bind_mode);
#else
  return THREAD_OK;
#endif  // BIND_CORE
}

int ThreadPool::SetCpuAffinity(const std::vector<int> &core_list) {
  if (workers_.empty()) {
    return THREAD_ERROR;
  }
#ifdef BIND_CORE
  THREAD_ERROR_IF_NULL(affinity_);
  return affinity_->BindThreads(workers_, core_list);
#else
  return THREAD_OK;
#endif  // BIND_CORE
}

int ThreadPool::SetProcessAffinity(BindMode bind_mode) const {
#ifdef BIND_CORE
  THREAD_ERROR_IF_NULL(affinity_);
  return affinity_->BindProcess(bind_mode);
#else
  return THREAD_OK;
#endif  // BIND_CORE
}

ThreadPool *ThreadPool::CreateThreadPool(size_t thread_num) {
  ThreadPool *pool = new (std::nothrow) ThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
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
}  // namespace mindspore
