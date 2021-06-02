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
#include <algorithm>
#include "thread/core_affinity.h"

namespace mindspore {
constexpr int kDefaultSpinCount = 300000;

float PartialScale(int partial, int total) { return (partial * 10.0 / total) / 10.0; }

ThreadPool::~ThreadPool() {
  alive_.store(false);
  DestructThreads();
}

void ThreadPool::DestructThreads() {
  for (auto &worker : workers_) {
    worker->cond_var.notify_one();
    if (worker->thread.joinable()) {
      worker->thread.join();
    }
    delete worker;
    worker = nullptr;
  }
  workers_.clear();
  delete affinity_;
  affinity_ = nullptr;
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
    worker->type = i < inter_thread_num_ ? kActorThread : kKernelThread;
    if (worker->type == kKernelThread) {
      freelist_.push_back(worker);
    }
    worker->thread = std::thread(&ThreadPool::ThreadAsyncRun, this, worker);
    workers_.push_back(worker);
    THREAD_INFO("create thread[%zu]", i);
  }
  return THREAD_OK;
}

void ThreadPool::ThreadAsyncRun(Worker *worker) {
  THREAD_RETURN_IF_NULL(worker);
  while (alive_) {
    KernelThreadRun(worker);
  }
}

void ThreadPool::KernelThreadRun(Worker *worker) {
  if (worker->active) {
    Task *task = worker->task;
    THREAD_RETURN_IF_NULL(task);
    task->status |= task->func(task->content, worker->task_id, worker->lhs_scale, worker->rhs_scale);
    {
      std::lock_guard<std::mutex> _l(worker->mutex);
      worker->task = nullptr;
      worker->active = false;
    }
    {
      std::lock_guard<std::mutex> _l(pool_mutex_);
      freelist_.push_back(worker);
    }
    ++task->finished;
  } else {
    worker->spin++;
    std::this_thread::yield();
  }
  if (worker->spin >= kDefaultSpinCount) {
    worker->spin = 0;
    std::unique_lock<std::mutex> _l(worker->mutex);
    worker->cond_var.wait(_l, [&]() { return worker->active || !alive_; });
  }
}

int ThreadPool::ParallelLaunch(const Func &func, Content content, int task_num) {
  // distribute task to the KernelThread and the free ActorThread,
  // if the task num is greater than the KernelThread num
  Task task = {func, content};
  Worker *curr = CurrentWorker();
  if (inter_thread_num_ == thread_num_ || curr == nullptr) {
    SyncRunTask(&task, task_num);
  } else {
    DistributeTask(&task, task_num);
    task.status |= task.func(task.content, 0, curr->lhs_scale, curr->rhs_scale);
    ++task.finished;
  }
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

void ThreadPool::SyncRunTask(Task *task, int task_num) const {
  float per_scale = kMaxScale / task_num;
  for (int i = 0; i < task_num; ++i) {
    float lhs_scale = i * per_scale;
    float rhs_scale = (i + 1) * per_scale;
    rhs_scale = i == task_num - 1 ? kMaxScale : rhs_scale;
    task->status |= task->func(task->content, i, lhs_scale, rhs_scale);
    ++task->finished;
  }
}

void ThreadPool::DistributeTask(Task *task, int task_num) {
  int count = 1;
  int sum_frequency = 0;
  std::vector<Worker *> assigned;
  Worker *curr = CurrentWorker();
  THREAD_RETURN_IF_NULL(curr);
  assigned.push_back(curr);
  sum_frequency += curr->frequency;

  Worker *worker;
  while (count < task_num) {
    {
      std::lock_guard<std::mutex> _l(pool_mutex_);
      if (freelist_.empty()) {
        continue;
      }
      worker = freelist_.back();
      freelist_.pop_back();
    }
    assigned.push_back(worker);
    sum_frequency += worker->frequency;
    count++;
  }

  CalculateScales(assigned, sum_frequency);
  ActiveWorkers(assigned, task, task_num);
}

void ThreadPool::CalculateScales(const std::vector<Worker *> &assigned, int sum_frequency) const {
  // Divide task according to computing power(core frequency)
  float start = 0.;
  for (const auto &worker : assigned) {
    THREAD_RETURN_IF_NULL(worker);
    worker->lhs_scale = start;
    start += PartialScale(worker->frequency, sum_frequency);
    start = start < 1 ? start : 1;
    worker->rhs_scale = start;
  }
}

void ThreadPool::ActiveWorkers(const std::vector<Worker *> &workers, Task *task, int task_num) const {
  for (int i = 1; i < task_num; ++i) {
    Worker *worker = workers[i];
    THREAD_RETURN_IF_NULL(worker);
    std::lock_guard<std::mutex> _l(worker->mutex);
    worker->task = task;
    worker->task_id = i;
    worker->active = true;
    worker->cond_var.notify_one();
  }
}

Worker *ThreadPool::CurrentWorker() const {
  for (const auto &worker : workers_) {
    if (worker->thread.get_id() == std::this_thread::get_id()) {
      return worker;
    }
  }
  return nullptr;
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
