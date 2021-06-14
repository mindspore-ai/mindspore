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
ThreadPool::~ThreadPool() {
  alive_ = false;
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
  thread_num = std::min(thread_num, core_num);
  THREAD_INFO("ThreadInfo, ThreadNum: [%zu], CoreNum: [%zu]", thread_num, core_num);
  if (thread_num <= 0) {
    THREAD_ERROR("thread num is invalid");
    return THREAD_ERROR;
  }
  std::lock_guard<std::mutex> _l(pool_mutex_);
  for (size_t i = 0; i < thread_num; ++i) {
    auto worker = new (std::nothrow) Worker();
    THREAD_ERROR_IF_NULL(worker);
    worker->thread = std::thread(&ThreadPool::AsyncRunTask, this, worker);
    workers_.push_back(worker);
    THREAD_INFO("create kernel thread[%zu]", i);
  }
  return THREAD_OK;
}

void ThreadPool::AsyncRunTask(Worker *worker) const {
  THREAD_RETURN_IF_NULL(worker);
  while (alive_) {
    if (RunLocalKernelTask(worker)) {
      worker->spin = 0;
    } else {
      YieldAndDeactive(worker);
    }
    if (worker->spin >= kDefaultSpinCount) {
      // wait until distribute KernelTask
      std::unique_lock<std::mutex> _l(worker->mutex);
      worker->spin = 0;
      worker->cond_var.wait(_l, [&] { return worker->running || !alive_; });
    }
  }
}

void ThreadPool::YieldAndDeactive(Worker *worker) const {
  // deactivate this worker only on the first entry
  if (worker->spin == 0) {
    worker->running = false;
  }
  worker->spin++;
  std::this_thread::yield();
}

bool ThreadPool::RunLocalKernelTask(Worker *worker) const {
  if (!worker->running || worker->task == nullptr) {
    return false;
  }
  Task *task = worker->task.load(std::memory_order_consume);
  int task_id = worker->task_id.load(std::memory_order_consume);
  task->status |= task->func(task->content, task_id, worker->lhs_scale, worker->rhs_scale);
  worker->task.store(nullptr, std::memory_order_relaxed);
  ++task->finished;
  return true;
}

int ThreadPool::ParallelLaunch(const Func &func, Content content, int task_num) const {
  // distribute task to the KernelThread and the free ActorThread,
  // if the task num is greater than the KernelThread num
  THREAD_INFO("launch: %d", task_num);
  Task task = Task(func, content);
  Worker *curr = CurrentWorker();
  if (curr == nullptr) {
    SyncRunTask(&task, task_num);
  } else {
    DistributeTask(&task, task_num);
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
  // run task sequentially
  // if the current thread is not the actor thread
  float per_scale = kMaxScale / task_num;
  for (int i = 0; i < task_num; ++i) {
    float lhs_scale = i * per_scale;
    float rhs_scale = (i + 1) * per_scale;
    rhs_scale = i == task_num - 1 ? kMaxScale : rhs_scale;
    task->status |= task->func(task->content, i, lhs_scale, rhs_scale);
    ++task->finished;
  }
}

void ThreadPool::DistributeTask(Task *task, int task_num) const {
  Worker *curr = CurrentWorker();
  THREAD_RETURN_IF_NULL(curr);

  int count = 1;
  int sum_frequency = 0;
  std::vector<Worker *> assigned;
  int num = static_cast<int>(workers_.size()) - 1;
  for (int i = num; i >= 0 && count < task_num; --i) {
    bool expected = false;
    if (workers_[i]->running.compare_exchange_strong(expected, true)) {
      assigned.push_back(workers_[i]);
      sum_frequency += workers_[i]->frequency;
      count++;
    }
  }
  assigned.push_back(curr);
  for (; count < task_num; ++count) {
    assigned.push_back(curr);
    sum_frequency += curr->frequency;
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
    start += worker->frequency * 1.0 / sum_frequency;
    start = start < 1 ? start : 1;
    worker->rhs_scale = start;
  }
}

void ThreadPool::ActiveWorkers(const std::vector<Worker *> &workers, Task *task, int task_num) const {
  Worker *curr = workers.back();
  for (int i = 0; i < task_num; ++i) {
    Worker *worker = workers[i];
    THREAD_RETURN_IF_NULL(worker);
    worker->task_id.store(i, std::memory_order_relaxed);
    worker->task.store(task, std::memory_order_relaxed);
    worker->cond_var.notify_one();
    if (worker == curr) {
      RunLocalKernelTask(worker);
    }
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
