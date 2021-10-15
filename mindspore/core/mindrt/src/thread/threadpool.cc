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
#include "thread/threadpool.h"
#include "thread/core_affinity.h"

namespace mindspore {
Worker::~Worker() {
  {
    std::lock_guard<std::mutex> _l(mutex_);
    alive_ = false;
  }
  cond_var_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Worker::CreateThread() { thread_ = std::thread(&Worker::Run, this); }

void Worker::SetAffinity() {
#ifdef BIND_CORE
#ifdef __ANDROID__
  int ret = sched_setaffinity(gettid(), sizeof(cpu_set_t), &mask_);
  if (ret != THREAD_OK) {
    THREAD_ERROR("bind thread %d to cpu failed. ERROR %d", gettid(), errno);
  }
  return;
#else
#if !defined(__APPLE__) && !defined(SUPPORT_MSVC)
  int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask_);
  if (ret != THREAD_OK) {
    THREAD_ERROR("bind thread %lu to cpu failed. ERROR %d", pthread_self(), errno);
  }
  return;
#endif
#endif
#endif
}

void Worker::Run() {
  SetAffinity();
#if !defined(__APPLE__) && !defined(SUPPORT_MSVC)
  static std::atomic_int index = {0};
  (void)pthread_setname_np(pthread_self(), ("KernelThread_" + std::to_string(index++)).c_str());
#endif
  while (alive_) {
    if (RunLocalKernelTask()) {
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

bool Worker::RunLocalKernelTask() {
  Task *task = task_.load(std::memory_order_consume);
  if (task == nullptr) {
    return false;
  }
  int task_id = task_id_.load(std::memory_order_consume);
  task->status |= task->func(task->content, task_id, lhs_scale_, rhs_scale_);
  task_.store(nullptr, std::memory_order_relaxed);
  (void)++task->finished;
  return true;
}

void Worker::YieldAndDeactive() {
  // deactivate this worker only on the first entry
  if (spin_count_ == 0) {
    status_.store(kThreadIdle);
  }
  spin_count_++;
  std::this_thread::yield();
}

void Worker::WaitUntilActive() {
  std::unique_lock<std::mutex> _l(mutex_);
  cond_var_.wait(_l, [&] { return status_ == kThreadBusy || active_num_ > 0 || !alive_; });
  active_num_--;
}

void Worker::set_scale(float lhs_scale, float rhs_scale) {
  lhs_scale_ = lhs_scale;
  rhs_scale_ = rhs_scale;
}

void Worker::Active(Task *task, int task_id) {
  {
    std::lock_guard<std::mutex> _l(mutex_);
    task_id_.store(task_id, std::memory_order_relaxed);
    task_.store(task, std::memory_order_relaxed);
    status_ = kThreadBusy;
  }
  cond_var_.notify_one();
}

void Worker::Active() {
  {
    std::lock_guard<std::mutex> _l(mutex_);
    active_num_++;
    status_ = kThreadBusy;
  }
  cond_var_.notify_one();
}

bool Worker::available() {
  int expected = kThreadIdle;
  return status_.compare_exchange_strong(expected, kThreadHeld);
}

ThreadPool::~ThreadPool() {
  for (auto &worker : workers_) {
    delete worker;
    worker = nullptr;
  }
  workers_.clear();
  delete affinity_;
  affinity_ = nullptr;
  THREAD_INFO("destruct success");
}

int ThreadPool::CreateThreads(size_t thread_num, const std::vector<int> &core_list) {
  size_t core_num = std::thread::hardware_concurrency();
  thread_num = thread_num < core_num ? thread_num : core_num;
  THREAD_INFO("ThreadInfo, Num: [%zu], CoreNum: [%zu]", thread_num, core_num);
  if (thread_num == 0) {
    THREAD_INFO("Current thread as working thread.");
    return THREAD_OK;
  }
  std::lock_guard<std::mutex> _l(pool_mutex_);
  for (size_t i = 0; i < thread_num; ++i) {
    auto worker = new (std::nothrow) Worker();
    THREAD_ERROR_IF_NULL(worker);
#ifdef BIND_CORE
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (core_list.size() > 0) {
      CPU_SET(core_list[workers_.size() % core_list.size()], &mask);
    }
    worker->set_mask(mask);
#endif
    worker->CreateThread();
    workers_.push_back(worker);
    THREAD_INFO("create kernel thread[%zu]", i);
  }
  return THREAD_OK;
}

int ThreadPool::ParallelLaunch(const Func &func, Content content, int task_num) const {
  // if single thread, run master thread
  if (thread_num() <= 1 || task_num <= 1) {
    for (int i = 0; i < task_num; ++i) {
      int ret = func(content, i, 0, 1);
      if (ret != 0) {
        return ret;
      }
    }
    return THREAD_OK;
  }

  // distribute task to the KernelThread and the idle ActorThread,
  // if the task num is greater than the KernelThread num
  THREAD_DEBUG("launch: %d", task_num);
  Task task = {func, content};

  DistributeTask(&task, task_num);
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

void ThreadPool::SyncRunTask(Task *task, int start_num, int task_num) const {
  // run task sequentially
  // if the current thread is not the actor thread
  float per_scale = kMaxScale / (task_num - start_num);
  for (int i = start_num; i < task_num; ++i) {
    float lhs_scale = i * per_scale;
    float rhs_scale = (i + 1) * per_scale;
    rhs_scale = i == task_num - 1 ? kMaxScale : rhs_scale;
    task->status |= task->func(task->content, i, lhs_scale, rhs_scale);
    (void)++task->finished;
  }
}

void ThreadPool::DistributeTask(Task *task, int task_num) const {
  Worker *curr = CurrentWorker();
  // if the current thread isn't nullptr, that is the curr is a ActorThread,
  // then assign (task_num - 1) tasks to workers, and run the last one by itself
  int count = 0;
  int num_assigned = curr != nullptr ? task_num - 1 : task_num;
  int sum_frequency = 0;
  std::vector<Worker *> assigned;
  int num = static_cast<int>(workers_.size()) - 1;
  int offset = 0;
  if (!occupied_actor_thread_) {
    offset = static_cast<int>(actor_thread_num_);
  }
  for (int i = num; i >= offset && count < num_assigned; --i) {
    if (workers_[i]->available()) {
      assigned.push_back(workers_[i]);
      sum_frequency += workers_[i]->frequency();
      count++;
    }
  }
  // when there are not enough free threads,
  // distribute other tasks to the master thread
  if (curr != nullptr) {
    for (; count < task_num; ++count) {
      assigned.push_back(curr);
      sum_frequency += curr->frequency();
    }
  } else if (assigned.size() != static_cast<size_t>(task_num)) {
    CalculateScales(assigned, sum_frequency);
    ActiveWorkers(assigned, task, assigned.size(), curr);
    SyncRunTask(task, assigned.size(), task_num);
    return;
  }
  CalculateScales(assigned, sum_frequency);
  ActiveWorkers(assigned, task, task_num, curr);
}

void ThreadPool::CalculateScales(const std::vector<Worker *> &assigned, int sum_frequency) const {
  // divide task according to computing power(core frequency)
  float lhs_scale = 0;
  float rhs_scale = 0;
  if (sum_frequency == 0) {
    return;
  }
  for (const auto &worker : assigned) {
    THREAD_RETURN_IF_NULL(worker);
    rhs_scale += worker->frequency() * 1.0 / sum_frequency;
    rhs_scale = rhs_scale < 1 ? rhs_scale : 1;
    worker->set_scale(lhs_scale, rhs_scale);
    lhs_scale = rhs_scale;
  }
}

void ThreadPool::ActiveWorkers(const std::vector<Worker *> &workers, Task *task, int task_num,
                               const Worker *curr) const {
  for (int i = 0; i < task_num; ++i) {
    Worker *worker = workers[i];
    THREAD_RETURN_IF_NULL(worker);
    worker->Active(task, i);
    if (worker == curr) {
      (void)worker->RunLocalKernelTask();
    }
  }
}

void ThreadPool::ActiveWorkers() const {
  for (auto &worker : workers_) {
    worker->Active();
  }
}

Worker *ThreadPool::CurrentWorker() const {
  for (const auto &worker : workers_) {
    if (worker->thread_id() == std::this_thread::get_id()) {
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

void ThreadPool::SetSpinCountMaxValue() {
  for (auto worker : workers_) {
    THREAD_RETURN_IF_NULL(worker);
    worker->SetMaxSpinCount(max_spin_count_);
  }
  return;
}

void ThreadPool::SetSpinCountMinValue() {
  for (auto worker : workers_) {
    THREAD_RETURN_IF_NULL(worker);
    worker->SetMaxSpinCount(min_spin_count_);
  }
  return;
}

void ThreadPool::SetMaxSpinCount(int spin_count) {
  if (spin_count <= 0) {
    return;
  }
  max_spin_count_ = spin_count;
}

void ThreadPool::SetMinSpinCount(int spin_count) {
  if (spin_count <= 0) {
    return;
  }
  min_spin_count_ = spin_count;
}

ThreadPool *ThreadPool::CreateThreadPool(size_t thread_num, const std::vector<int> &core_list) {
  ThreadPool *pool = new (std::nothrow) ThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
  int ret = pool->CreateThreads(thread_num, core_list);
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
