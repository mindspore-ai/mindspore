/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "thread/parallel_thread_pool_manager.h"
#include <map>
#include <string>
#ifdef THREAD_POOL_MANAGER
#include "thread/parallel_threadpool.h"
#endif

namespace mindspore {
#ifdef THREAD_POOL_MANAGER
namespace {
const char *kInnerModelParallelRunner = "inner_model_parallel_runner";
const char *kInnerRunnerID = "inner_runner_id";
const char *kInnerModelID = "inner_model_id";
}  // namespace
#endif
ParallelThreadPoolManager *ParallelThreadPoolManager::GetInstance() {
  static ParallelThreadPoolManager instance;
  return &instance;
}

void ParallelThreadPoolManager::Init(bool enable_shared_thread_pool, const std::string &runner_id, int worker_num,
                                     int remaining_thread_num, int thread_num_limit) {
#ifdef THREAD_POOL_MANAGER
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  if (enable_shared_thread_pool_.find(runner_id) != enable_shared_thread_pool_.end()) {
    THREAD_ERROR("Not need to repeat init.");
    return;
  }
  enable_shared_thread_pool_[runner_id] = enable_shared_thread_pool;
  if (!enable_shared_thread_pool) {
    THREAD_INFO("not enable shared parallel thread pool.");
    return;
  }
  std::vector<ParallelThreadPool *> runner_pools(worker_num, nullptr);
  runner_id_pools_[runner_id] = runner_pools;
  remaining_thread_num_[runner_id] = remaining_thread_num;
  thread_num_limit_[runner_id] = thread_num_limit;
  idle_pool_num_[runner_id] = worker_num;
  runner_worker_num_[runner_id] = worker_num;
  worker_init_num_[runner_id] = 0;
#endif
}

void ParallelThreadPoolManager::SetHasIdlePool(std::string runner_id, bool is_idle) {
#ifdef THREAD_POOL_MANAGER
  has_idle_pool_[runner_id] = is_idle;
#endif
}

int ParallelThreadPoolManager::GetTaskNum(
  const std::map<std::string, std::map<std::string, std::string>> *config_info) {
#ifdef THREAD_POOL_MANAGER
  if (config_info == nullptr) {
    THREAD_ERROR("config_info is nullptr.");
    return -1;
  }
  std::string runner_id;
  auto it_id = config_info->find(kInnerModelParallelRunner);
  if (it_id != config_info->end()) {
    auto item_runner = it_id->second.find(kInnerRunnerID);
    if (item_runner != it_id->second.end()) {
      runner_id = it_id->second.at(kInnerRunnerID);
    }
  }
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  if (runner_id.empty() || !enable_shared_thread_pool_[runner_id]) {
    THREAD_INFO("not enable shared parallel thread pool.");
    return -1;
  }
  return thread_num_limit_[runner_id];
#endif
  return -1;
}

int ParallelThreadPoolManager::GetThreadPoolSize(ThreadPool *pool) {
#ifdef THREAD_POOL_MANAGER
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  ParallelThreadPool *thread_pool = static_cast<ParallelThreadPool *>(pool);
  if (thread_pool == nullptr) {
    return -1;
  }
  if (pool_workers_.find(thread_pool) != pool_workers_.end()) {
    return pool_workers_[thread_pool].size();
  } else {
    return -1;
  }
#endif
  return -1;
}

void ParallelThreadPoolManager::BindPoolToRunner(
  ThreadPool *pool, const std::map<std::string, std::map<std::string, std::string>> *config_info) {
#ifdef THREAD_POOL_MANAGER
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  if (config_info == nullptr) {
    THREAD_ERROR("config_info is nullptr.");
    return;
  }
  std::string runner_id;
  auto it_id = config_info->find(kInnerModelParallelRunner);
  if (it_id != config_info->end()) {
    auto item_runner = it_id->second.find(kInnerRunnerID);
    if (item_runner != it_id->second.end()) {
      runner_id = it_id->second.at(kInnerRunnerID);
    }
  }
  if (!enable_shared_thread_pool_[runner_id]) {
    THREAD_ERROR("not use parallel thread pool shared.");
    return;
  }
  auto parallel_pool = static_cast<ParallelThreadPool *>(pool);
  if (parallel_pool == nullptr) {
    THREAD_ERROR("parallel pool is nullptr.");
  }
  int model_id = 0;
  auto item_runner = it_id->second.find(kInnerModelID);
  if (item_runner != it_id->second.end()) {
    model_id = std::atoi(it_id->second.at(kInnerModelID).c_str());
  }
  runner_id_pools_[runner_id].at(model_id) = parallel_pool;
  auto all_workers = parallel_pool->GetParallelPoolWorkers();
  for (size_t i = 0; i < all_workers.size(); i++) {
    auto worker = static_cast<ParallelWorker *>(all_workers[i]);
    pool_workers_[parallel_pool].push_back(worker);
  }
  worker_init_num_[runner_id]++;
#endif
}

bool ParallelThreadPoolManager::GetEnableSharedThreadPool(std::string runner_id) {
#ifdef THREAD_POOL_MANAGER
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  return enable_shared_thread_pool_[runner_id];
#endif
  return false;
}

void ParallelThreadPoolManager::ActivatePool(const std::string &runner_id, int model_id) {
#ifdef THREAD_POOL_MANAGER
  std::shared_lock<std::shared_mutex> l(pool_manager_mutex_);
  if (!enable_shared_thread_pool_[runner_id]) {
    return;
  }
  auto &pool = runner_id_pools_[runner_id][model_id];
  idle_pool_num_[runner_id]--;
  pool->UseThreadPool(1);
  auto &workers = pool_workers_[pool];
  for (auto &worker : workers) {
    worker->ActivateByOtherPoolTask();
  }
#endif
}

void ParallelThreadPoolManager::SetFreePool(const std::string &runner_id, int model_id) {
#ifdef THREAD_POOL_MANAGER
  std::shared_lock<std::shared_mutex> l(pool_manager_mutex_);
  if (!enable_shared_thread_pool_[runner_id]) {
    return;
  }
  auto &pool = runner_id_pools_[runner_id][model_id];
  pool->UseThreadPool(-1);
  idle_pool_num_[runner_id]++;
#endif
}

#ifdef ENABLE_MINDRT
ParallelThreadPool *ParallelThreadPoolManager::GetIdleThreadPool(const std::string &runner_id, ParallelTask *task) {
#ifdef THREAD_POOL_MANAGER
  if (runner_worker_num_[runner_id] != worker_init_num_[runner_id] || idle_pool_num_[runner_id] <= 0) {
    return nullptr;
  }
  std::shared_lock<std::shared_mutex> l(pool_manager_mutex_);
  auto &all_pools = runner_id_pools_[runner_id];
  for (int pool_index = all_pools.size() - 1; pool_index >= 0; pool_index--) {
    auto &pool = all_pools[pool_index];
    if (pool->IsIdlePool()) {
      auto &workers = pool_workers_[pool];
      for (size_t i = 0; i < workers.size() - remaining_thread_num_[runner_id]; i++) {
        workers[i]->ActivateByOtherPoolTask(task);
      }
      return pool;
    }
  }
#endif
  return nullptr;
}
#endif

void ParallelThreadPoolManager::ResetParallelThreadPoolManager(const std::string &runner_id) {
#ifdef THREAD_POOL_MANAGER
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  if (runner_id_pools_.find(runner_id) == runner_id_pools_.end()) {
    return;
  }
  auto pools = runner_id_pools_[runner_id];
  for (auto &pool : pools) {
    pool_workers_.erase(pool);
  }
  runner_id_pools_.erase(runner_id);
  has_idle_pool_.erase(runner_id);
  enable_shared_thread_pool_.erase(runner_id);
  remaining_thread_num_.erase(runner_id);
  thread_num_limit_.erase(runner_id);
  runner_worker_num_.erase(runner_id);
  worker_init_num_.erase(runner_id);
  idle_pool_num_.erase(runner_id);
#endif
}

ParallelThreadPoolManager::~ParallelThreadPoolManager() {
#ifdef THREAD_POOL_MANAGER
  THREAD_INFO("~ParallelThreadPoolManager start.");
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  pool_workers_.clear();
  runner_id_pools_.clear();
  has_idle_pool_.clear();
  enable_shared_thread_pool_.clear();
  remaining_thread_num_.clear();
  thread_num_limit_.clear();
  runner_worker_num_.clear();
  worker_init_num_.clear();
  idle_pool_num_.clear();
  THREAD_INFO("~ParallelThreadPoolManager end.");
#endif
}
}  // namespace mindspore
