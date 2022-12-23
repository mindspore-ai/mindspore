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
#include "thread/parallel_threadpool.h"

namespace mindspore {
namespace {
const char *kInnerIDs = "inner_ids";
const char *kInnerRunnerID = "inner_runner_id";
const char *kInnerModelID = "inner_model_id";
}  // namespace
ParallelThreadPoolManager *ParallelThreadPoolManager::GetInstance() {
  static ParallelThreadPoolManager instance;
  return &instance;
}

void ParallelThreadPoolManager::Init(bool enable_shared_thread_pool, const std::string &runner_id, int worker_num,
                                     int remaining_thread_num, int thread_num_limit) {
#ifndef MS_COMPILE_IOS
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  enable_shared_thread_pool_ = enable_shared_thread_pool;
  if (!enable_shared_thread_pool) {
    THREAD_INFO("not enable shared parallel thread pool.");
    return;
  }
  std::vector<ParallelThreadPool *> runner_pools(worker_num, nullptr);
  runner_id_pools_[runner_id] = runner_pools;
  remaining_thread_num_ = remaining_thread_num;
  thread_num_limit_ = thread_num_limit;
#endif
}

void ParallelThreadPoolManager::SetHasIdlePool(bool is_idle) {
#ifndef MS_COMPILE_IOS
  has_idle_pool_ = is_idle;
#endif
}

int ParallelThreadPoolManager::GetTaskNum() {
#ifndef MS_COMPILE_IOS
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  if (!enable_shared_thread_pool_) {
    THREAD_INFO("not enable shared parallel thread pool.");
    return -1;
  }
  return thread_num_limit_;
#endif
  return 0;
}

int ParallelThreadPoolManager::GetThreadPoolSize() {
#ifndef MS_COMPILE_IOS
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  return pool_workers_.begin()->second.size();
#endif
  return 0;
}

void ParallelThreadPoolManager::BindPoolToRunner(
#ifndef MS_COMPILE_IOS
  ThreadPool *pool, const std::map<std::string, std::map<std::string, std::string>> *config_info) {
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  if (!enable_shared_thread_pool_ || config_info == nullptr) {
    THREAD_ERROR("not use parallel thread pool shared.");
    return;
  }
  std::string runner_id;
  auto it_id = config_info->find(kInnerIDs);
  if (it_id != config_info->end()) {
    auto item_runner = it_id->second.find(kInnerRunnerID);
    if (item_runner != it_id->second.end()) {
      runner_id = it_id->second.at(kInnerRunnerID);
    }
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
  parallel_pool->SetRunnerID(runner_id);
  auto all_workers = parallel_pool->GetParallelPoolWorkers();
  for (size_t i = 0; i < all_workers.size(); i++) {
    auto worker = static_cast<ParallelWorker *>(all_workers[i]);
    pool_workers_[parallel_pool].push_back(worker);
  }
#endif
}

bool ParallelThreadPoolManager::GetEnableSharedThreadPool() {
#ifndef MS_COMPILE_IOS
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  return enable_shared_thread_pool_;
#endif
  return false;
}

void ParallelThreadPoolManager::ActivatePool(const std::string &runner_id, int model_id) {
#ifndef MS_COMPILE_IOS
  std::shared_lock<std::shared_mutex> l(pool_manager_mutex_);
  auto &pool = runner_id_pools_[runner_id][model_id];
  pool->UseThreadPool(1);
  auto &workers = pool_workers_[pool];
  for (auto &worker : workers) {
    worker->ActivateByOtherPoolTask();
  }
#endif
}

void ParallelThreadPoolManager::SetFreePool(const std::string &runner_id, int model_id) {
#ifndef MS_COMPILE_IOS
  std::shared_lock<std::shared_mutex> l(pool_manager_mutex_);
  auto &pool = runner_id_pools_[runner_id][model_id];
  pool->UseThreadPool(-1);
#endif
}

ParallelThreadPool *ParallelThreadPoolManager::GetIdleThreadPool(const std::string &runner_id, ParallelTask *task) {
#ifndef MS_COMPILE_IOS
  if (!has_idle_pool_) {
    return nullptr;
  }
  std::shared_lock<std::shared_mutex> l(pool_manager_mutex_);
  auto &all_pools = runner_id_pools_[runner_id];
  for (int pool_index = all_pools.size() - 1; pool_index >= 0; pool_index--) {
    auto &pool = all_pools[pool_index];
    if (pool->IsIdlePool()) {
      auto &workers = pool_workers_[pool];
      for (size_t i = 0; i < workers.size() - remaining_thread_num_; i++) {
        workers[i]->ActivateByOtherPoolTask(task);
      }
      return pool;
    }
  }
#endif
  return nullptr;
}

void ParallelThreadPoolManager::ResetParallelThreadPoolManager(const std::string &runner_id) {
#ifndef MS_COMPILE_IOS
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  if (runner_id_pools_.find(runner_id) == runner_id_pools_.end()) {
    return;
  }
  auto pools = runner_id_pools_[runner_id];
  for (auto &pool : pools) {
    pool_workers_.erase(pool);
  }
  runner_id_pools_.erase(runner_id);
#endif
}

ParallelThreadPoolManager::~ParallelThreadPoolManager() {
#ifndef MS_COMPILE_IOS
  THREAD_INFO("~ParallelThreadPoolManager start.");
  std::unique_lock<std::shared_mutex> l(pool_manager_mutex_);
  pool_workers_.clear();
  runner_id_pools_.clear();
  THREAD_INFO("~ParallelThreadPoolManager end.");
#endif
}
}  // namespace mindspore
