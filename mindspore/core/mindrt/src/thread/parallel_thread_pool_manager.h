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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_PARALLEL_THREAD_POOL_MANAGER_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_PARALLEL_THREAD_POOL_MANAGER_H_

#if defined(PARALLEL_INFERENCE) && defined(ENABLE_MINDRT)
#define THREAD_POOL_MANAGER
#endif

#include <queue>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <condition_variable>
#include <map>
#include <string>

namespace mindspore {
class ThreadPool;
// class Worker;
#ifdef ENABLE_MINDRT
class ParallelThreadPool;
struct ParallelTask;
class ParallelWorker;
#endif

class ParallelThreadPoolManager {
 public:
  static ParallelThreadPoolManager *GetInstance();

  ~ParallelThreadPoolManager();

  void Init(bool enable_shared_thread_pool, const std::string &runner_id, int worker_num, int remaining_thread_num,
            int task_num);

  void SetHasIdlePool(std::string runner_id, bool is_idle);

  void ResetParallelThreadPoolManager(const std::string &runner_id);

  bool GetEnableSharedThreadPool(std::string runner_id);

  void ActivatePool(const std::string &runner_id, int model_id);

  void SetFreePool(const std::string &runner_id, int model_id);

  void BindPoolToRunner(ThreadPool *pool, const std::map<std::string, std::map<std::string, std::string>> *config_info);

  int GetThreadPoolSize(ThreadPool *pool);

  int GetTaskNum(const std::map<std::string, std::map<std::string, std::string>> *config_info);

#ifdef ENABLE_MINDRT
  ParallelThreadPool *GetIdleThreadPool(const std::string &runner_id, ParallelTask *task);
#endif

 private:
  ParallelThreadPoolManager() = default;

 private:
#ifdef THREAD_POOL_MANAGER
  // runner id <=> thread pool(a model has a thread pool)
  std::map<std::string, std::vector<ParallelThreadPool *>> runner_id_pools_;
  // pool sorted by model worker id
  std::unordered_map<ParallelThreadPool *, std::vector<ParallelWorker *>> pool_workers_;

  std::shared_mutex pool_manager_mutex_;
  std::map<std::string, bool> has_idle_pool_;
  std::map<std::string, bool> enable_shared_thread_pool_;
  std::map<std::string, int> runner_worker_num_;
  std::map<std::string, int> worker_init_num_;
  std::map<std::string, int> idle_pool_num_;
  std::map<std::string, int> remaining_thread_num_;
  std::map<std::string, int> thread_num_limit_;
#endif
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_PARALLEL_THREAD_POOL_MANAGER_H_
