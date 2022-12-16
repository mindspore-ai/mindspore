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

#ifndef MINDSPORE_LITE_SRC_PARALLEL_THREAD_POOL_MANAGER_H_
#define MINDSPORE_LITE_SRC_PARALLEL_THREAD_POOL_MANAGER_H_

#include <queue>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <condition_variable>
#include <map>
#include <string>

namespace mindspore {
class ParallelThreadPool;
class ThreadPool;
struct ParallelTask;
class ParallelWorker;
class Worker;
class ParallelThreadPoolManager {
 public:
  static ParallelThreadPoolManager *GetInstance();

  ~ParallelThreadPoolManager();

  void Init(bool enable_shared_thread_pool, const std::string &runner_id, int worker_num, int remaining_thread_num);

  void SetHasIdlePool(bool is_idle);

  ParallelThreadPool *GetIdleThreadPool(const std::string &runner_id, ParallelTask *task);

  void ResetParallelThreadPoolManager(const std::string &runner_id);

  bool GetEnableSharedThreadPool();

  void ActivatePool(const std::string &runner_id, int model_id);

  void SetFreePool(const std::string &runner_id, int model_id);

  void BindPoolToRunner(ThreadPool *pool, const std::map<std::string, std::map<std::string, std::string>> *config_info);

  int GetThreadPoolSize();

 private:
  ParallelThreadPoolManager() = default;

 private:
  // runner id <=> thread pool(a model has a thread pool)
  std::map<std::string, std::vector<ParallelThreadPool *>> runner_id_pools_;
  // pool sorted by model worker id
  std::unordered_map<ParallelThreadPool *, std::vector<ParallelWorker *>> pool_workers_;

  std::shared_mutex pool_manager_mutex_;
  bool has_idle_pool_ = true;
  bool enable_shared_thread_pool_ = false;
  int remaining_thread_num_ = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PARALLEL_THREAD_POOL_MANAGER_H_
