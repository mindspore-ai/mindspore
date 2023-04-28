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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_MODEL_POOL_RESOURCE_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_MODEL_POOL_RESOURCE_MANAGER_H_
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <mutex>
#include "src/common/log_adapter.h"
#include "include/api/status.h"
#include "src/extendrt/cxx_api/model_pool/model_worker.h"
namespace mindspore {
class ResourceManager {
 public:
  static ResourceManager *GetInstance();
  ~ResourceManager() = default;

  std::vector<int> ParseCpuCoreList(size_t *percentage);
  Status DistinguishPhysicalAndLogical(std::vector<int> *physical_cores, std::vector<int> *logical_cores);
  Status DistinguishPhysicalAndLogicalByNuma(std::vector<std::vector<int>> *numa_physical_cores,
                                             std::vector<std::vector<int>> *numa_logical_cores);
  std::string GenRunnerID();

 private:
  ResourceManager() = default;
  std::mutex manager_mutex_;
  size_t can_use_core_num_ = 0;
  int core_num_ = 0;
  bool can_use_all_resource_ = true;
  std::vector<int> cpu_cores_;
  std::vector<int> physical_core_ids_ = {};
  std::vector<int> logical_core_ids_ = {};
  std::vector<std::vector<int>> numa_physical_core_ids_;
  std::vector<std::vector<int>> numa_logical_core_ids_;
  size_t runner_id_ = 1;
};

class InitWorkerThread {
 public:
  InitWorkerThread() = default;
  ~InitWorkerThread();

  void CreateInitThread() { thread_ = std::thread(&InitWorkerThread::Run, this); }

  void Run();

  void Launch(std::shared_ptr<ModelWorker> worker, const char *model_buf, size_t size,
              const std::shared_ptr<WorkerConfig> &worker_config,
              const std::shared_ptr<PredictTaskQueue> &predict_task_queue, bool *create_success, ModelType model_type);

  bool IsIdle() { return is_idle_; }

  void Destroy();

 private:
  std::shared_ptr<ModelWorker> model_worker_;
  const char *model_buf_;
  size_t size_;
  std::shared_ptr<WorkerConfig> worker_config_;
  std::shared_ptr<PredictTaskQueue> predict_task_queue_;
  bool *create_success_;
  ModelType model_type_;

  bool is_destroy_ = false;
  bool is_idle_ = true;
  bool is_launch_ = false;
  std::condition_variable init_cond_var_;
  std::mutex mtx_init_;
  std::thread thread_;
};

class InitWorkerManager {
 public:
  static InitWorkerManager *GetInstance();
  ~InitWorkerManager();

  void InitModelWorker(std::shared_ptr<ModelWorker> worker, const char *model_buf, size_t size,
                       const std::shared_ptr<WorkerConfig> &worker_config,
                       const std::shared_ptr<PredictTaskQueue> &predict_task_queue, bool *create_success,
                       ModelType model_type);

 private:
  InitWorkerManager() = default;
  std::mutex manager_mutex_;
  // numa id <=> reuse worker init thread
  std::unordered_map<int, std::vector<std::shared_ptr<InitWorkerThread>>> all_init_worker_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_MODEL_POOL_RESOURCE_MANAGER_H_
