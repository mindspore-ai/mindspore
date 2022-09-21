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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_MODEL_POOL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_MODEL_POOL_H_
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>
#include <string>
#include <queue>
#include <shared_mutex>
#include <map>
#include <set>
#include "src/extendrt/dynamic_mem_allocator.h"
#include "include/api/status.h"
#include "include/api/context.h"
#include "include/api/model_parallel_runner.h"
#include "src/extendrt/cxx_api/model_pool/model_worker.h"
#include "src/extendrt/cxx_api/model_pool/predict_task_queue.h"
namespace mindspore {
using ModelPoolConfig = std::vector<std::shared_ptr<WorkerConfig>>;

struct TensorInfo {
  std::string name = "";
  enum DataType data_type;
  std::vector<int64_t> shape;
  mindspore::Format format;
  std::vector<QuantParam> quant_param;
};
struct WorkerInfo {
  std::shared_ptr<WorkerConfig> worker_config = nullptr;
  std::shared_ptr<ModelWorker> worker = nullptr;
};

struct ModelPoolInfo {
  size_t all_workers_num_ = 0;
  size_t mix_workers_num_ = 0;
  size_t workers_num_ = 0;
  int task_queue_num_ = 0;
  bool use_numa = false;
  size_t used_numa_node_num_ = 0;
  std::shared_ptr<PredictTaskQueue> predict_task_queue_ = nullptr;
  std::vector<std::shared_ptr<WorkerConfig>> model_pool_config;
  std::unordered_map<int, std::shared_ptr<Allocator>> numa_allocator_;
};

class ModelPool {
 public:
  ModelPool() = default;

  ~ModelPool();

  Status InitByPath(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config = nullptr);

  Status InitByBuf(const char *model_data, size_t size, const std::shared_ptr<RunnerConfig> &runner_config = nullptr);

  Status Init(const char *model_buf, size_t size, const std::shared_ptr<RunnerConfig> &runner_config);

  Status UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config);

  std::vector<MSTensor> GetInputs();

  std::vector<MSTensor> GetOutputs();

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

  bool IsInitialized() { return is_initialized_; }

 private:
  ModelPoolConfig CreateBaseStrategyModelPoolConfig(const std::shared_ptr<RunnerConfig> &runner_config,
                                                    Strategy strategy);
  std::shared_ptr<Context> GetInitContext(const std::shared_ptr<RunnerConfig> &runner_config, Strategy strategy);

  Status SetWorkersNum(const std::shared_ptr<RunnerConfig> &runner_config, const std::shared_ptr<Context> &context,
                       Strategy strategy);

  std::shared_ptr<mindspore::Context> GetDefaultContext();
  std::shared_ptr<Context> GetUserDefineContext(const std::shared_ptr<RunnerConfig> &runner_config);
  Status SetDefaultOptimalModelNum(int thread_num, Strategy strategy);

  Status SetModelBindMode(std::vector<std::vector<int>> *all_worker_bind_list, std::vector<int> *numa_node_id,
                          std::vector<int> *task_queue_id, int thread_num, Strategy strategy);
  Status SetNumaBindStrategy(std::vector<std::vector<int>> *all_worker_bind_list, std::vector<int> *numa_node_id,
                             std::vector<int> *task_queue_id, int thread_num, Strategy strategy);
  Status SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, std::vector<int> *numa_node_id,
                         std::vector<int> *task_queue_id, int thread_num, Strategy strategy);

  std::shared_ptr<ModelWorker> GetMaxWaitWorkerNum(int *max_wait_worker_node_id, int *max_wait_worker_num,
                                                   Strategy strategy);

  PredictTask *CreatePredictTask(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                 const MSKernelCallBack &before, const MSKernelCallBack &after, size_t *task_id);

  void UpdateFreeTaskId(size_t id);

  Status DistinguishPhysicalAndLogicalByNuma(const std::vector<int> &physical_core_list,
                                             const std::vector<int> &logical_core_list);

  Status InitNumaParameter(const std::shared_ptr<RunnerConfig> &runner_config);

  Status InitModelPoolBindList(const std::shared_ptr<Context> &init_context,
                               std::vector<std::vector<int>> *bind_core_list, std::vector<int> *bind_numa_list,
                               std::vector<int> *task_queue_id, Strategy strategy);

  Status SetWorkersNumaId(std::vector<int> *numa_node_id, std::vector<int> *task_queue_id, Strategy strategy);

  ModelPoolConfig CreateGpuModelPoolConfig(const std::shared_ptr<RunnerConfig> &runner_config,
                                           const std::shared_ptr<Context> &init_context, Strategy strategy);

  ModelPoolConfig CreateCpuModelPoolConfig(const std::shared_ptr<RunnerConfig> &runner_config,
                                           const std::shared_ptr<Context> &init_context,
                                           const std::vector<std::vector<int>> &all_worker_bind_list,
                                           const std::vector<int> &numa_node_id, const std::vector<int> &task_queue_id,
                                           Strategy strategy);

  Status CreateWorkers(const char *graph_buf, size_t size, const ModelPoolConfig &model_pool_config, Strategy strategy);

  Status CheckAffinityCoreList(const std::shared_ptr<RunnerConfig> &runner_config);

  Status CheckThreadNum(const std::shared_ptr<RunnerConfig> &runner_config);

  Status InitAdvancedStrategy(const char *model_buf, size_t size, int base_thread_num);

  Status InitBaseStrategy(const char *model_buf, size_t size, const std::shared_ptr<RunnerConfig> &runner_config);

  Status CreateModelPoolWorker(const char *model_buf, size_t size, ModelPoolConfig model_pool_config,
                               Strategy strategy);

  Strategy UpdateStrategy();

  Status CanUseAllPhysicalResources(int *percentage);

  int GetDefaultThreadNum(int worker_num = 0);

 private:
  bool use_advanced_strategy_ = false;
  bool use_gpu_ = false;

  // different workers get tasks from different task queues.
  // currently task queues are distinguished according to different numa node numbers.
  // if you do not distinguish between numa nodes, the default task queue number is 0.
  // store worker-related information
  std::unordered_map<Strategy, std::vector<std::shared_ptr<WorkerInfo>>> all_workers_;
  // store model pool related information
  std::unordered_map<Strategy, ModelPoolInfo> model_pool_info_;
  std::vector<char *> model_bufs_;
  char *graph_buf_ = nullptr;

  // save all worker thread
  std::vector<std::thread> worker_thread_vec_;
  std::mutex predict_task_mutex_;
  std::vector<TensorInfo> inputs_info_;
  std::vector<TensorInfo> outputs_info_;

  // create predict task
  PredictTask *tasks_ = nullptr;
  std::mutex task_id_mutex_;
  std::queue<size_t> free_tasks_id_;

  // bind core
  bool is_user_core_list_ = false;

  // use numa
  bool numa_available_ = false;
  size_t numa_node_num_ = -1;
  // numa id -> core id
  std::vector<std::vector<int>> numa_physical_cores_;
  std::vector<std::vector<int>> numa_logical_cores_;

  bool use_numa_bind_mode_ = false;
  bool bind_core_available_ = true;
  bool can_use_all_physical_core_ = true;
  int can_use_core_num_ = -1;
  int all_core_num_ = -1;
  std::shared_mutex model_pool_mutex_;
  bool is_initialized_ = false;
  std::string model_path_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_MODEL_POOL_H_
