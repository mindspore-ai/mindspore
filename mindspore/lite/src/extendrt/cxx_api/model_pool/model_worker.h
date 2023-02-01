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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_MODEL_WORKER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_MODEL_WORKER_H_
#include <queue>
#include <string>
#include <mutex>
#include <future>
#include <vector>
#include <utility>
#include <memory>
#include <map>
#include "include/api/model.h"
#include "src/extendrt/cxx_api/model_pool/predict_task_queue.h"
namespace mindspore {
class PredictTaskQueue;

struct WorkerConfig {
  std::map<std::string, std::map<std::string, std::string>> config_info;
  std::string config_path = "";
  std::shared_ptr<Context> context = nullptr;
  int numa_id = -1;
  int worker_id = -1;
};

class ModelWorker {
 public:
  ModelWorker() = default;

  ~ModelWorker() = default;

  Status Init(const char *model_buf, size_t size);

  Status UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config);

  std::vector<MSTensor> GetInputs();

  std::vector<MSTensor> GetOutputs();

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

  void WaitCreateWorkerDone();

  bool IsAvailable();

  void InitModelWorker(const char *model_buf, size_t size, const std::shared_ptr<WorkerConfig> &worker_config,
                       const std::shared_ptr<PredictTaskQueue> &predict_task_queue, bool *create_success);

  inline bool ModelIsNull() { return model_is_nullptr_; }

  inline int GetWorkerID() { return worker_id_; }

  void Run();

 private:
  std::pair<std::vector<std::vector<int64_t>>, bool> GetModelResize(const std::vector<MSTensor> &model_inputs,
                                                                    const std::vector<MSTensor> &inputs);

  Status CopyOutputTensor(std::vector<MSTensor> model_outputs, std::vector<MSTensor> *user_outputs);

  void PrintWorkerInfo();

 private:
  mindspore::Model *model_ = nullptr;
  std::shared_ptr<WorkerConfig> worker_config_ = nullptr;
  std::shared_ptr<PredictTaskQueue> predict_task_queue_ = nullptr;
  std::vector<MSTensor> origin_worker_inputs_;
  std::vector<MSTensor> origin_worker_outputs_;
  // Init worker
  bool create_work_done_ = false;
  std::mutex create_work_done_mutex_;
  std::condition_variable create_work_done_condition_;
  // run
  std::mutex mtx_worker_;
  std::atomic_bool available_ = true;
  bool model_is_nullptr_ = false;
  int worker_id_ = -1;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_MODEL_WORKER_H_
