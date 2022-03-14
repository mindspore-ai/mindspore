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

#ifndef MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_MODEL_WORKER_H_
#define MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_MODEL_WORKER_H_
#include <queue>
#include <string>
#include <mutex>
#include <future>
#include <vector>
#include <utility>
#include <memory>
#include "include/api/model.h"
#include "src/cxx_api/model_pool/predict_task_queue.h"
namespace mindspore {
class ModelWorker {
 public:
  ModelWorker() = default;

  ~ModelWorker() = default;

  Status Init(const char *model_buf, size_t size, const std::shared_ptr<Context> &model_context, int node_id = -1);

  std::vector<MSTensor> GetInputs();

  std::vector<MSTensor> GetOutputs();

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

  void Run(int node_id, const std::shared_ptr<PredictTaskQueue> &predict_task_queue);

 private:
  std::pair<std::vector<std::vector<int64_t>>, bool> GetModelResize(const std::vector<MSTensor> &model_inputs,
                                                                    const std::vector<MSTensor> &inputs);

 private:
  std::shared_ptr<mindspore::Model> model_ = nullptr;
  std::mutex mtx_model_;
  bool need_copy_output_ = true;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_CXX_API_MODEL_POOL_MODEL_WORKER_H_
