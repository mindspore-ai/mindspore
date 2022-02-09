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
#ifndef MINDSPORE_INCLUDE_API_MODEL_POOL_MODEL_POOL_H
#define MINDSPORE_INCLUDE_API_MODEL_POOL_MODEL_POOL_H
#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <queue>
#include <map>
#include "include/api/status.h"
#include "include/api/context.h"
#include "src/cxx_api/model_pool/model_thread.h"
#include "src/cxx_api/model_pool/predict_task_queue.h"
namespace mindspore {
struct RunnerConfig {
  RunnerConfig(std::shared_ptr<Context> &ctx, int num) : model_ctx(ctx), num_model(num) {}
  std::shared_ptr<Context> model_ctx = nullptr;
  int num_model = 10;
};

class ModelPool {
 public:
  static ModelPool *GetInstance();
  ~ModelPool();

  Status Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config = nullptr,
              const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm);

  std::vector<MSTensor> GetInputs();

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

 private:
  ModelPool() = default;
  void SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, int thread_num);
  ModelPoolContex CreateModelContext(const std::shared_ptr<RunnerConfig> &runner_config);
  std::shared_ptr<Context> InitContext(const std::shared_ptr<RunnerConfig> &runner_config);
  Status SplitTensorByBatch(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                            std::vector<std::vector<MSTensor>> *new_inputs);
  Status ConcatPredictOutput(std::vector<std::vector<MSTensor>> *outputs, std::vector<MSTensor> *new_outputs);

  void *all_out_data = nullptr;
  std::vector<std::thread> model_thread_vec_;
  std::vector<MSTensor> model_inputs_;
  size_t num_models_ = 10;
  size_t batch_split_num_ = 4;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_MODEL_POOL_MODEL_POOL_H
