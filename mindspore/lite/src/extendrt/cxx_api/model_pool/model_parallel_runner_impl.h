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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_MODEL_PARALLEL_RUNNER_IMPL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_MODEL_PARALLEL_RUNNER_IMPL_H_
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <string>
#include "src/extendrt/cxx_api/model_pool/model_pool.h"
#include "include/api/context.h"
namespace mindspore {
class ModelParallelRunnerImpl {
 public:
  ModelParallelRunnerImpl() = default;
  ~ModelParallelRunnerImpl();

  Status Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config = nullptr);

  Status Init(const void *model_data, size_t data_size, const std::shared_ptr<RunnerConfig> &runner_config = nullptr);

  std::vector<MSTensor> GetInputs();

  std::vector<MSTensor> GetOutputs();

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

 private:
  ModelPool *model_pool_ = nullptr;
  std::shared_mutex model_parallel_runner_impl_mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_MODEL_PARALLEL_RUNNER_IMPL_H_
