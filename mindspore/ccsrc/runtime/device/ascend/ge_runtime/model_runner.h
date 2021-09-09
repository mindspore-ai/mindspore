/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_MODEL_RUNNER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_MODEL_RUNNER_H_

#include <memory>
#include <map>
#include <vector>
#include <tuple>
#include <string>
#include "runtime/device/ascend/ge_runtime/davinci_model.h"

namespace mindspore::ge::model_runner {
class RuntimeModel;
using RuntimeInfo = std::tuple<uint32_t, uint32_t, void *>;
class ModelRunner {
 public:
  static ModelRunner &Instance();

  void LoadDavinciModel(uint32_t device_id, uint64_t session_id, uint32_t model_id,
                        const std::shared_ptr<DavinciModel> &davinci_model);

  void DistributeTask(uint32_t model_id);

  void LoadModelComplete(uint32_t model_id);

  const std::vector<uint32_t> &GetTaskIdList(uint32_t model_id) const;

  const std::vector<uint32_t> &GetStreamIdList(uint32_t model_id) const;

  const std::map<std::string, std::shared_ptr<RuntimeInfo>> &GetRuntimeInfoMap(uint32_t model_id) const;

  void *GetModelHandle(uint32_t model_id) const;

  void *GetModelStream(uint32_t model_id) const;

  void UnloadModel(uint32_t model_id);

  void RunModel(uint32_t model_id);

 private:
  ModelRunner() = default;
  ~ModelRunner() = default;

  std::map<uint32_t, std::shared_ptr<RuntimeModel>> runtime_models_;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_MODEL_RUNNER_H_
