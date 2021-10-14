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

#include "runtime/device/ascend/ge_runtime/model_runner.h"
#include "runtime/device/ascend/ge_runtime/runtime_model.h"
#include "runtime/device/ascend/ge_runtime/davinci_model.h"
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore::ge::model_runner {
ModelRunner &ModelRunner::Instance() {
  static ModelRunner instance{};  // Guaranteed to be destroyed.
  return instance;
}

void ModelRunner::LoadDavinciModel(uint32_t device_id, uint64_t session_id, uint32_t model_id,
                                   const std::shared_ptr<DavinciModel> &davinci_model) {
  std::shared_ptr<RuntimeModel> model = std::make_shared<RuntimeModel>();
  model->Load(device_id, session_id, davinci_model);
  runtime_models_[model_id] = model;
}

void ModelRunner::DistributeTask(uint32_t model_id) {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    MS_LOG(EXCEPTION) << "Model id " << model_id << " not found.";
  }
  MS_EXCEPTION_IF_NULL(model_iter->second);
  model_iter->second->DistributeTask();
}

void ModelRunner::LoadModelComplete(uint32_t model_id) {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    MS_LOG(EXCEPTION) << "Model id " << model_id << " not found.";
  }
  MS_EXCEPTION_IF_NULL(model_iter->second);
  model_iter->second->LoadComplete();
}

const std::vector<uint32_t> &ModelRunner::GetTaskIdList(uint32_t model_id) const {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    MS_LOG(EXCEPTION) << "Model id " << model_id << " not found.";
  }
  MS_EXCEPTION_IF_NULL(model_iter->second);
  return model_iter->second->GetTaskIdList();
}

const std::vector<uint32_t> &ModelRunner::GetStreamIdList(uint32_t model_id) const {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    MS_LOG(EXCEPTION) << "Model id " << model_id << " not found.";
  }
  MS_EXCEPTION_IF_NULL(model_iter->second);
  return model_iter->second->GetStreamIdList();
}

const std::map<std::string, std::shared_ptr<RuntimeInfo>> &ModelRunner::GetRuntimeInfoMap(uint32_t model_id) const {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    MS_LOG(EXCEPTION) << "Model id " << model_id << " not found.";
  }
  MS_EXCEPTION_IF_NULL(model_iter->second);
  return model_iter->second->GetRuntimeInfoMap();
}

void *ModelRunner::GetModelHandle(uint32_t model_id) const {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    MS_LOG(EXCEPTION) << "Model id " << model_id << " not found.";
  }
  MS_EXCEPTION_IF_NULL(model_iter->second);
  return model_iter->second->GetModelHandle();
}

void *ModelRunner::GetModelStream(uint32_t model_id) const {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    MS_LOG(EXCEPTION) << "Model id " << model_id << " not found.";
  }
  MS_EXCEPTION_IF_NULL(model_iter->second);
  return model_iter->second->GetModelStream();
}

void ModelRunner::UnloadModel(uint32_t model_id) {
  auto iter = runtime_models_.find(model_id);
  if (iter != runtime_models_.end()) {
    (void)runtime_models_.erase(iter);
  }
}

void ModelRunner::RunModel(uint32_t model_id) {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    MS_LOG(EXCEPTION) << "Model id " << model_id << " not found.";
  }
  MS_EXCEPTION_IF_NULL(model_iter->second);
  model_iter->second->Run();
}
}  // namespace mindspore::ge::model_runner
