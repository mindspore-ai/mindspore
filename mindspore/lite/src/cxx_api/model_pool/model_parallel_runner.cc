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
#include "src/cxx_api/model_pool/model_parallel_runner.h"
#include "src/cxx_api/model_pool/model_pool.h"
#include "src/common/log.h"

namespace mindspore {
Status ModelParallelRunner::Init(const std::string &model_path, const std::string &config_path, const Key &dec_key,
                                 const std::string &dec_mode) {
  auto status = ModelPool::GetInstance()->Init(model_path, config_path, dec_key, dec_mode);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model runner init failed.";
    return kLiteError;
  }
  return status;
}

std::vector<MSTensor> ModelParallelRunner::GetInputs() {
  auto inputs = ModelPool::GetInstance()->GetInputs();
  if (inputs.empty()) {
    MS_LOG(ERROR) << "model pool input is empty.";
    return {};
  }
  return inputs;
}

Status ModelParallelRunner::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                    const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "predict output is nullptr.";
    return kLiteNullptr;
  }
  auto status = ModelPool::GetInstance()->Predict(inputs, outputs, before, after);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model runner predict failed.";
    return kLiteError;
  }
  return kSuccess;
}
}  // namespace mindspore
