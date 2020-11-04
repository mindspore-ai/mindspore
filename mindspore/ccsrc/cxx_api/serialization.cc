/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "include/api/serialization.h"
#include "utils/log_adapter.h"

namespace mindspore::api {
Status Serialization::LoadCheckPoint(const std::string &ckpt_file, std::map<std::string, Buffer> *parameters) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status Serialization::SetParameters(const std::map<std::string, Buffer> &parameters, Model *model) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, Buffer *model_data) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, const std::string &model_file) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}
}  // namespace mindspore::api
