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
#include "core/version_control/model.h"
#include <string>
#include "mindspore/ccsrc/utils/log_adapter.h"

namespace mindspore {
namespace serving {

MindSporeModel::MindSporeModel(const std::string &model_name, const std::string &model_path,
                               const std::string &model_version, const time_t &last_update_time)
    : model_name_(model_name),
      model_path_(model_path),
      model_version_(model_version),
      last_update_time_(last_update_time) {
  MS_LOG(INFO) << "init mindspore model, model_name = " << model_name_ << ", model_path = " << model_path_
               << ", model_version = " << model_version_ << ", last_update_time = " << last_update_time_;
}
}  // namespace serving
}  // namespace mindspore
