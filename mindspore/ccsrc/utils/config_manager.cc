/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "utils/config_manager.h"

#include <map>

#include "utils/log_adapter.h"

namespace mindspore {

ConfigManager& ConfigManager::GetInstance() noexcept {
  static ConfigManager instance;
  return instance;
}

void ConfigManager::SetDatasetModeConfig(const std::string& mode) {
  static const std::map<std::string, DatasetMode> mode_map = {{"normal", DS_NORMAL_MODE}, {"sink", DS_SINK_MODE}};
  if (mode_map.find(mode) == mode_map.end()) {
    MS_LOG(ERROR) << "Invalid dataset mode:" << mode;
    return;
  }
  GetInstance().dataset_mode_ = mode_map.at(mode);
}

void ConfigManager::ResetConfig() noexcept {
  parallel_strategy_ = ONE_DEVICE;
  dataset_mode_ = DS_NORMAL_MODE;
  dataset_param_ = DatasetGraphParam("", 0, 0, {}, {}, {});
  iter_num_ = 1;
}

}  // namespace mindspore
