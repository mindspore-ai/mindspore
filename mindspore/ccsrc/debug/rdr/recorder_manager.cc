/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "debug/rdr/recorder_manager.h"
#include <utility>
#include "debug/rdr/base_recorder.h"
#include "debug/env_config_parser.h"

namespace mindspore {
void RecorderManager::UpdateRdrEnable() {
  static bool updated = false;
  if (updated) {
    return;
  }
  auto &config_parser = mindspore::EnvConfigParser::GetInstance();
  rdr_enable_ = config_parser.rdr_enabled();
  if (config_parser.has_rdr_setting()) {
#ifdef __linux__
    if (!rdr_enable_) {
      MS_LOG(WARNING) << "Please set the 'enable' as true using 'rdr' setting in file '" << config_parser.config_path()
                      << "' if you want to use RDR.";
    }
#else
    if (rdr_enable_) {
      MS_LOG(WARNING) << "The RDR only supports linux os currently.";
    }
    rdr_enable_ = false;
#endif
  }
  updated = true;
}

bool RecorderManager::RecordObject(const BaseRecorderPtr &recorder) {
  if (!rdr_enable_) {
    return false;
  }

  if (recorder == nullptr) {
    MS_LOG(ERROR) << "Register recorder module with nullptr.";
    return false;
  }
  std::string module = recorder->GetModule();
  std::lock_guard<std::mutex> lock(mtx_);
  recorder_container_[module].push_back(std::move(recorder));
  return true;
}

void RecorderManager::TriggerAll() {
  if (!rdr_enable_) {
    return;
  }

  bool trigger = false;
  std::lock_guard<std::mutex> lock(mtx_);
  for (auto iter = recorder_container_.begin(); iter != recorder_container_.end(); ++iter) {
    for (auto &recorder : iter->second) {
      recorder->Export();
      trigger = true;
    }
  }
  if (!trigger) {
    MS_LOG(WARNING) << "There is no recorder to export.";
  }
}

void RecorderManager::ClearAll() {
  std::lock_guard<std::mutex> lock(mtx_);
  recorder_container_.clear();
}
}  // namespace mindspore
