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
  rdr_enable_ = config_parser.RdrEnabled();
  if (config_parser.HasRdrSetting()) {
#ifdef __linux__
    if (!rdr_enable_) {
      MS_LOG(WARNING) << "Not enabling RDR. You can enable RDR through configuration file or environment variables.";
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
  std::string name = recorder->GetName();
  std::pair<std::string, std::string> recorder_key(module, name);
  std::lock_guard<std::mutex> lock(mtx_);
  recorder_container_[recorder_key] = recorder;
  MS_LOG(INFO) << "RDR record object " << name << " in module \"" << module << "\".";
  return true;
}

BaseRecorderPtr RecorderManager::GetRecorder(std::string module, std::string name) {
  if (!rdr_enable_) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(mtx_);
  std::pair<std::string, std::string> recorder_key(module, name);
  auto item = recorder_container_.find(recorder_key);
  if (item != recorder_container_.end()) {
    return item->second;
  }
  return nullptr;
}

bool RecorderManager::RdrEnable() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return rdr_enable_;
}

bool RecorderManager::CheckRdrMemIsRecord() const {
  if (!rdr_enable_) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mtx_);
  return rdr_has_record_mem_;
}

void RecorderManager::SetRdrMemIsRecord(bool is_enable) {
  if (!rdr_enable_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mtx_);
  rdr_has_record_mem_ = is_enable;
}

void RecorderManager::TriggerAll() {
  if (!rdr_enable_) {
    return;
  }
  bool trigger = false;
  std::lock_guard<std::mutex> lock(mtx_);
  for (auto iter = recorder_container_.begin(); iter != recorder_container_.end(); ++iter) {
    iter->second->Export();
    trigger = true;
  }
  if (!trigger) {
    MS_LOG(WARNING) << "There is no recorder to export.";
  } else {
    MS_LOG(INFO) << "RDR export all recorders.";
  }
}

void RecorderManager::ClearAll() {
  if (!rdr_enable_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mtx_);
  recorder_container_.clear();
  rdr_has_record_mem_ = false;
  MS_LOG(INFO) << "RDR clear all recorders.";
}
}  // namespace mindspore
