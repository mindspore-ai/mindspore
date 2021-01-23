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
#include "mindspore/core/base/base.h"
#include "mindspore/core/ir/func_graph.h"

namespace mindspore {
bool RecorderManager::RecordObject(const BaseRecorderPtr &recorder) {
  if (recorder == nullptr) {
    MS_LOG(ERROR) << "register recorder module with nullptr.";
    return false;
  }
  std::string module = recorder->GetModule();
  std::lock_guard<std::mutex> lock(mtx_);
  recorder_container_[module].push_back(std::move(recorder));
  return true;
}

void RecorderManager::TriggerAll() {
  auto &config_parser_ptr = mindspore::EnvConfigParser::GetInstance();
  config_parser_ptr.Parse();
  if (!config_parser_ptr.rdr_enabled()) {
    MS_LOG(INFO) << "RDR is not enable.";
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
}  // namespace mindspore
