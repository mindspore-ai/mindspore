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
#ifndef MINDSPORE_CCSRC_DEBUG_RDR_RECORDER_MANAGER_H_
#define MINDSPORE_CCSRC_DEBUG_RDR_RECORDER_MANAGER_H_
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace mindspore {
class BaseRecorder;
using BaseRecorderPtr = std::shared_ptr<BaseRecorder>;
using BaseRecorderPtrList = std::vector<BaseRecorderPtr>;
class RecorderManager {
 public:
  static RecorderManager &Instance() {
    static RecorderManager manager;
    manager.UpdateRdrEnable();
    return manager;
  }

  void UpdateRdrEnable();
  bool RdrEnable() const { return rdr_enable_; }
  bool RecordObject(const BaseRecorderPtr &recorder);
  void TriggerAll();
  void ClearAll();

 private:
  RecorderManager() {}
  ~RecorderManager() {}

  bool rdr_enable_{false};

  mutable std::mutex mtx_;
  // module, BaserRecorderPtrList
  std::unordered_map<std::string, BaseRecorderPtrList> recorder_container_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_RECORDER_MANAGER_H_
