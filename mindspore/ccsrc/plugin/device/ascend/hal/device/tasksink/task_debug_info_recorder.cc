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
#include "plugin/device/ascend/hal/device/tasksink/task_debug_info_recorder.h"
#include <utility>
#include "plugin/device/ascend/hal/device/tasksink/task_generator.h"
#include "include/common/debug/rdr/recorder_manager.h"

namespace mindspore {
void TaskDebugInfoRecorder::Export() {
  auto realpath = GetFileRealPath();
  if (!realpath.has_value()) {
    return;
  }
  std::string file_path = realpath.value() + ".ir";
  device::ascend::tasksink::TaskGenerator::DumpTaskInfo(file_path, task_debug_info_);
}

namespace RDR {
bool RecordTaskDebugInfo(SubModuleId module, const std::string &name,
                         const std::vector<TaskDebugInfoPtr> &task_debug_info_list) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  TaskDebugInfoRecorderPtr task_debug_info_recorder =
    std::make_shared<TaskDebugInfoRecorder>(submodule_name, name, task_debug_info_list);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(task_debug_info_recorder));
  return ans;
}
}  // namespace RDR
}  // namespace mindspore
