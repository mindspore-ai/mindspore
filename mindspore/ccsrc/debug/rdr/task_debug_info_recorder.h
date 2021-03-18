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
#ifndef MINDSPORE_CCSRC_DEBUG_RDR_TASK_DEBUG_INFO_RECORDER_H_
#define MINDSPORE_CCSRC_DEBUG_RDR_TASK_DEBUG_INFO_RECORDER_H_
#include <vector>
#include <string>
#include <memory>

#include "debug/rdr/base_recorder.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
class TaskDebugInfo;
}  // namespace tasksink
}  // namespace ascend
}  // namespace device

using TaskDebugInfoPtr = std::shared_ptr<device::ascend::tasksink::TaskDebugInfo>;
class TaskDebugInfoRecorder : public BaseRecorder {
 public:
  TaskDebugInfoRecorder() {}
  TaskDebugInfoRecorder(const std::string &module, const std::string &name,
                        const std::vector<TaskDebugInfoPtr> &task_debug_info)
      : BaseRecorder(module, name), task_debug_info_(task_debug_info) {}
  ~TaskDebugInfoRecorder() {}
  virtual void Export();

 private:
  std::vector<TaskDebugInfoPtr> task_debug_info_;
};
using TaskDebugInfoRecorderPtr = std::shared_ptr<TaskDebugInfoRecorder>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_TASK_DEBUG_INFO_RECORDER_H_
