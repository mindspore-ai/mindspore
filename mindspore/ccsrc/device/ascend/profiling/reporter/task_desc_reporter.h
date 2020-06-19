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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_REPORTER_TASK_DESC_REPORTER_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_REPORTER_TASK_DESC_REPORTER_H_

#include <utility>
#include <string>
#include <vector>
#include "device/ascend/profiling/reporter/desc_reporter.h"

namespace mindspore {
namespace device {
namespace ascend {
class TaskDescReporter : public DescReporter {
 public:
  TaskDescReporter(int device_id, const std::string &file_name, std::vector<CNodePtr> cnode_list)
      : DescReporter(device_id, file_name, std::move(cnode_list)) {}
  ~TaskDescReporter() override = default;
  void ReportData() override;
  void set_task_ids(const std::vector<uint32_t> &task_ids) { task_ids_ = task_ids; }
  void set_stream_ids(const std::vector<uint32_t> &stream_ids) { stream_ids_ = stream_ids; }

 private:
  std::vector<uint32_t> task_ids_;
  std::vector<uint32_t> stream_ids_;
  void CheckStreamTaskValid(uint32_t task_id, uint32_t stream_id);
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_REPORTER_TASK_DESC_REPORTER_H_
