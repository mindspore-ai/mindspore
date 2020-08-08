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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_REPORTER_POINT_REPORTER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_REPORTER_POINT_REPORTER_H_

#include <memory>
#include <string>
#include "runtime/device/ascend/profiling/reporter/desc_reporter.h"

namespace mindspore {
namespace device {
namespace ascend {
class PointReporter : public DescReporter {
 public:
  PointReporter(uint32_t device_id, const std::string &file_name) : DescReporter(device_id, file_name) {}
  ~PointReporter() override = default;
  void ReportData() override;
  void AddReportData(const std::shared_ptr<ProfDesc> &prof_desc);
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_REPORTER_POINT_REPORTER_H_
