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

#include <algorithm>
#include "runtime/device/ascend/profiling/reporter/desc_reporter.h"
#include "runtime/device/ascend/profiling/profiling_manager.h"
#include "utils/log_adapter.h"

constexpr size_t kReportMaxLen = 1024;

namespace mindspore {
namespace device {
namespace ascend {
DescReporter::~DescReporter() = default;

void DescReporter::ReportByLine(const std::string &data, const std::string &file_name) const {
  auto tot_size = data.size();
  size_t cur_size = 0;
  while (cur_size < tot_size) {
    size_t remain_size = tot_size - cur_size;
    size_t report_size = std::min(remain_size, kReportMaxLen);

    ReporterData report_data{};
    report_data.deviceId = device_id_;
    report_data.dataLen = report_size;
    report_data.data = (unsigned char *)data.c_str() + cur_size;
    auto ret = memcpy_s(report_data.tag, MSPROF_ENGINE_MAX_TAG_LEN + 1, file_name.c_str(), file_name.length());
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "Memcpy_s report data tag failed";
    }
    auto report_ret = ProfilingManager::GetInstance().CallMsprofReport(NOT_NULL(&report_data));
    if (report_ret != 0) {
      MS_LOG(EXCEPTION) << "Report data failed";
    }
    if (report_size == 0) {
      MS_LOG(WARNING) << "Report_size is 0";
      break;
    }
    cur_size += report_size;
  }
}

void DescReporter::ReportAllLine() {
  for (const auto &desc : prof_desc_list_) {
    auto data = desc->ToString();
    ReportByLine(data, file_name_);
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
