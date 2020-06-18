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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_REPORTER_DESC_REPORTER_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_REPORTER_DESC_REPORTER_H_

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "toolchain/prof_reporter.h"
#include "device/ascend/profiling/reporter/profiling_desc.h"
#include "utils/contract.h"
#include "session/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
class DescReporter {
 public:
  virtual ~DescReporter() = 0;
  DescReporter(int device_id, std::string file_name) : device_id_(device_id), file_name_(std::move(file_name)) {}

  virtual void ReportData() = 0;

 protected:
  void ReportByLine(const std::string &data, const std::string &file_name) const;
  void ReportAllLine();

  int device_id_;
  std::string file_name_;
  std::vector<std::shared_ptr<ProfDesc>> prof_desc_list_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_REPORTER_DESC_REPORTER_H_
