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

#include "runtime/device/ascend/profiling/reporter/op_name_task_stream_reporter.h"

namespace mindspore {
namespace device {
namespace ascend {
void OpNameTaskStreamReporter::ReportData() {
  MS_LOG(INFO) << "ReportData start";

  std::map<std::string, std::vector<std::pair<uint32_t, uint32_t>>> op_name_map;
  for (auto &iter : stream_id_task_id_op_name_map_) {
    auto pair = iter.first;
    auto op_name = iter.second;
    auto ret = op_name_map.find(op_name);
    if (ret == op_name_map.end()) {
      auto vect = std::vector<std::pair<uint32_t, uint32_t>>(1, pair);
      auto emplace_ret = op_name_map.emplace(op_name, vect);
      if (!emplace_ret.second) {
        MS_LOG(WARNING) << "Duplicate op_name:" << op_name << " task_id:" << pair.first << " stream_id:" << pair.second;
      }
    } else {
      ret->second.emplace_back(pair);
    }
  }

  for (const auto &iter : op_name_map) {
    auto desc_ptr = std::make_shared<TaskStreamOpNameDesc>(iter.first, iter.second);
    prof_desc_list_.emplace_back(desc_ptr);
  }
  ReportAllLine();
  MS_LOG(INFO) << "ReportData end";
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
