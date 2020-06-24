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

#include <fstream>
#include <string>
#include "dataset/engine/perf/device_queue_tracing.h"
#include "dataset/util/path.h"
namespace mindspore {
namespace dataset {

Status DeviceQueueTracing::Record(const int32_t type, const int32_t extra_info, const int32_t batch_num,
                                  const int32_t value) {
  // Format: "type extra-info batch-num value"
  // type: 0: time,  1: connector size
  // extra-info: if type is 0 - 0: pipeline time, 1: push tdt time, 2: batch time
  //             if type is 1 - connector capacity
  // batch-num: batch number
  // value: if type is 0 - value is time(ms)
  //        if type is 1 - value is connector size
  // Examples:
  // 0 0 20 10 - The 20th batch took 10ms to get data from pipeline.
  // 1 64 20 5 - Connector size is 5 when get the 20th batch.Connector capacity is 64.
  std::string data = std::to_string(type) + " " + std::to_string(extra_info) + " " + std::to_string(batch_num) + " " +
                     std::to_string(value);
  value_.emplace_back(data);
  return Status::OK();
}

Status DeviceQueueTracing::SaveToFile() {
  if (value_.empty()) {
    return Status::OK();
  }

  std::ofstream handle(file_path_, std::ios::trunc);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Profiling file can not be opened.");
  }
  for (auto value : value_) {
    handle << value << "\n";
  }
  handle.close();

  return Status::OK();
}

Status DeviceQueueTracing::Init(const std::string &dir_path, const std::string &device_id) {
  file_path_ = (Path(dir_path) / Path("device_queue_profiling_" + device_id + ".txt")).toString();
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
