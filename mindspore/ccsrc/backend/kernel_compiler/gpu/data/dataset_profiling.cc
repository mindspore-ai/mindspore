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
#include "backend/kernel_compiler/gpu/data/dataset_profiling.h"

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
GetNextProfiling::GetNextProfiling(const std::string &path) : profiling_path_(path) {}

void GetNextProfiling::GetDeviceId() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device_id_ = std::to_string(device_id);
}

void GetNextProfiling::Init() {
  GetDeviceId();
  file_name_ = profiling_path_ + "/minddata_getnext_profiling_" + device_id_ + ".txt";
  op_name_ = kGetNextOpName;
}

void GetNextProfiling::SaveProfilingData() {
  std::ofstream handle(file_name_, std::ios::trunc);
  if (!handle.is_open()) {
    MS_LOG(ERROR) << "Open get-next profiling file failed.";
    return;
  }
  for (uint32_t index = 0; index < queue_size_.size(); index++) {
    handle << Name() << " " << time_stamp_[index].first << " " << time_stamp_[index].second << " " << queue_size_[index]
           << std::endl;
  }
  handle.close();

  ChangeFileMode();
}

void GetNextProfiling::ChangeFileMode() {
  if (chmod(common::SafeCStr(file_name_), S_IRUSR | S_IWUSR) == -1) {
    MS_LOG(ERROR) << "Modify file:" << file_name_ << " to rw fail.";
    return;
  }
}

void GetNextProfiling::RecordData(uint32_t queue_size, uint64_t start_time_stamp, uint64_t end_time_stamp) {
  queue_size_.emplace_back(queue_size);
  std::pair<uint64_t, uint64_t> time_stamp(start_time_stamp, end_time_stamp);
  time_stamp_.emplace_back(time_stamp);
}

uint64_t GetNextProfiling::GetTimeStamp() const {
  auto cur_sys_clock = std::chrono::system_clock::now();
  uint64_t time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(cur_sys_clock.time_since_epoch()).count();
  return time_stamp;
}
}  // namespace kernel
}  // namespace mindspore
