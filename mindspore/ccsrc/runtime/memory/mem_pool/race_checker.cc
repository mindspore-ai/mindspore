/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>
#include "runtime/memory/mem_pool/race_checker.h"

namespace mindspore {
namespace device {
namespace tracker {
namespace graph {
void RaceChecker::RecordEvent(size_t stream_id, const std::string &event_id) {
  st_vec_[stream_id][stream_id] = st_vec_[stream_id][stream_id] + 1;
  event_vec_map_[event_id] = st_vec_[stream_id];
}

void RaceChecker::WaitEvent(size_t stream_id, const std::string &event_id) {
  auto iter = event_vec_map_.find(event_id);
  if (iter == event_vec_map_.end()) {
    MS_LOG(EXCEPTION) << "RaceChecker: Event id " << event_id << " is not found.";
  }
  for (size_t i = 0; i < iter->second.size(); i++) {
    st_vec_[stream_id][i] = std::max(st_vec_[stream_id][i], iter->second[i]);
  }
}

bool RaceChecker::CheckRead(uintptr_t start_addr, uintptr_t end_addr, size_t stream_id) {
  uint32_t start_index = discretizer_.GetDiscreteId(start_addr);
  uint32_t end_index = discretizer_.GetDiscreteId(end_addr);
  read_segment_tree_.Update(start_index, end_index, stream_id, st_vec_[stream_id][stream_id]);
  for (size_t i = 0; i < stream_size_; i++) {
    if (i == stream_id) {
      continue;
    }
    if (write_segment_tree_.Query(start_index, end_index, i) >= st_vec_[stream_id][i]) {
      MS_LOG(ERROR) << "RaceChecker: Read error, stream id: " << stream_id << ", other stream id: " << i
                    << ", start_addr: 0x" << std::hex << start_addr << ", end_addr: 0x" << end_addr << std::dec;
      return true;
    }
  }
  return false;
}

bool RaceChecker::CheckWrite(uintptr_t start_addr, uintptr_t end_addr, size_t stream_id) {
  uint32_t start_index = discretizer_.GetDiscreteId(start_addr);
  uint32_t end_index = discretizer_.GetDiscreteId(end_addr);
  write_segment_tree_.Update(start_index, end_index, stream_id, st_vec_[stream_id][stream_id]);
  for (size_t i = 0; i < stream_size_; i++) {
    if (i == stream_id) {
      continue;
    }
    if (read_segment_tree_.Query(start_index, end_index, i) >= st_vec_[stream_id][i]) {
      MS_LOG(ERROR) << "RaceChecker: Write error, stream id: " << stream_id << ", other stream id: " << i
                    << ", start_addr: 0x" << std::hex << start_addr << ", end_addr: 0x" << end_addr << std::dec;
      return true;
    }
    if (write_segment_tree_.Query(start_index, end_index, i) >= st_vec_[stream_id][i]) {
      MS_LOG(ERROR) << "RaceChecker: Write error, stream id: " << stream_id << ", other stream id: " << i
                    << ", start_addr: 0x" << std::hex << start_addr << ", end_addr: 0x" << end_addr << std::dec;
      return true;
    }
  }
  return false;
}

}  // namespace graph
}  // namespace tracker
}  // namespace device
}  // namespace mindspore
