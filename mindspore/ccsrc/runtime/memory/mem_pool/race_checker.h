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

#ifndef MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_RACE_CHECKER_H_
#define MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_RACE_CHECKER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "runtime/memory/mem_pool/tracker_graph.h"
#include "runtime/memory/mem_pool/address_discretizer.h"
#include "runtime/memory/mem_pool/max_segment_tree.h"

namespace mindspore {
namespace device {
namespace tracker {
namespace graph {

class RaceChecker {
 public:
  static constexpr size_t kBias = 5;
  RaceChecker() = delete;
  RaceChecker(std::vector<uintptr_t> addresses, size_t stream_size)
      : discretizer_(addresses),
        read_segment_tree_(discretizer_.GetDiscretizedCount() + kBias, stream_size),
        write_segment_tree_(discretizer_.GetDiscretizedCount() + kBias, stream_size),
        stream_size_(stream_size) {
    st_vec_.resize(stream_size, std::vector<uint32_t>(stream_size, 1));
  }

  void RecordEvent(size_t stream_id, const std::string &event_id);
  void WaitEvent(size_t stream_id, const std::string &event_id);
  bool CheckRead(uintptr_t start_addr, uintptr_t end_addr, size_t stream_id);
  bool CheckWrite(uintptr_t start_addr, uintptr_t end_addr, size_t stream_id);

 private:
  AddressDiscretizer discretizer_;
  MaxSegmentTree<uint32_t> read_segment_tree_;
  MaxSegmentTree<uint32_t> write_segment_tree_;
  size_t stream_size_;
  std::unordered_map<std::string, std::vector<uint32_t>> event_vec_map_;
  std::vector<std::vector<uint32_t>> st_vec_;
};
}  // namespace graph
}  // namespace tracker
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_RACE_CHECKER_H_
