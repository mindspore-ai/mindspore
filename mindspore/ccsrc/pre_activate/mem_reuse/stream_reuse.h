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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_STREAM_REUSE_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_STREAM_REUSE_H_
#include <cmath>
#include <map>
#include <set>
#include <list>
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <utility>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include "session/anf_runtime_algorithm.h"
#include "pre_activate/mem_reuse/kernel_refcount.h"

#ifdef ENABLE_D
#include "device/ascend/ascend_stream_assign.h"
#endif

namespace mindspore {
namespace memreuse {
class StreamReuse {
 public:
  StreamReuse() = default;
  ~StreamReuse() = default;
  void SetStreamReuseResource();
  void InitReusableStreamMap();
  std::vector<std::pair<uint32_t, uint32_t>> SortLogicPhysicMapToList();
  std::unordered_map<int, std::set<uint32_t>> GetLogicPhysicsStreamMap();
  void set_logic_physic_map(const std::unordered_map<uint32_t, uint32_t> &logic_physic_map) {
    logic_physic_map_ = logic_physic_map;
  }
  void set_logic_independent_map(const std::unordered_map<uint32_t, uint32_t> &logic_independent_map) {
    logic_independent_map_ = logic_independent_map;
  }
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> parallel_streams_map() { return parallel_streams_map_; }

 private:
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> parallel_streams_map_;
  std::unordered_map<uint32_t, uint32_t> logic_physic_map_;
  std::unordered_map<uint32_t, uint32_t> logic_independent_map_;
};
}  // namespace memreuse
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_STREAM_REUSE_H_
