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

#include "pre_activate/mem_reuse/stream_reuse.h"

namespace mindspore {
namespace memreuse {
void StreamReuse::SetStreamReuseResource() {
#ifdef ENABLE_D
  auto logic_physic_map = device::ascend::AscendStreamAssign::GetInstance().logic_to_physic_map();
  auto logic_independent_map = device::ascend::AscendStreamAssign::GetInstance().logic_to_independent_map();
  MS_LOG(INFO) << "stream mem reuse for Davici";
  if (!logic_independent_map.empty() && !logic_physic_map.empty()) {
    set_logic_physic_map(logic_physic_map);
    set_logic_independent_map(logic_independent_map);
    InitReusableStreamMap();
  } else {
    MS_LOG(INFO) << "Non task sink or No Parallel stream exists";
  }
#endif
  MS_LOG(INFO) << "no need to set stream mem reuse resource";
}

std::vector<std::pair<uint32_t, uint32_t>> StreamReuse::SortLogicPhysicMapToList() {
  std::vector<std::pair<uint32_t, uint32_t>> logic_physic_list;
  (void)std::transform(logic_physic_map_.begin(), logic_physic_map_.end(), std::back_inserter(logic_physic_list),
                       [](std::pair<uint32_t, uint32_t> log_phy) { return log_phy; });
  std::sort(
    logic_physic_list.begin(), logic_physic_list.end(),
    [](const std::pair<uint32_t, uint32_t> &logic_phyic_pair1, const std::pair<uint32_t, uint32_t> &logic_phyic_pair2) {
      return logic_phyic_pair1.second < logic_phyic_pair2.second;
    });
  return logic_physic_list;
}

std::unordered_map<int, std::set<uint32_t>> StreamReuse::GetLogicPhysicsStreamMap() {
  auto logic_physic_list = SortLogicPhysicMapToList();
  std::unordered_map<int, std::set<uint32_t>> logic_phyics_map;
  for (size_t i = 0; i < logic_physic_list.size() - IntToSize(1); ++i) {
    auto curr_logic_physic = logic_physic_list.at(i);
    auto next_logic_physic = logic_physic_list.at(i + 1);
    for (auto j = curr_logic_physic.second; j < next_logic_physic.second; ++j) {
      (void)logic_phyics_map[curr_logic_physic.first].insert(j);
    }
  }
  // sort the logic independ map by value
  std::map<uint32_t, uint32_t> temp_map;
  for (const auto &logic_independ : logic_independent_map_) {
    (void)temp_map.insert(std::make_pair(logic_independ.second, logic_independ.first));
  }
  auto first_independent_stream_id = (*temp_map.begin()).first;
  auto last_physic_logic_stream_id = (*logic_physic_list.rbegin()).second;
  for (auto i = last_physic_logic_stream_id; i < first_independent_stream_id; ++i) {
    (void)logic_phyics_map[(*logic_physic_list.rbegin()).first].insert(i);
  }
  return logic_phyics_map;
}

void StreamReuse::InitReusableStreamMap() {
  // logic_phyics_map, key, logic_stream_id; value, physic_strema_ids included in that logic stream
  auto logic_phyics_map = GetLogicPhysicsStreamMap();
  // parallel_streams_map: key, current_stream_id; value, streams parallel to current stream
  for (const auto &logic_to_phyics : logic_phyics_map) {
    auto logic_stream_id = logic_to_phyics.first;
    auto iter_inde = logic_independent_map_.find(logic_stream_id);
    if (iter_inde != logic_independent_map_.end()) {
      // exist independent steam parallel to these logic streams
      auto independent_stream_id = iter_inde->second;
      auto physics_stream_id = logic_to_phyics.second;
      for (const auto &physic : physics_stream_id) {
        (void)parallel_streams_map_[physic].insert(independent_stream_id);
      }
    }
  }
  for (const auto &logic_to_independent : logic_independent_map_) {
    auto logic_stream_id = logic_to_independent.first;
    auto independent_stream_id = logic_to_independent.second;
    auto iter_physics = logic_phyics_map.find(logic_stream_id);
    if (iter_physics != logic_phyics_map.end()) {
      // exist logic steam parallel to these independent streams, default
      auto physics_set = iter_physics->second;
      for (const auto &physic : physics_set) {
        (void)parallel_streams_map_[independent_stream_id].insert(physic);
      }
    }
  }
}
}  // namespace memreuse
}  // namespace mindspore
