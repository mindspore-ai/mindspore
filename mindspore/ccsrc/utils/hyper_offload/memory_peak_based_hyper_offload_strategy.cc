/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "utils/hyper_offload/memory_peak_based_hyper_offload_strategy.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {

constexpr double kMemoryThresholdRatio = 0.90;
constexpr size_t kOffloadZoneMergeMaxGap = 5;

size_t GetNodeSize(const AnfNodePtr &node, size_t index) {
  try {
    ShapeVector shape = common::AnfAlgo::GetOutputInferShape(node, index);
    TypeId type_id = common::AnfAlgo::GetOutputInferDataType(node, index);
    size_t elem_size = abstract::TypeIdSize(type_id);
    if (elem_size == 0) {
      return 0;
    }
    int64_t num_elements = 1;
    for (int64_t dim : shape) {
      if (dim <= 0) {
        return 0;
      }
      num_elements *= dim;
    }
    return static_cast<size_t>(num_elements) * elem_size;
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Failed to get tensor size for node: " << node->DebugString() << ", index: " << index;
    return 0;
  }
}

void UpdateMemoryProfile(const size_t start_index, const size_t end_index, const size_t size,
                         std::vector<size_t> *memory_profile) {
  if (start_index >= end_index) {
    return;
  }
  for (size_t i = start_index; i < end_index; ++i) {
    if (i < memory_profile->size()) {
      (*memory_profile)[i] += size;
    }
  }
}

void ReduceMemoryProfile(const size_t start_index, const size_t end_index, const size_t size,
                         std::vector<size_t> *memory_profile) {
  if (start_index >= end_index) {
    return;
  }
  for (size_t i = start_index; i < end_index; ++i) {
    if (i < memory_profile->size()) {
      if ((*memory_profile)[i] >= size) {
        (*memory_profile)[i] -= size;
      } else {
        (*memory_profile)[i] = 0;
      }
    }
  }
}

size_t ComputeTotalParameterMemory(const FuncGraphPtr &graph) {
  if (graph == nullptr) {
    return 0;
  }
  return std::accumulate(graph->parameters().begin(), graph->parameters().end(), size_t(0),
                         [](size_t sum, const AnfNodePtr &param) {
                           if (param != nullptr && param->isa<Parameter>()) {
                             return sum + GetNodeSize(param, 0);
                           }
                           return sum;
                         });
}

struct CandidateInfo {
  UserInfo user_info;
  size_t size;
  size_t creation_index;
  size_t last_usage_index;
};

void BuildNodeToIndexMap(const CNodePtrList &execution_order, std::map<AnfNodePtr, size_t> *node_to_index) {
  for (size_t i = 0; i < execution_order.size(); ++i) {
    (*node_to_index)[execution_order[i]] = i;
  }
}

void BuildMemoryProfileAndCandidates(const UserInfoList &user_info_list,
                                     const std::map<AnfNodePtr, size_t> &node_to_index, size_t order_size,
                                     size_t parameter_memory_baseline, std::vector<size_t> *memory_profile,
                                     std::vector<CandidateInfo> *all_candidates) {
  memory_profile->assign(order_size, parameter_memory_baseline);
  all_candidates->clear();
  for (const auto &info : user_info_list) {
    if (!IsValidOffloadNode(info)) {
      continue;
    }
    auto node = info.node.first;
    if (node_to_index.find(node) == node_to_index.end()) {
      continue;
    }
    size_t creation_idx = node_to_index.at(node);
    size_t last_usage_idx = creation_idx;
    bool found_user = false;
    for (const auto &user : info.users) {
      auto user_node = user.first;
      if (node_to_index.find(user_node) != node_to_index.end()) {
        size_t user_idx = node_to_index.at(user_node);
        if (user_idx > last_usage_idx) {
          last_usage_idx = user_idx;
          found_user = true;
        }
      }
    }
    if (!found_user || last_usage_idx <= creation_idx) {
      continue;
    }
    size_t size = GetNodeSize(node, info.node.second);
    if (size == 0) {
      continue;
    }
    UpdateMemoryProfile(creation_idx, last_usage_idx + 1, size, memory_profile);
    all_candidates->push_back({info, size, creation_idx, last_usage_idx});
  }
}

std::optional<int> SelectBestCandidateIdx(const std::vector<CandidateInfo> &all_candidates,
                                          const std::set<AnfNodePtr> &zone_offloaded, const OffloadZone &target_zone) {
  const auto &config = HyperOffloadConfig::GetInstance();
  const double w1 = config.score_weight_size();
  const double w2 = config.score_weight_cover_start();
  const double w3 = config.score_weight_cover_end();
  const double w4 = config.score_weight_neighbor_impact();
  double best_score = std::numeric_limits<double>::lowest();
  std::optional<int> best_candidate_idx;
  for (size_t i = 0; i < all_candidates.size(); ++i) {
    if (zone_offloaded.count(all_candidates[i].user_info.node.first)) {
      continue;
    }
    const auto &cand = all_candidates[i];
    bool covers_zone =
      (cand.creation_index < target_zone.start_index) && (cand.last_usage_index >= target_zone.end_index);
    bool covers_peak =
      (cand.creation_index < target_zone.peak_index) && (cand.last_usage_index >= target_zone.peak_index);
    bool overlaps_zone =
      (cand.creation_index < target_zone.end_index) && (cand.last_usage_index >= target_zone.start_index);
    if (!covers_zone && !covers_peak && !overlaps_zone) {
      continue;
    }
    double neighbor_impact = 0.0;
    for (const auto &other : all_candidates) {
      if (!zone_offloaded.count(other.user_info.node.first)) {
        continue;
      }
      size_t overlap_start = std::max(cand.creation_index, other.creation_index);
      size_t overlap_end = std::min(cand.last_usage_index, other.last_usage_index);
      if (overlap_start < overlap_end) {
        neighbor_impact += static_cast<double>(overlap_end - overlap_start);
      }
    }
    double score =
      w1 * cand.size +
      w2 * (target_zone.start_index > cand.creation_index ? target_zone.start_index - cand.creation_index : 0) +
      w3 * (cand.last_usage_index > target_zone.end_index ? cand.last_usage_index - target_zone.end_index : 0) -
      w4 * neighbor_impact;
    if (score > best_score) {
      best_score = score;
      best_candidate_idx = SizeToInt(i);
    }
  }
  return best_candidate_idx;
}

OffloadInfoList ProcessSingleZone(size_t zone_idx, const OffloadZone &target_zone, std::vector<size_t> zone_profile,
                                  const std::vector<CandidateInfo> &all_candidates, size_t max_mem) {
  OffloadInfoList zone_results;
  std::set<AnfNodePtr> zone_offloaded;
  while (true) {
    size_t cur_zone_peak = 0;
    for (size_t k = target_zone.start_index; k < target_zone.end_index && k < zone_profile.size(); ++k) {
      if (zone_profile[k] > cur_zone_peak) {
        cur_zone_peak = zone_profile[k];
      }
    }
    if (cur_zone_peak <= max_mem) {
      break;
    }
    auto best_candidate_idx = SelectBestCandidateIdx(all_candidates, zone_offloaded, target_zone);
    if (!best_candidate_idx.has_value()) {
      MS_LOG(WARNING) << "Cannot resolve OOM in OffloadZone(" << zone_idx << ") [" << target_zone.start_index << ", "
                      << target_zone.end_index << "], peak: " << cur_zone_peak;
      break;
    }
    const auto &best_cand = all_candidates[IntToSize(*best_candidate_idx)];
    zone_offloaded.insert(best_cand.user_info.node.first);
    ReduceMemoryProfile(best_cand.creation_index, best_cand.last_usage_index + 1, best_cand.size, &zone_profile);
    ReplaceInfo replace_info;
    replace_info.replace_start_node = best_cand.user_info.node;
    for (const auto &u : best_cand.user_info.users) {
      replace_info.replace_rest_nodes.push_back(u);
    }
    if (!replace_info.replace_rest_nodes.empty()) {
      OffloadInfo info;
      info.data_node = best_cand.user_info.node;
      info.replace_info_list.push_back(replace_info);
      zone_results.push_back(info);
    }
  }
  return zone_results;
}

void MergeZoneResultsIntoList(const std::vector<OffloadInfoList> &per_zone_results,
                              OffloadInfoList *offload_info_list) {
  for (const auto &zone_results : per_zone_results) {
    for (const auto &info : zone_results) {
      auto it = std::find_if(offload_info_list->begin(), offload_info_list->end(), [&info](const OffloadInfo &o) {
        return o.data_node.first == info.data_node.first && o.data_node.second == info.data_node.second;
      });
      if (it == offload_info_list->end()) {
        offload_info_list->push_back(info);
      } else {
        for (const auto &rep : info.replace_info_list) {
          for (const auto &u : rep.replace_rest_nodes) {
            auto &rest = it->replace_info_list[0].replace_rest_nodes;
            if (std::find(rest.begin(), rest.end(), u) == rest.end()) {
              rest.push_back(u);
            }
          }
        }
      }
    }
  }
  MS_LOG(INFO) << "MergeZoneResultsIntoList: merged " << per_zone_results.size()
               << " zones, total offload entries=" << offload_info_list->size();
}

std::vector<OffloadZone> FindOffloadZones(const std::vector<size_t> &memory_profile, size_t max_device_memory,
                                          size_t merge_gap = kOffloadZoneMergeMaxGap) {
  std::vector<OffloadZone> zones;
  bool in_zone = false;
  size_t start = 0;
  size_t peak_mem = 0;
  size_t peak_idx = 0;

  for (size_t i = 0; i < memory_profile.size(); ++i) {
    if (memory_profile[i] > max_device_memory) {
      if (!in_zone) {
        in_zone = true;
        start = i;
        peak_mem = memory_profile[i];
        peak_idx = i;
      } else {
        if (memory_profile[i] > peak_mem) {
          peak_mem = memory_profile[i];
          peak_idx = i;
        }
      }
    } else {
      if (in_zone) {
        in_zone = false;
        OffloadZone zone{start, i, peak_idx, peak_mem};
        zones.push_back(zone);
      }
    }
  }
  if (in_zone) {
    OffloadZone zone{start, memory_profile.size(), peak_idx, peak_mem};
    zones.push_back(zone);
  }

  if (zones.empty()) {
    return zones;
  }

  std::vector<OffloadZone> merged_zones;
  merged_zones.push_back(zones[0]);

  for (size_t i = 1; i < zones.size(); ++i) {
    auto &last = merged_zones.back();
    if (zones[i].start_index <= last.end_index + merge_gap) {
      last.end_index = zones[i].end_index;
      if (zones[i].peak_memory > last.peak_memory) {
        last.peak_memory = zones[i].peak_memory;
        last.peak_index = zones[i].peak_index;
      }
    } else {
      merged_zones.push_back(zones[i]);
    }
  }
  return merged_zones;
}
}  // namespace

OffloadInfoList MemoryPeakBasedHyperOffloadStrategy::Run(const CNodePtrList &execution_order,
                                                         const UserInfoList &user_info_list,
                                                         const FuncGraphPtr &graph) {
  OffloadInfoList offload_info_list;
  if (execution_order.empty()) {
    return offload_info_list;
  }

  size_t configured_max = HyperOffloadConfig::GetInstance().max_device_memory();
  if (configured_max == 0) {
    return offload_info_list;
  }
  // Apply safety margin: use lower threshold so offload starts before hitting configured limit.
  size_t max_mem = static_cast<size_t>(configured_max * kMemoryThresholdRatio);

  size_t parameter_memory = ComputeTotalParameterMemory(graph);

  std::map<AnfNodePtr, size_t> node_to_index;
  BuildNodeToIndexMap(execution_order, &node_to_index);

  std::vector<size_t> memory_profile;
  std::vector<CandidateInfo> all_candidates;
  BuildMemoryProfileAndCandidates(user_info_list, node_to_index, execution_order.size(), parameter_memory,
                                  &memory_profile, &all_candidates);

  std::vector<OffloadZone> zones = FindOffloadZones(memory_profile, max_mem);
  if (zones.empty()) {
    return offload_info_list;
  }
  size_t peak_memory = *std::max_element(memory_profile.begin(), memory_profile.end());
  size_t peak_activation = (peak_memory > parameter_memory) ? (peak_memory - parameter_memory) : 0;
  MS_LOG(INFO) << "Found " << zones.size() << " offload zones, memory profile peak: " << peak_memory
               << " (parameter: " << parameter_memory << ", activation: " << peak_activation << ")"
               << ", max_device_memory(configured): " << configured_max << ", effective_threshold: " << max_mem;

  std::vector<OffloadInfoList> per_zone_results;
  per_zone_results.reserve(zones.size());
  for (size_t z = 0; z < zones.size(); ++z) {
    per_zone_results.push_back(ProcessSingleZone(z, zones[z], memory_profile, all_candidates, max_mem));
  }

  MergeZoneResultsIntoList(per_zone_results, &offload_info_list);
  return offload_info_list;
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
