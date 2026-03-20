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
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "utils/hyper_offload/hyper_offload_mem_estimate.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {

constexpr double kMemoryThresholdRatio = 0.90;
constexpr size_t kOffloadZoneMergeMaxGap = 5;
constexpr double kBytesPerMb = 1024.0 * 1024.0;
constexpr size_t kPeakGapBonus = 1000;
constexpr size_t kNoPeakGapBonus = 1;
constexpr size_t kMinGapListSize = 2;

void BuildNodeToIndexMap(const CNodePtrList &execution_order, std::map<AnfNodePtr, size_t> *node_to_index) {
  for (size_t i = 0; i < execution_order.size(); ++i) {
    (*node_to_index)[execution_order[i]] = i;
  }
}

struct GapInfo {
  KernelWithIndex replace_start_node;
  std::vector<KernelWithIndex> replace_rest_nodes;
  size_t gap_start_index;
  size_t gap_end_index;
};

std::optional<GapInfo> GenerateGapBasedReplaceInfo(const PeakEstimateCandidate &cand,
                                                   const std::map<AnfNodePtr, size_t> &node_to_index,
                                                   const OffloadZone &target_zone) {
  std::vector<std::pair<KernelWithIndex, size_t>> abs_list;
  abs_list.push_back({cand.user_info.node, cand.creation_index});
  for (const auto &u : cand.user_info.users) {
    auto it = node_to_index.find(u.first);
    if (it != node_to_index.end()) {
      abs_list.push_back({u, it->second});
    }
  }
  if (abs_list.size() < kMinGapListSize) {
    return std::nullopt;
  }
  std::sort(abs_list.begin(), abs_list.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

  struct RelDist {
    KernelWithIndex start_node;
    KernelWithIndex end_node;
    size_t distance;
  };

  std::vector<RelDist> rel_list;
  for (size_t i = 1; i < abs_list.size(); ++i) {
    size_t d = abs_list[i].second - abs_list[i - 1].second;
    rel_list.push_back({abs_list[i - 1].first, abs_list[i].first, d});
  }

  std::vector<bool> need_offload;
  need_offload.reserve(rel_list.size());
  std::transform(rel_list.begin(), rel_list.end(), std::back_inserter(need_offload),
                 [](const RelDist &r) { return r.distance > 1U; });
  if (!std::any_of(need_offload.begin(), need_offload.end(), [](bool b) { return b; })) {
    return std::nullopt;
  }

  size_t best_gap_idx = 0;
  size_t best_score = 0;
  for (size_t i = 0; i < rel_list.size(); ++i) {
    if (!need_offload[i]) {
      continue;
    }
    size_t gap_start = abs_list[i].second + 1;
    size_t gap_end = abs_list[i + 1].second;
    bool contains_peak = (gap_start <= target_zone.peak_index && target_zone.peak_index < gap_end);
    size_t overlap_start = std::max(gap_start, target_zone.start_index);
    size_t overlap_end = std::min(gap_end, target_zone.end_index);
    size_t overlap = (overlap_start < overlap_end) ? (overlap_end - overlap_start) : 0;
    size_t bonus = contains_peak ? kPeakGapBonus : kNoPeakGapBonus;
    size_t score = rel_list[i].distance * bonus + overlap;
    if (score > best_score) {
      best_score = score;
      best_gap_idx = i;
    }
  }

  const auto &best = rel_list[best_gap_idx];
  size_t gap_start_idx = abs_list[best_gap_idx].second + 1;
  size_t gap_end_idx = abs_list[best_gap_idx + 1].second;

  GapInfo info;
  info.replace_start_node = best.start_node;
  info.replace_rest_nodes.push_back(best.end_node);
  info.gap_start_index = gap_start_idx;
  info.gap_end_index = gap_end_idx;

  for (size_t j = best_gap_idx + 1; j < rel_list.size() && !need_offload[j]; ++j) {
    info.replace_rest_nodes.push_back(rel_list[j].end_node);
  }

  return info;
}

size_t ComputeZonePeak(const OffloadZone &target_zone, const std::vector<size_t> &zone_profile) {
  size_t cur_peak = 0;
  for (size_t k = target_zone.start_index; k < target_zone.end_index && k < zone_profile.size(); ++k) {
    if (zone_profile[k] > cur_peak) {
      cur_peak = zone_profile[k];
    }
  }
  return cur_peak;
}

std::optional<size_t> SelectBestCandidateIdx(const std::vector<PeakEstimateCandidate> &all_candidates,
                                             const std::set<AnfNodePtr> &zone_offloaded,
                                             const OffloadZone &target_zone) {
  const auto &config = HyperOffloadConfig::GetInstance();
  const double w1 = config.score_weight_size();
  const double w2 = config.score_weight_cover_start();
  const double w3 = config.score_weight_cover_end();
  const double w4 = config.score_weight_neighbor_impact();
  double best_score = std::numeric_limits<double>::lowest();
  std::optional<size_t> best_candidate_idx;
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
      best_candidate_idx = i;
    }
  }
  return best_candidate_idx;
}

void TryProcessBestCandidate(const PeakEstimateCandidate &best_cand, const std::map<AnfNodePtr, size_t> &node_to_index,
                             const OffloadZone &target_zone, std::vector<size_t> *zone_profile,
                             std::set<AnfNodePtr> *zone_offloaded, OffloadInfoList *zone_results) {
  auto gap_opt = GenerateGapBasedReplaceInfo(best_cand, node_to_index, target_zone);
  zone_offloaded->insert(best_cand.user_info.node.first);
  if (!gap_opt.has_value()) {
    return;
  }
  const auto &gap = gap_opt.value();
  hyper_offload::ReduceMemoryProfile(gap.gap_start_index, gap.gap_end_index, best_cand.size, zone_profile);
  if (gap.replace_rest_nodes.empty()) {
    return;
  }
  ReplaceInfo replace_info;
  replace_info.replace_start_node = gap.replace_start_node;
  replace_info.replace_rest_nodes = gap.replace_rest_nodes;
  OffloadInfo info;
  info.data_node = best_cand.user_info.node;
  info.replace_info_list.push_back(replace_info);
  zone_results->push_back(info);
}

OffloadInfoList ProcessSingleZone(size_t zone_idx, const OffloadZone &target_zone, std::vector<size_t> zone_profile,
                                  const std::vector<PeakEstimateCandidate> &all_candidates, size_t max_mem,
                                  size_t configured_max, const std::map<AnfNodePtr, size_t> &node_to_index) {
  OffloadInfoList zone_results;
  std::set<AnfNodePtr> zone_offloaded;
  while (true) {
    size_t cur_zone_peak = ComputeZonePeak(target_zone, zone_profile);
    if (cur_zone_peak <= max_mem) {
      break;
    }
    auto best_candidate_idx = SelectBestCandidateIdx(all_candidates, zone_offloaded, target_zone);
    if (!best_candidate_idx.has_value()) {
      MS_LOG(ERROR) << "[HyperOffload] All offload candidates exhausted but zone peak still exceeds threshold. "
                    << "OffloadZone(" << zone_idx << ") [" << target_zone.start_index << ", " << target_zone.end_index
                    << "], zone_peak=" << cur_zone_peak << " (" << (cur_zone_peak / kBytesPerMb) << " MB)"
                    << ", configured_max_device_memory=" << configured_max << " (" << (configured_max / kBytesPerMb)
                    << " MB), effective_threshold=" << max_mem << " (" << (max_mem / kBytesPerMb) << " MB)"
                    << ", offloaded_count=" << zone_offloaded.size();
      break;
    }
    const size_t best_idx = best_candidate_idx.value();
    const auto &best_cand = all_candidates[best_idx];
    TryProcessBestCandidate(best_cand, node_to_index, target_zone, &zone_profile, &zone_offloaded, &zone_results);
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
}

void ProcessExceedStep(size_t i, size_t mem_val, bool *in_zone, size_t *start, size_t *peak_mem, size_t *peak_idx) {
  if (!*in_zone) {
    *in_zone = true;
    *start = i;
    *peak_mem = mem_val;
    *peak_idx = i;
  } else if (mem_val > *peak_mem) {
    *peak_mem = mem_val;
    *peak_idx = i;
  }
}

std::vector<OffloadZone> IdentifyExceedZones(const std::vector<size_t> &memory_profile, size_t max_device_memory) {
  std::vector<OffloadZone> zones;
  bool in_zone = false;
  size_t start = 0;
  size_t peak_mem = 0;
  size_t peak_idx = 0;

  for (size_t i = 0; i < memory_profile.size(); ++i) {
    if (memory_profile[i] > max_device_memory) {
      ProcessExceedStep(i, memory_profile[i], &in_zone, &start, &peak_mem, &peak_idx);
      continue;
    }
    if (in_zone) {
      in_zone = false;
      zones.push_back(OffloadZone{start, i, peak_idx, peak_mem});
    }
  }
  if (in_zone) {
    zones.push_back(OffloadZone{start, memory_profile.size(), peak_idx, peak_mem});
  }
  return zones;
}

std::vector<OffloadZone> MergeAdjacentOffloadZones(const std::vector<OffloadZone> &zones, size_t merge_gap) {
  if (zones.empty()) {
    return zones;
  }
  std::vector<OffloadZone> merged;
  merged.push_back(zones[0]);
  for (size_t i = 1; i < zones.size(); ++i) {
    OffloadZone &last = merged.back();
    if (zones[i].start_index <= last.end_index + merge_gap) {
      last.end_index = zones[i].end_index;
      if (zones[i].peak_memory > last.peak_memory) {
        last.peak_memory = zones[i].peak_memory;
        last.peak_index = zones[i].peak_index;
      }
    } else {
      merged.push_back(zones[i]);
    }
  }
  return merged;
}

std::vector<OffloadZone> FindOffloadZones(const std::vector<size_t> &memory_profile, size_t max_device_memory,
                                          size_t merge_gap) {
  std::vector<OffloadZone> zones = IdentifyExceedZones(memory_profile, max_device_memory);
  return MergeAdjacentOffloadZones(zones, merge_gap);
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
  size_t max_mem = static_cast<size_t>(configured_max * kMemoryThresholdRatio);
  size_t parameter_memory = hyper_offload::ComputeTotalParameterMemory(graph);

  const CNodePtrList *effective_order = &execution_order;
  CNodePtrList mem_reuse_order_storage;
  auto kg = graph != nullptr ? graph->cast<KernelGraphPtr>() : nullptr;
  if (kg != nullptr && kg->subgraph_multi_call()) {
    const auto &mem_reuse = kg->mem_reuse_exec_order();
    if (!mem_reuse.empty()) {
      mem_reuse_order_storage = mem_reuse;
      effective_order = &mem_reuse_order_storage;
    }
  }

  std::map<AnfNodePtr, size_t> node_to_index;
  BuildNodeToIndexMap(*effective_order, &node_to_index);

  std::vector<size_t> memory_profile;
  std::vector<PeakEstimateCandidate> all_candidates;
  hyper_offload::BuildMemoryProfileAndCandidates(user_info_list, node_to_index, effective_order->size(),
                                                 parameter_memory, &memory_profile, &all_candidates, *effective_order,
                                                 graph);

  std::vector<OffloadZone> zones = FindOffloadZones(memory_profile, max_mem, kOffloadZoneMergeMaxGap);
  if (zones.empty()) {
    return offload_info_list;
  }

  std::vector<OffloadInfoList> per_zone_results;
  per_zone_results.reserve(zones.size());
  for (size_t z = 0; z < zones.size(); ++z) {
    per_zone_results.push_back(
      ProcessSingleZone(z, zones[z], memory_profile, all_candidates, max_mem, configured_max, node_to_index));
  }

  MergeZoneResultsIntoList(per_zone_results, &offload_info_list);
  return offload_info_list;
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
