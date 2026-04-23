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
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "include/utils/utils.h"
#include "utils/compile_config.h"
#include "utils/hyper_offload/hyper_offload_mem_estimate.h"
#include "utils/hyper_offload/hyper_offload_strategy.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {

// The configured max memory comes from the actual available memory of the target device.
// A modeled headroom is subtracted as a safety margin to absorb estimation error.
// It also absorbs runtime memory fluctuation. Tune it from the observed gap between
// modeled peak memory and real device memory usage on the target model/device.
constexpr size_t kDefaultPeakModeledHeadroomBytes = 4ULL * 1024 * 1024 * 1024;
// Avoid division by zero when normalizing cover distances.
constexpr size_t kMinScoreDenominator = 1;

struct AbsoluteDistance {
  KernelWithIndex node;
  size_t absolute_distance;
};
using AbsoluteDistanceList = std::vector<AbsoluteDistance>;

struct RelativeDistance {
  KernelWithIndex start_node;
  KernelWithIndex end_node;
};
using RelativeDistanceList = std::vector<RelativeDistance>;
using GapRange = std::pair<size_t, size_t>;
using GapRangeList = std::vector<GapRange>;
using CandidateGapRangeMap = std::map<size_t, GapRangeList>;

struct PeakSelectionInputs {
  const std::vector<PeakEstimateCandidate> &all_candidates;
  const CNodePtrList &index_order;
  const std::map<AnfNodePtr, size_t> &node_to_index;
  size_t profile_size;
  size_t max_mem_bytes;
  const CandidateGapRangeMap &candidate_gap_ranges;
  const std::vector<size_t> &sorted_indices_by_creation;
};

struct PeakSelectionOutputs {
  std::vector<size_t> *working_profile;
  UserInfoList *selected_user_infos;
  std::vector<std::optional<size_t>> *selected_producer_anchors;
  std::set<KernelWithIndex> *successfully_offloaded;
  std::set<KernelWithIndex> *excluded_from_selection;
};

struct PeakCoverageContext {
  const std::vector<PeakEstimateCandidate> &all_candidates;
  const std::set<KernelWithIndex> &excluded_from_selection;
  const CandidateGapRangeMap &candidate_gap_ranges;
};

struct PeakSelectionLoopInputs {
  const std::vector<PeakEstimateCandidate> &all_candidates;
  const CNodePtrList &index_order;
  const std::map<AnfNodePtr, size_t> &node_to_index;
  size_t profile_size;
  size_t max_mem_bytes;
};

struct PeakSelectionLoopOutputs {
  std::vector<size_t> *working_profile;
  UserInfoList *selected_user_infos;
  std::vector<std::optional<size_t>> *selected_producer_anchors;
  std::set<KernelWithIndex> *successfully_offloaded;
};

void BuildNodeToIndexMap(const CNodePtrList &execution_order, std::map<AnfNodePtr, size_t> *node_to_index) {
  MS_EXCEPTION_IF_NULL(node_to_index);
  for (size_t i = 0; i < execution_order.size(); ++i) {
    const auto &node = execution_order[i];
    if (node != nullptr) {
      (*node_to_index)[node] = i;
    }
  }
}

AbsoluteDistanceList GeneratePeakAbsoluteDistanceList(const UserInfo &user_info, const CNodePtrList &execution_order,
                                                      const std::optional<size_t> &producer_exec_anchor) {
  auto node = user_info.node.first;
  size_t node_distance = 0;
  if (producer_exec_anchor.has_value()) {
    node_distance = producer_exec_anchor.value();
    MS_EXCEPTION_IF_CHECK_FAIL(node_distance < execution_order.size(),
                               "producer_exec_anchor out of range for execution order.");
  } else {
    auto node_iter = std::find(execution_order.begin(), execution_order.end(), node);
    MS_EXCEPTION_IF_CHECK_FAIL(node_iter != execution_order.end(), "Failed to find node in execution order.");
    node_distance = static_cast<size_t>(std::distance(execution_order.begin(), node_iter));
  }
  AbsoluteDistanceList absolute_distance_list{AbsoluteDistance{user_info.node, node_distance}};
  for (const auto &node_user : user_info.users) {
    auto user_node_iter = std::find(execution_order.begin(), execution_order.end(), node_user.first);
    if (user_node_iter == execution_order.end()) {
      MS_LOG(EXCEPTION) << "Failed to find in executor for user node: " << node_user.first->DebugString()
                        << ", node is " << node->DebugString();
    }
    size_t absolute_distance = static_cast<size_t>(std::distance(execution_order.begin(), user_node_iter));
    (void)absolute_distance_list.emplace_back(AbsoluteDistance{node_user, absolute_distance});
  }
  std::sort(absolute_distance_list.begin(), absolute_distance_list.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.absolute_distance < rhs.absolute_distance; });
  return absolute_distance_list;
}

RelativeDistanceList GeneratePeakRelativeDistanceList(const AbsoluteDistanceList &absolute_distance_list) {
  RelativeDistanceList relative_distance_list;
  for (size_t i = 1; i < absolute_distance_list.size(); ++i) {
    (void)relative_distance_list.emplace_back(
      RelativeDistance{absolute_distance_list[i - 1].node, absolute_distance_list[i].node});
  }
  return relative_distance_list;
}

ReplaceInfoList GeneratePeakReplaceInfoList(const RelativeDistanceList &relative_distance_list) {
  ReplaceInfoList replace_info_list;
  if (relative_distance_list.empty()) {
    return replace_info_list;
  }
  for (const auto &distance_info : relative_distance_list) {
    const auto &new_need_change_node = distance_info.end_node;
    const auto &upper_bound = distance_info.start_node;
    std::vector<KernelWithIndex> need_change_nodes{new_need_change_node};
    replace_info_list.emplace_back(ReplaceInfo{upper_bound, need_change_nodes});
  }
  MS_EXCEPTION_IF_CHECK_FAIL(!replace_info_list.empty(), "Replace info list should not be empty");
  return replace_info_list;
}

std::optional<OffloadInfo> GeneratePeakOffloadInfo(const UserInfo &user_info, const CNodePtrList &execution_order,
                                                   const std::optional<size_t> &producer_exec_anchor = std::nullopt) {
  if (!IsValidOffloadNode(user_info)) {
    return std::nullopt;
  }
  const auto &absolute_distance_list =
    GeneratePeakAbsoluteDistanceList(user_info, execution_order, producer_exec_anchor);
  if (absolute_distance_list.size() <= 1) {
    return std::nullopt;
  }
  const auto &relative_distance_list = GeneratePeakRelativeDistanceList(absolute_distance_list);
  const auto &replace_info_list = GeneratePeakReplaceInfoList(relative_distance_list);
  if (replace_info_list.empty()) {
    return std::nullopt;
  }
  return OffloadInfo{user_info.node, replace_info_list};
}

OffloadInfoList GeneratePeakOffloadInfoFromUserList(const CNodePtrList &execution_order,
                                                    const UserInfoList &user_info_list,
                                                    const std::vector<std::optional<size_t>> *producer_exec_anchors) {
  OffloadInfoList offload_info_list;
  for (size_t i = 0; i < user_info_list.size(); ++i) {
    const auto &cur_user_info = user_info_list[i];
    std::optional<size_t> anchor;
    if (producer_exec_anchors != nullptr && i < producer_exec_anchors->size()) {
      anchor = (*producer_exec_anchors)[i];
    }
    const auto &offload_info_opt = GeneratePeakOffloadInfo(cur_user_info, execution_order, anchor);
    if (!offload_info_opt.has_value()) {
      continue;
    }
    (void)offload_info_list.emplace_back(offload_info_opt.value());
  }
  return offload_info_list;
}

// Extracts every actual offload gap [start, end) from ReplaceInfo for memory simulation.
std::optional<GapRangeList> GetGapRangesFromOffloadInfo(const OffloadInfo &info,
                                                        const std::map<AnfNodePtr, size_t> &node_to_index) {
  GapRangeList gap_ranges;
  for (const auto &replace_info : info.replace_info_list) {
    if (replace_info.replace_rest_nodes.empty()) {
      continue;
    }
    const AnfNodePtr start_node = replace_info.replace_start_node.first;
    const AnfNodePtr end_node = replace_info.replace_rest_nodes[0].first;
    if (start_node == nullptr || end_node == nullptr) {
      continue;
    }
    auto start_it = node_to_index.find(start_node);
    auto end_it = node_to_index.find(end_node);
    if (start_it == node_to_index.end() || end_it == node_to_index.end()) {
      continue;
    }
    const size_t gap_start = start_it->second + 1;
    const size_t gap_end = end_it->second;
    if (gap_start >= gap_end) {
      continue;
    }
    gap_ranges.emplace_back(gap_start, gap_end);
  }
  if (gap_ranges.empty()) {
    return std::nullopt;
  }
  std::sort(gap_ranges.begin(), gap_ranges.end());
  return gap_ranges;
}

std::optional<GapRangeList> GetCandidateOffloadGaps(const PeakEstimateCandidate &candidate,
                                                    const CNodePtrList &execution_order,
                                                    const std::map<AnfNodePtr, size_t> &node_to_index) {
  UserInfoList single{candidate.user_info};
  std::vector<std::optional<size_t>> anchors{std::optional<size_t>(candidate.creation_index)};
  auto offload_list = GeneratePeakOffloadInfoFromUserList(execution_order, single, &anchors);
  if (offload_list.empty()) {
    return std::nullopt;
  }
  return GetGapRangesFromOffloadInfo(offload_list[0], node_to_index);
}

// Returns (peak_index, peak_value). Uses last occurrence when multiple indices share the maximum.
std::optional<std::pair<size_t, size_t>> FindGlobalPeak(const std::vector<size_t> &memory_profile) {
  if (memory_profile.empty()) {
    return std::nullopt;
  }
  size_t peak_idx = 0;
  size_t peak_val = memory_profile[0];
  for (size_t i = 1; i < memory_profile.size(); ++i) {
    if (memory_profile[i] >= peak_val) {
      peak_val = memory_profile[i];
      peak_idx = i;
    }
  }
  return std::make_pair(peak_idx, peak_val);
}

size_t GetPeakModeledHeadroomBytes() {
  static const size_t peak_modeled_headroom_bytes = []() -> size_t {
    const std::string value = common::GetCompileConfig("HYPER_OFFLOAD_PEAK_MODELED_HEADROOM_BYTES");
    if (value.empty()) {
      return kDefaultPeakModeledHeadroomBytes;
    }
    try {
      const auto parsed = std::stoull(value, nullptr, 10);
      if (parsed > std::numeric_limits<size_t>::max()) {
        MS_LOG(WARNING) << "[HyperOffload] invalid HYPER_OFFLOAD_PEAK_MODELED_HEADROOM_BYTES=" << value
                        << ", use default " << kDefaultPeakModeledHeadroomBytes;
        return kDefaultPeakModeledHeadroomBytes;
      }
      return static_cast<size_t>(parsed);
    } catch (const std::exception &) {
      MS_LOG(WARNING) << "[HyperOffload] failed to parse HYPER_OFFLOAD_PEAK_MODELED_HEADROOM_BYTES=" << value
                      << ", use default " << kDefaultPeakModeledHeadroomBytes;
      return kDefaultPeakModeledHeadroomBytes;
    }
  }();
  return peak_modeled_headroom_bytes;
}

CandidateGapRangeMap BuildCandidateGapRangeMap(const std::vector<PeakEstimateCandidate> &all_candidates,
                                               const CNodePtrList &execution_order,
                                               const std::map<AnfNodePtr, size_t> &node_to_index) {
  CandidateGapRangeMap result;
  for (size_t i = 0; i < all_candidates.size(); ++i) {
    auto gap_ranges = GetCandidateOffloadGaps(all_candidates[i], execution_order, node_to_index);
    if (gap_ranges.has_value()) {
      result.emplace(i, std::move(gap_ranges.value()));
    }
  }
  return result;
}

std::vector<size_t> BuildSortedIndicesByCreation(const std::vector<PeakEstimateCandidate> &all_candidates) {
  std::vector<size_t> indices(all_candidates.size());
  for (size_t i = 0; i < all_candidates.size(); ++i) {
    indices[i] = i;
  }
  std::sort(indices.begin(), indices.end(), [&all_candidates](size_t a, size_t b) {
    return all_candidates[a].creation_index < all_candidates[b].creation_index;
  });
  return indices;
}

std::vector<std::pair<size_t, size_t>> BuildOffloadedLifecycles(
  const std::vector<PeakEstimateCandidate> &all_candidates, const std::set<KernelWithIndex> &successfully_offloaded) {
  std::vector<std::pair<size_t, size_t>> lifecycles;
  lifecycles.reserve(successfully_offloaded.size());
  for (const auto &candidate : all_candidates) {
    if (successfully_offloaded.count(candidate.user_info.node) != 0) {
      lifecycles.emplace_back(candidate.creation_index, candidate.last_usage_index);
    }
  }
  return lifecycles;
}

double ComputeNormalizedNeighborImpact(const PeakEstimateCandidate &candidate,
                                       const std::vector<std::pair<size_t, size_t>> &offloaded_lifecycles) {
  if (candidate.last_usage_index <= candidate.creation_index || offloaded_lifecycles.empty()) {
    return 0.0;
  }

  std::vector<std::pair<size_t, size_t>> clipped_ranges;
  clipped_ranges.reserve(offloaded_lifecycles.size());
  for (const auto &lifecycle : offloaded_lifecycles) {
    const size_t overlap_start = std::max(candidate.creation_index, lifecycle.first);
    const size_t overlap_end = std::min(candidate.last_usage_index, lifecycle.second);
    if (overlap_start < overlap_end) {
      clipped_ranges.emplace_back(overlap_start, overlap_end);
    }
  }
  if (clipped_ranges.empty()) {
    return 0.0;
  }

  std::sort(clipped_ranges.begin(), clipped_ranges.end());
  size_t merged_overlap = 0;
  size_t current_start = clipped_ranges.front().first;
  size_t current_end = clipped_ranges.front().second;
  for (size_t i = 1; i < clipped_ranges.size(); ++i) {
    const auto &[range_start, range_end] = clipped_ranges[i];
    if (range_start <= current_end) {
      current_end = std::max(current_end, range_end);
      continue;
    }
    merged_overlap += current_end - current_start;
    current_start = range_start;
    current_end = range_end;
  }
  merged_overlap += current_end - current_start;

  const size_t candidate_span = candidate.last_usage_index - candidate.creation_index;
  const double denom_neighbor = static_cast<double>(std::max(kMinScoreDenominator, candidate_span));
  return static_cast<double>(merged_overlap) / denom_neighbor;
}

bool GapRangeCoversPeak(const GapRange &gap_range, size_t peak_index) {
  return gap_range.first <= peak_index && peak_index < gap_range.second;
}

bool GapRangesCoverPeak(const GapRangeList &gap_ranges, size_t peak_index) {
  return std::any_of(gap_ranges.begin(), gap_ranges.end(),
                     [peak_index](const auto &gap_range) { return GapRangeCoversPeak(gap_range, peak_index); });
}

std::optional<GapRange> FindPeakCoveringGap(const GapRangeList &gap_ranges, size_t peak_index) {
  auto it = std::find_if(gap_ranges.begin(), gap_ranges.end(),
                         [peak_index](const auto &gap_range) { return GapRangeCoversPeak(gap_range, peak_index); });
  if (it == gap_ranges.end()) {
    return std::nullopt;
  }
  return *it;
}

bool HasCoveringCandidate(const std::vector<PeakEstimateCandidate> &all_candidates,
                          const std::set<KernelWithIndex> &excluded_from_selection, size_t peak_index,
                          const CandidateGapRangeMap &candidate_gap_ranges) {
  for (size_t i = 0; i < all_candidates.size(); ++i) {
    if (excluded_from_selection.count(all_candidates[i].user_info.node) != 0) {
      continue;
    }
    auto gap_it = candidate_gap_ranges.find(i);
    if (gap_it == candidate_gap_ranges.end()) {
      continue;
    }
    if (GapRangesCoverPeak(gap_it->second, peak_index)) {
      return true;
    }
  }
  return false;
}

size_t FindPeakPlateauStart(const std::vector<size_t> &working_profile, size_t right_peak_idx, size_t peak_value) {
  size_t left_plateau = right_peak_idx;
  while (left_plateau > 0 && working_profile[left_plateau - 1] == peak_value) {
    left_plateau--;
  }
  return left_plateau;
}

size_t FindEffectivePeakIndexInPlateau(const PeakCoverageContext &coverage_context, size_t left_plateau,
                                       size_t right_peak_idx) {
  size_t effective = right_peak_idx;
  while (effective > left_plateau &&
         !HasCoveringCandidate(coverage_context.all_candidates, coverage_context.excluded_from_selection, effective,
                               coverage_context.candidate_gap_ranges)) {
    effective--;
  }
  return effective;
}

size_t EffectivePeakIndexForSelection(const std::vector<size_t> &working_profile, size_t right_peak_idx,
                                      size_t peak_value, const PeakCoverageContext &coverage_context) {
  const size_t left_plateau = FindPeakPlateauStart(working_profile, right_peak_idx, peak_value);
  return FindEffectivePeakIndexInPlateau(coverage_context, left_plateau, right_peak_idx);
}

std::optional<size_t> SelectBestCandidateIdx(const std::vector<PeakEstimateCandidate> &all_candidates,
                                             const std::set<KernelWithIndex> &excluded_from_selection,
                                             const std::set<KernelWithIndex> &successfully_offloaded, size_t peak_index,
                                             size_t profile_size, const CandidateGapRangeMap &candidate_gap_ranges,
                                             const std::vector<size_t> &sorted_indices_by_creation) {
  const auto &config = HyperOffloadConfig::GetInstance();
  const double weight_size = config.score_weight_size();
  const double weight_cover_start = config.score_weight_cover_start();
  const double weight_cover_end = config.score_weight_cover_end();
  const double weight_neighbor_impact = config.score_weight_neighbor_impact();
  double best_score = std::numeric_limits<double>::lowest();
  std::optional<size_t> best_candidate_idx;

  const auto offloaded_lifecycles = BuildOffloadedLifecycles(all_candidates, successfully_offloaded);
  const double denom_start = static_cast<double>(std::max(kMinScoreDenominator, peak_index));
  const double denom_end =
    static_cast<double>(std::max(kMinScoreDenominator, profile_size > 0 ? profile_size - 1 - peak_index : size_t{0}));
  size_t max_candidate_size = 0;
  for (size_t idx : sorted_indices_by_creation) {
    const auto &candidate = all_candidates[idx];
    if (candidate.creation_index >= peak_index) {
      break;
    }
    if (excluded_from_selection.count(candidate.user_info.node) != 0) {
      continue;
    }
    auto gap_it = candidate_gap_ranges.find(idx);
    if (gap_it == candidate_gap_ranges.end()) {
      continue;
    }
    if (!GapRangesCoverPeak(gap_it->second, peak_index)) {
      continue;
    }
    max_candidate_size = std::max(max_candidate_size, candidate.size);
  }
  const double denom_size = static_cast<double>(std::max(kMinScoreDenominator, max_candidate_size));

  for (size_t idx : sorted_indices_by_creation) {
    const auto &candidate = all_candidates[idx];
    if (candidate.creation_index >= peak_index) {
      break;
    }
    if (excluded_from_selection.count(candidate.user_info.node)) {
      continue;
    }
    auto gap_it = candidate_gap_ranges.find(idx);
    if (gap_it == candidate_gap_ranges.end()) {
      continue;
    }
    const auto covering_gap = FindPeakCoveringGap(gap_it->second, peak_index);
    if (!covering_gap.has_value()) {
      continue;
    }
    const double norm_size = static_cast<double>(candidate.size) / denom_size;
    const size_t cover_start_raw = peak_index - covering_gap->first;
    const size_t cover_end_raw = (covering_gap->second - 1) - peak_index;
    const double norm_cover_start = static_cast<double>(cover_start_raw) / denom_start;
    const double norm_cover_end = static_cast<double>(cover_end_raw) / denom_end;
    const double norm_neighbor = ComputeNormalizedNeighborImpact(candidate, offloaded_lifecycles);
    const double score = weight_size * norm_size + weight_cover_start * norm_cover_start +
                         weight_cover_end * norm_cover_end - weight_neighbor_impact * norm_neighbor;
    if (score > best_score) {
      best_score = score;
      best_candidate_idx = idx;
    }
  }
  return best_candidate_idx;
}

// Merge ReplaceInfo lists per data_node so one D2H + shared host buffer matches distance strategy.
OffloadInfoList MergeOffloadInfosByDataNode(const OffloadInfoList &list) {
  if (list.size() <= 1) {
    return list;
  }
  std::vector<KernelWithIndex> first_seen_order;
  std::map<KernelWithIndex, ReplaceInfoList> by_data_node;
  for (const auto &info : list) {
    auto it = by_data_node.find(info.data_node);
    if (it == by_data_node.end()) {
      (void)first_seen_order.emplace_back(info.data_node);
      by_data_node[info.data_node] = info.replace_info_list;
    } else {
      auto &replace_infos = it->second;
      replace_infos.insert(replace_infos.end(), info.replace_info_list.begin(), info.replace_info_list.end());
    }
  }
  OffloadInfoList merged_offload_info_list;
  merged_offload_info_list.reserve(first_seen_order.size());
  for (const auto &data_node : first_seen_order) {
    (void)merged_offload_info_list.emplace_back(OffloadInfo{data_node, by_data_node[data_node]});
  }
  return merged_offload_info_list;
}

void ReduceWorkingProfileForGaps(const GapRangeList &gap_ranges, size_t reduce_size,
                                 std::vector<size_t> *working_profile) {
  MS_EXCEPTION_IF_NULL(working_profile);
  for (const auto &[gap_start, gap_end] : gap_ranges) {
    hyper_offload::ReduceMemoryProfile(gap_start, gap_end, reduce_size, working_profile);
  }
}

void RecordSelectedOffload(const PeakEstimateCandidate &candidate, UserInfoList *selected_user_infos,
                           std::vector<std::optional<size_t>> *selected_producer_anchors) {
  MS_EXCEPTION_IF_NULL(selected_user_infos);
  selected_user_infos->push_back(candidate.user_info);
  if (selected_producer_anchors != nullptr) {
    selected_producer_anchors->push_back(candidate.creation_index);
  }
}

bool ApplySingleOffload(const PeakEstimateCandidate &best_cand, const PeakSelectionInputs &inputs,
                        const PeakSelectionOutputs &outputs) {
  MS_EXCEPTION_IF_NULL(outputs.working_profile);
  MS_EXCEPTION_IF_NULL(outputs.selected_user_infos);
  const auto gap_ranges = GetCandidateOffloadGaps(best_cand, inputs.index_order, inputs.node_to_index);
  if (!gap_ranges.has_value()) {
    return false;
  }
  ReduceWorkingProfileForGaps(gap_ranges.value(), best_cand.size, outputs.working_profile);
  RecordSelectedOffload(best_cand, outputs.selected_user_infos, outputs.selected_producer_anchors);
  return true;
}

size_t EffectiveMemoryThresholdBytes(size_t configured_max_bytes) {
  const size_t peak_modeled_headroom_bytes = GetPeakModeledHeadroomBytes();
  if (configured_max_bytes <= peak_modeled_headroom_bytes) {
    return 0;
  }
  return configured_max_bytes - peak_modeled_headroom_bytes;
}

void ValidatePeakSelectionOutputs(const PeakSelectionOutputs &outputs) {
  MS_EXCEPTION_IF_NULL(outputs.working_profile);
  MS_EXCEPTION_IF_NULL(outputs.selected_user_infos);
  MS_EXCEPTION_IF_NULL(outputs.selected_producer_anchors);
  MS_EXCEPTION_IF_NULL(outputs.successfully_offloaded);
  MS_EXCEPTION_IF_NULL(outputs.excluded_from_selection);
}

std::optional<size_t> FindPeakIndexForSelection(const PeakSelectionInputs &inputs,
                                                const PeakSelectionOutputs &outputs) {
  const auto peak_info_opt = FindGlobalPeak(*outputs.working_profile);
  if (!peak_info_opt.has_value() || peak_info_opt->second <= inputs.max_mem_bytes) {
    return std::nullopt;
  }
  const PeakCoverageContext coverage_context{inputs.all_candidates, *outputs.excluded_from_selection,
                                             inputs.candidate_gap_ranges};
  return EffectivePeakIndexForSelection(*outputs.working_profile, peak_info_opt->first, peak_info_opt->second,
                                        coverage_context);
}

std::optional<size_t> SelectBestCandidateForPeak(const PeakSelectionInputs &inputs, const PeakSelectionOutputs &outputs,
                                                 size_t peak_index) {
  return SelectBestCandidateIdx(inputs.all_candidates, *outputs.excluded_from_selection,
                                *outputs.successfully_offloaded, peak_index, inputs.profile_size,
                                inputs.candidate_gap_ranges, inputs.sorted_indices_by_creation);
}

void ApplySelectedPeakCandidate(size_t best_candidate_idx, const PeakSelectionInputs &inputs,
                                const PeakSelectionOutputs &outputs) {
  const auto &best_candidate = inputs.all_candidates[best_candidate_idx];
  if (ApplySingleOffload(best_candidate, inputs, outputs)) {
    (void)outputs.successfully_offloaded->insert(best_candidate.user_info.node);
  }
  (void)outputs.excluded_from_selection->insert(best_candidate.user_info.node);
}

bool RunPeakOffloadSelectionRound(const PeakSelectionInputs &inputs, const PeakSelectionOutputs &outputs) {
  ValidatePeakSelectionOutputs(outputs);
  const auto peak_index = FindPeakIndexForSelection(inputs, outputs);
  if (!peak_index.has_value()) {
    return false;
  }
  const auto best_candidate_idx = SelectBestCandidateForPeak(inputs, outputs, peak_index.value());
  if (!best_candidate_idx.has_value()) {
    return false;
  }
  ApplySelectedPeakCandidate(best_candidate_idx.value(), inputs, outputs);
  return true;
}

void RunPeakOffloadSelectionRounds(const PeakSelectionInputs &inputs, const PeakSelectionOutputs &outputs) {
  while (RunPeakOffloadSelectionRound(inputs, outputs)) {
  }
}

void ValidatePeakSelectionLoopOutputs(const PeakSelectionLoopOutputs &outputs) {
  MS_EXCEPTION_IF_NULL(outputs.working_profile);
  MS_EXCEPTION_IF_NULL(outputs.selected_user_infos);
  MS_EXCEPTION_IF_NULL(outputs.selected_producer_anchors);
  MS_EXCEPTION_IF_NULL(outputs.successfully_offloaded);
}

void RunPeakOffloadSelectionWithCaches(const PeakSelectionInputs &inputs,
                                       const PeakSelectionLoopOutputs &loop_outputs) {
  std::set<KernelWithIndex> excluded_from_selection;
  const PeakSelectionOutputs outputs{loop_outputs.working_profile, loop_outputs.selected_user_infos,
                                     loop_outputs.selected_producer_anchors, loop_outputs.successfully_offloaded,
                                     &excluded_from_selection};
  RunPeakOffloadSelectionRounds(inputs, outputs);
}

// Runs iterative peak reduction; returns selected UserInfo entries and updated working_profile.
void RunPeakOffloadSelectionLoop(const PeakSelectionLoopInputs &loop_inputs,
                                 const PeakSelectionLoopOutputs &loop_outputs) {
  ValidatePeakSelectionLoopOutputs(loop_outputs);
  const auto candidate_gap_ranges =
    BuildCandidateGapRangeMap(loop_inputs.all_candidates, loop_inputs.index_order, loop_inputs.node_to_index);
  const auto sorted_indices_by_creation = BuildSortedIndicesByCreation(loop_inputs.all_candidates);
  const PeakSelectionInputs inputs{loop_inputs.all_candidates, loop_inputs.index_order,   loop_inputs.node_to_index,
                                   loop_inputs.profile_size,   loop_inputs.max_mem_bytes, candidate_gap_ranges,
                                   sorted_indices_by_creation};
  RunPeakOffloadSelectionWithCaches(inputs, loop_outputs);
}

void LogPeakReductionFailureIfNeeded(const std::optional<std::pair<size_t, size_t>> &initial_peak_opt,
                                     const std::optional<std::pair<size_t, size_t>> &final_peak_opt, size_t max_mem,
                                     const CNodePtrList &execution_order, const UserInfoList &selected_user_infos) {
  if (!final_peak_opt.has_value() || final_peak_opt->second <= max_mem) {
    return;
  }
  std::string peak_operator = "[profile_padding]";
  size_t workspace_pulse = 0;
  if (final_peak_opt->first < execution_order.size()) {
    const auto &peak_node = execution_order[final_peak_opt->first];
    if (peak_node != nullptr) {
      peak_operator = peak_node->fullname_with_scope();
      if (peak_operator.empty()) {
        peak_operator = peak_node->DebugString();
      }
      workspace_pulse = GetNodeWorkspacePulseForProfile(peak_node);
    }
  }
  MS_LOG(WARNING) << "[HyperOffload] peak strategy could not reduce modeled peak to target threshold. "
                  << "initial_peak=" << initial_peak_opt->second << " bytes, final_peak=" << final_peak_opt->second
                  << " bytes, threshold=" << max_mem << " bytes, selected_candidates=" << selected_user_infos.size()
                  << ", peak_index=" << final_peak_opt->first << ", peak_operator=" << peak_operator
                  << ", workspace_pulse=" << workspace_pulse << " bytes";
}

}  // namespace

OffloadInfoList MemoryPeakBasedHyperOffloadStrategy::Run(const CNodePtrList &execution_order,
                                                         const UserInfoList &user_info_list,
                                                         const FuncGraphPtr &graph) {
  OffloadInfoList offload_info_list;
  if (execution_order.empty()) {
    return offload_info_list;
  }

  std::map<AnfNodePtr, size_t> node_to_index;
  BuildNodeToIndexMap(execution_order, &node_to_index);

  std::vector<size_t> memory_profile;
  std::vector<PeakEstimateCandidate> all_candidates;
  const size_t parameter_memory_baseline = ComputeTotalParameterMemory(graph);
  // Build the modeled profile before evaluating the max-device-memory threshold.
  BuildMemoryProfileAndCandidates(user_info_list, node_to_index, execution_order.size(), parameter_memory_baseline,
                                  &memory_profile, &all_candidates, execution_order, graph);

  const size_t configured_max = HyperOffloadConfig::GetInstance().max_device_memory();
  if (configured_max == 0) {
    MS_LOG(WARNING) << "[HyperOffload] peak strategy enabled but HYPER_OFFLOAD_MAX_DEVICE_MEMORY is 0, skip peak "
                    << "offload selection.";
    return offload_info_list;
  }
  const size_t max_mem = EffectiveMemoryThresholdBytes(configured_max);

  const auto initial_peak_opt = FindGlobalPeak(memory_profile);
  if (!initial_peak_opt.has_value() || initial_peak_opt->second <= max_mem) {
    return offload_info_list;
  }

  const size_t profile_size = memory_profile.size();
  std::vector<size_t> working_profile = memory_profile;
  UserInfoList selected_user_infos;
  std::vector<std::optional<size_t>> selected_producer_anchors;
  std::set<KernelWithIndex> successfully_offloaded;

  const PeakSelectionLoopInputs selection_inputs{all_candidates, execution_order, node_to_index, profile_size, max_mem};
  const PeakSelectionLoopOutputs selection_outputs{&working_profile, &selected_user_infos, &selected_producer_anchors,
                                                   &successfully_offloaded};
  RunPeakOffloadSelectionLoop(selection_inputs, selection_outputs);

  const auto final_peak_opt = FindGlobalPeak(working_profile);
  LogPeakReductionFailureIfNeeded(initial_peak_opt, final_peak_opt, max_mem, execution_order, selected_user_infos);

  offload_info_list =
    GeneratePeakOffloadInfoFromUserList(execution_order, selected_user_infos, &selected_producer_anchors);
  offload_info_list = MergeOffloadInfosByDataNode(offload_info_list);

  return offload_info_list;
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
