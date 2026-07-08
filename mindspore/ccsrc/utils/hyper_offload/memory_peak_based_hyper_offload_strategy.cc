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
#include <unordered_set>
#include <utility>
#include <vector>

#include "include/utils/utils.h"
#include "utils/compile_config.h"
#include "utils/hyper_offload/hyper_offload_mem_estimate.h"
#include "utils/hyper_offload/hyper_offload_strategy.h"
#include "utils/hyper_offload/hyper_offload_utils.h"
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
  std::vector<GapRangeList> *selected_gap_ranges;
  std::vector<size_t> *selected_peak_indices;
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
  std::vector<GapRangeList> *selected_gap_ranges;
  std::vector<size_t> *selected_peak_indices;
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

std::vector<KernelWithIndex> FilterUsersInExecutionOrder(const UserInfo &user_info,
                                                         const std::map<AnfNodePtr, size_t> &node_to_index) {
  std::vector<KernelWithIndex> users;
  std::unordered_set<AnfNodePtr> user_nodes;
  auto append_user = [&users, &user_nodes, &node_to_index](const KernelWithIndex &user) {
    if (user.first == nullptr) {
      return;
    }
    if (node_to_index.find(user.first) == node_to_index.end()) {
      return;
    }
    if (user_nodes.insert(user.first).second) {
      users.emplace_back(user);
    }
  };
  for (const auto &user : user_info.users) {
    append_user(user);
  }
  return users;
}

AbsoluteDistanceList GeneratePeakAbsoluteDistanceList(const UserInfo &user_info, const CNodePtrList &execution_order,
                                                      const std::map<AnfNodePtr, size_t> &node_to_index,
                                                      const std::optional<size_t> &producer_exec_anchor) {
  auto node = user_info.node.first;
  size_t node_distance = 0;
  if (producer_exec_anchor.has_value()) {
    node_distance = producer_exec_anchor.value();
    MS_EXCEPTION_IF_CHECK_FAIL(node_distance < execution_order.size(),
                               "producer_exec_anchor out of range for execution order.");
  } else {
    auto node_iter = node_to_index.find(node);
    MS_EXCEPTION_IF_CHECK_FAIL(node_iter != node_to_index.end(), "Failed to find node in execution order.");
    node_distance = node_iter->second;
  }
  AbsoluteDistanceList absolute_distance_list{AbsoluteDistance{user_info.node, node_distance}};
  for (const auto &node_user : FilterUsersInExecutionOrder(user_info, node_to_index)) {
    auto user_node_iter = node_to_index.find(node_user.first);
    if (user_node_iter == node_to_index.end()) {
      MS_LOG(EXCEPTION) << "Failed to find in executor for user node: " << node_user.first->DebugString()
                        << ", node is " << node->DebugString();
    }
    size_t absolute_distance = user_node_iter->second;
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
    replace_info_list.emplace_back(ReplaceInfo{upper_bound, need_change_nodes, upper_bound});
  }
  MS_EXCEPTION_IF_CHECK_FAIL(!replace_info_list.empty(), "Replace info list should not be empty");
  return replace_info_list;
}

std::optional<OffloadInfo> GeneratePeakOffloadInfo(const UserInfo &user_info, const CNodePtrList &execution_order,
                                                   const std::map<AnfNodePtr, size_t> &node_to_index,
                                                   const std::optional<size_t> &producer_exec_anchor = std::nullopt) {
  if (!IsValidOffloadNode(user_info)) {
    return std::nullopt;
  }
  const auto &absolute_distance_list =
    GeneratePeakAbsoluteDistanceList(user_info, execution_order, node_to_index, producer_exec_anchor);
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
  const auto offload_info =
    GeneratePeakOffloadInfo(candidate.user_info, execution_order, node_to_index, candidate.creation_index);
  if (!offload_info.has_value()) {
    return std::nullopt;
  }
  return GetGapRangesFromOffloadInfo(offload_info.value(), node_to_index);
}

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

size_t FindFirstUseAfterPeak(const PeakEstimateCandidate &candidate, size_t peak_index, size_t profile_size,
                             const std::map<AnfNodePtr, size_t> &node_to_index) {
  size_t first_use_after_peak = profile_size;
  for (const auto &user : candidate.user_info.users) {
    auto user_it = node_to_index.find(user.first);
    if (user_it != node_to_index.end() && user_it->second > peak_index) {
      first_use_after_peak = std::min(first_use_after_peak, user_it->second);
    }
  }
  return first_use_after_peak;
}

bool IsCandidateEligible(const PeakEstimateCandidate &candidate, size_t idx, size_t peak_index,
                         const std::set<KernelWithIndex> &excluded_from_selection,
                         const CandidateGapRangeMap &candidate_gap_ranges) {
  if (candidate.creation_index >= peak_index) {
    return false;
  }
  if (excluded_from_selection.count(candidate.user_info.node) != 0) {
    return false;
  }
  auto gap_it = candidate_gap_ranges.find(idx);
  if (gap_it == candidate_gap_ranges.end()) {
    return false;
  }
  return GapRangesCoverPeak(gap_it->second, peak_index);
}

struct CandidateScoreNormalizers {
  double denom_size;
  double denom_start;
  double denom_end;
};

struct CandidateSelectionCommonInputs {
  const std::vector<PeakEstimateCandidate> &all_candidates;
  size_t profile_size;
  const CandidateGapRangeMap &candidate_gap_ranges;
  const std::vector<size_t> &sorted_indices_by_creation;
  const std::map<AnfNodePtr, size_t> &node_to_index;
};

struct CandidateScoreNormalizerRawValues {
  size_t max_candidate_size{0};
  size_t max_cover_start_raw{0};
  size_t max_cover_end_raw{0};
};

void UpdateCandidateScoreNormalizerRawValues(const PeakEstimateCandidate &candidate, size_t peak_index,
                                             size_t profile_size, const std::map<AnfNodePtr, size_t> &node_to_index,
                                             CandidateScoreNormalizerRawValues *raw_values) {
  MS_EXCEPTION_IF_NULL(raw_values);
  raw_values->max_candidate_size = std::max(raw_values->max_candidate_size, candidate.size);
  raw_values->max_cover_start_raw = std::max(raw_values->max_cover_start_raw, peak_index - candidate.creation_index);

  size_t first_use_after_peak = FindFirstUseAfterPeak(candidate, peak_index, profile_size, node_to_index);
  if (first_use_after_peak < profile_size) {
    raw_values->max_cover_end_raw = std::max(raw_values->max_cover_end_raw, first_use_after_peak - peak_index);
  }
}

CandidateScoreNormalizers BuildCandidateScoreNormalizers(const CandidateScoreNormalizerRawValues &raw_values) {
  return CandidateScoreNormalizers{static_cast<double>(std::max(kMinScoreDenominator, raw_values.max_candidate_size)),
                                   static_cast<double>(std::max(kMinScoreDenominator, raw_values.max_cover_start_raw)),
                                   static_cast<double>(std::max(kMinScoreDenominator, raw_values.max_cover_end_raw))};
}

CandidateScoreNormalizers ComputeScoreNormalizers(const CandidateSelectionCommonInputs &common_inputs,
                                                  const std::set<KernelWithIndex> &excluded_from_selection,
                                                  size_t peak_index) {
  CandidateScoreNormalizerRawValues raw_values;

  for (size_t idx : common_inputs.sorted_indices_by_creation) {
    const auto &candidate = common_inputs.all_candidates[idx];
    if (!IsCandidateEligible(candidate, idx, peak_index, excluded_from_selection, common_inputs.candidate_gap_ranges)) {
      continue;
    }
    UpdateCandidateScoreNormalizerRawValues(candidate, peak_index, common_inputs.profile_size,
                                            common_inputs.node_to_index, &raw_values);
  }

  return BuildCandidateScoreNormalizers(raw_values);
}

struct CandidateScoreWeights {
  double size;
  double cover_start;
  double cover_end;
  double neighbor_impact;
};

CandidateScoreWeights GetCandidateScoreWeights() {
  const auto &config = HyperOffloadConfig::GetInstance();
  return CandidateScoreWeights{config.score_weight_size(), config.score_weight_cover_start(),
                               config.score_weight_cover_end(), config.score_weight_neighbor_impact()};
}

struct CandidateScoreContext {
  size_t peak_index;
  size_t profile_size;
  const CandidateScoreNormalizers &normalizers;
  const std::vector<std::pair<size_t, size_t>> &offloaded_lifecycles;
  const CandidateScoreWeights &weights;
  const std::map<AnfNodePtr, size_t> &node_to_index;
};

double ComputeCandidateScore(const PeakEstimateCandidate &candidate, const CandidateScoreContext &context) {
  const double norm_size = static_cast<double>(candidate.size) / context.normalizers.denom_size;
  const size_t cover_start_raw = context.peak_index - candidate.creation_index;
  const double norm_cover_start = static_cast<double>(cover_start_raw) / context.normalizers.denom_start;

  size_t first_use_after_peak =
    FindFirstUseAfterPeak(candidate, context.peak_index, context.profile_size, context.node_to_index);
  const size_t cover_end_raw =
    (first_use_after_peak < context.profile_size) ? (first_use_after_peak - context.peak_index) : 0;
  const double norm_cover_end = static_cast<double>(cover_end_raw) / context.normalizers.denom_end;
  const double norm_neighbor = ComputeNormalizedNeighborImpact(candidate, context.offloaded_lifecycles);

  return context.weights.size * norm_size + context.weights.cover_start * norm_cover_start +
         context.weights.cover_end * norm_cover_end - context.weights.neighbor_impact * norm_neighbor;
}

std::optional<size_t> SelectBestCandidateIdx(const CandidateSelectionCommonInputs &common_inputs,
                                             const std::set<KernelWithIndex> &excluded_from_selection,
                                             const std::set<KernelWithIndex> &successfully_offloaded,
                                             size_t peak_index) {
  const auto weights = GetCandidateScoreWeights();
  const auto offloaded_lifecycles = BuildOffloadedLifecycles(common_inputs.all_candidates, successfully_offloaded);
  const auto normalizers = ComputeScoreNormalizers(common_inputs, excluded_from_selection, peak_index);
  const CandidateScoreContext score_context{peak_index, common_inputs.profile_size, normalizers, offloaded_lifecycles,
                                            weights,    common_inputs.node_to_index};

  double best_score = std::numeric_limits<double>::lowest();
  std::optional<size_t> best_candidate_idx;

  for (size_t idx : common_inputs.sorted_indices_by_creation) {
    const auto &candidate = common_inputs.all_candidates[idx];
    if (candidate.creation_index >= peak_index) {
      break;
    }
    if (excluded_from_selection.count(candidate.user_info.node)) {
      continue;
    }
    auto gap_it = common_inputs.candidate_gap_ranges.find(idx);
    if (gap_it == common_inputs.candidate_gap_ranges.end()) {
      continue;
    }
    const auto covering_gap = FindPeakCoveringGap(gap_it->second, peak_index);
    if (!covering_gap.has_value()) {
      continue;
    }

    const double score = ComputeCandidateScore(candidate, score_context);
    if (score > best_score) {
      best_score = score;
      best_candidate_idx = idx;
    }
  }
  return best_candidate_idx;
}

void ReduceWorkingProfileForGaps(const GapRangeList &gap_ranges, size_t reduce_size,
                                 std::vector<size_t> *working_profile) {
  MS_EXCEPTION_IF_NULL(working_profile);
  for (const auto &[gap_start, gap_end] : gap_ranges) {
    hyper_offload::ReduceMemoryProfile(gap_start, gap_end, reduce_size, working_profile);
  }
}

struct SelectedOffloadRecordInputs {
  const PeakEstimateCandidate &candidate;
  const GapRangeList &gap_ranges;
  size_t peak_index;
  UserInfoList *selected_user_infos;
  std::vector<std::optional<size_t>> *selected_producer_anchors;
  std::vector<GapRangeList> *selected_gap_ranges;
  std::vector<size_t> *selected_peak_indices;
};

void RecordSelectedOffload(const SelectedOffloadRecordInputs &inputs) {
  MS_EXCEPTION_IF_NULL(inputs.selected_user_infos);
  inputs.selected_user_infos->push_back(inputs.candidate.user_info);
  if (inputs.selected_producer_anchors != nullptr) {
    inputs.selected_producer_anchors->push_back(inputs.candidate.creation_index);
  }
  if (inputs.selected_gap_ranges != nullptr) {
    inputs.selected_gap_ranges->push_back(inputs.gap_ranges);
  }
  if (inputs.selected_peak_indices != nullptr) {
    inputs.selected_peak_indices->push_back(inputs.peak_index);
  }
}

void ApplySingleOffload(const PeakEstimateCandidate &best_cand, const PeakSelectionOutputs &outputs,
                        const GapRangeList &gap_ranges, size_t peak_index) {
  MS_EXCEPTION_IF_NULL(outputs.working_profile);
  MS_EXCEPTION_IF_NULL(outputs.selected_user_infos);
  ReduceWorkingProfileForGaps(gap_ranges, best_cand.size, outputs.working_profile);
  const SelectedOffloadRecordInputs record_inputs{best_cand,
                                                  gap_ranges,
                                                  peak_index,
                                                  outputs.selected_user_infos,
                                                  outputs.selected_producer_anchors,
                                                  outputs.selected_gap_ranges,
                                                  outputs.selected_peak_indices};
  RecordSelectedOffload(record_inputs);
  MS_LOG(DEBUG) << "[HyperOffload] selected peak offload candidate, node: "
                << (best_cand.user_info.node.first == nullptr ? "nullptr"
                                                              : best_cand.user_info.node.first->DebugString())
                << ", output_index: " << best_cand.user_info.node.second << ", peak_index: " << peak_index
                << ", candidate_size: " << best_cand.size << ", gap_count: " << gap_ranges.size()
                << ", selected_count: " << outputs.selected_user_infos->size();
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
  MS_EXCEPTION_IF_NULL(outputs.selected_gap_ranges);
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
  const CandidateSelectionCommonInputs common_inputs{inputs.all_candidates, inputs.profile_size,
                                                     inputs.candidate_gap_ranges, inputs.sorted_indices_by_creation,
                                                     inputs.node_to_index};
  return SelectBestCandidateIdx(common_inputs, *outputs.excluded_from_selection, *outputs.successfully_offloaded,
                                peak_index);
}

void ApplySelectedPeakCandidate(const PeakEstimateCandidate &best_candidate, const PeakSelectionOutputs &outputs,
                                const GapRangeList &gap_ranges, size_t peak_index) {
  ApplySingleOffload(best_candidate, outputs, gap_ranges, peak_index);
  (void)outputs.successfully_offloaded->insert(best_candidate.user_info.node);
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
  const auto gap_it = inputs.candidate_gap_ranges.find(best_candidate_idx.value());
  if (gap_it == inputs.candidate_gap_ranges.end()) {
    return false;
  }
  ApplySelectedPeakCandidate(inputs.all_candidates[best_candidate_idx.value()], outputs, gap_it->second,
                             peak_index.value());
  return true;
}

void ValidatePeakSelectionLoopOutputs(const PeakSelectionLoopOutputs &outputs) {
  MS_EXCEPTION_IF_NULL(outputs.working_profile);
  MS_EXCEPTION_IF_NULL(outputs.selected_user_infos);
  MS_EXCEPTION_IF_NULL(outputs.selected_producer_anchors);
  MS_EXCEPTION_IF_NULL(outputs.successfully_offloaded);
  MS_EXCEPTION_IF_NULL(outputs.selected_gap_ranges);
}

void RunPeakOffloadSelectionWithCaches(const PeakSelectionInputs &inputs,
                                       const PeakSelectionLoopOutputs &loop_outputs) {
  std::set<KernelWithIndex> excluded_from_selection;
  const PeakSelectionOutputs outputs{
    loop_outputs.working_profile,        loop_outputs.selected_user_infos, loop_outputs.selected_producer_anchors,
    loop_outputs.successfully_offloaded, &excluded_from_selection,         loop_outputs.selected_gap_ranges,
    loop_outputs.selected_peak_indices};
  while (RunPeakOffloadSelectionRound(inputs, outputs)) {
  }
}

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

size_t GetLastUserIndexForOffloadInfo(const OffloadInfo &info, const std::map<AnfNodePtr, size_t> &node_to_index) {
  size_t last_user_index = 0;

  for (const auto &replace_info : info.replace_info_list) {
    for (const auto &rest_node : replace_info.replace_rest_nodes) {
      auto rest_it = node_to_index.find(rest_node.first);
      if (rest_it != node_to_index.end()) {
        last_user_index = std::max(last_user_index, rest_it->second);
      }
    }
  }

  return last_user_index;
}

size_t GetDataNodeSizeBytes(const AnfNodePtr &data_node) {
  if (data_node == nullptr) {
    return 0;
  }
  const auto &output_mem_sizes = data_node->user_data<std::vector<size_t>>(kHyperOffloadOutputMemSizes);
  if (output_mem_sizes == nullptr || output_mem_sizes->empty()) {
    return 0;
  }
  return (*output_mem_sizes)[0];
}

size_t CalculateFreeIndex(size_t anchor_index, size_t distance, size_t profile_size, bool is_release) {
  if (is_release) {
    return std::min(anchor_index + distance, profile_size - 1);
  }
  return (anchor_index > distance) ? (anchor_index - distance) : 0;
}

size_t SimulateMemoryPeakWithDistanceChange(size_t data_size, size_t old_index, size_t new_index,
                                            const std::vector<size_t> &memory_profile, bool is_release) {
  if (memory_profile.empty() || data_size == 0) {
    return memory_profile.empty() ? 0 : *std::max_element(memory_profile.begin(), memory_profile.end());
  }
  if (old_index == new_index) {
    return *std::max_element(memory_profile.begin(), memory_profile.end());
  }

  size_t max_memory = 0;
  for (size_t i = 0; i < memory_profile.size(); ++i) {
    size_t mem = memory_profile[i];
    if (is_release) {
      if (i >= old_index && i < new_index) {
        mem += data_size;
      }
    } else {
      if (i >= new_index && i < old_index) {
        mem += data_size;
      }
    }
    max_memory = std::max(max_memory, mem);
  }
  return max_memory;
}

size_t SimulateMemoryPeakWithReleaseDistance(const OffloadInfo &info, size_t old_distance, size_t new_distance,
                                             size_t last_user_index, const std::vector<size_t> &memory_profile) {
  size_t data_size = GetDataNodeSizeBytes(info.data_node.first);
  size_t old_free_index = CalculateFreeIndex(last_user_index, old_distance, memory_profile.size(), true);
  size_t new_free_index = CalculateFreeIndex(last_user_index, new_distance, memory_profile.size(), true);
  return SimulateMemoryPeakWithDistanceChange(data_size, old_free_index, new_free_index, memory_profile, true);
}

void UpdateMemoryProfileWithDistanceChange(size_t data_size, size_t old_index, size_t new_index,
                                           std::vector<size_t> &memory_profile, bool is_release) {
  if (memory_profile.empty() || data_size == 0 || old_index == new_index) {
    return;
  }

  if (is_release) {
    for (size_t i = old_index; i < new_index && i < memory_profile.size(); ++i) {
      memory_profile[i] += data_size;
    }
  } else {
    for (size_t i = new_index; i < old_index && i < memory_profile.size(); ++i) {
      memory_profile[i] += data_size;
    }
  }
}

void UpdateMemoryProfileWithReleaseDistance(const OffloadInfo &info, size_t old_distance, size_t new_distance,
                                            size_t last_user_index, std::vector<size_t> &memory_profile) {
  size_t data_size = GetDataNodeSizeBytes(info.data_node.first);
  size_t old_free_index = CalculateFreeIndex(last_user_index, old_distance, memory_profile.size(), true);
  size_t new_free_index = CalculateFreeIndex(last_user_index, new_distance, memory_profile.size(), true);
  UpdateMemoryProfileWithDistanceChange(data_size, old_free_index, new_free_index, memory_profile, true);
}

struct SafeDistanceSearchState {
  size_t left;
  size_t right;
  size_t best_distance;
};

struct SafeDistanceSearchInput {
  size_t data_size;
  size_t anchor_index;
  size_t min_distance;
  size_t max_distance;
  const std::vector<size_t> &memory_profile;
  size_t memory_threshold;
  bool is_release;
  size_t profile_size;
};

bool UpdateSafeDistanceSearchState(size_t mid, size_t simulated_peak, size_t memory_threshold,
                                   SafeDistanceSearchState *state) {
  MS_EXCEPTION_IF_NULL(state);
  if (simulated_peak <= memory_threshold) {
    state->best_distance = mid;
    state->left = mid + 1;
    return true;
  }
  if (mid == 0) {
    return false;
  }
  state->right = mid - 1;
  return true;
}

size_t FindMaxSafeDistanceValue(const SafeDistanceSearchInput &input) {
  SafeDistanceSearchState state{input.min_distance, input.max_distance, input.min_distance};

  size_t min_index = CalculateFreeIndex(input.anchor_index, input.min_distance, input.profile_size, input.is_release);

  while (state.left <= state.right) {
    size_t mid = state.left + (state.right - state.left) / 2;
    size_t mid_index = CalculateFreeIndex(input.anchor_index, mid, input.profile_size, input.is_release);
    size_t simulated_peak = SimulateMemoryPeakWithDistanceChange(input.data_size, min_index, mid_index,
                                                                 input.memory_profile, input.is_release);
    if (!UpdateSafeDistanceSearchState(mid, simulated_peak, input.memory_threshold, &state)) {
      break;
    }
  }
  return state.best_distance;
}

struct SafeReleaseDistanceSearchInput {
  const OffloadInfo &info;
  size_t min_distance;
  size_t max_distance;
  size_t last_user_index;
  const std::vector<size_t> &memory_profile;
  size_t memory_threshold;
};

size_t FindMaxSafeReleaseDistanceValue(const SafeReleaseDistanceSearchInput &input) {
  size_t data_size = GetDataNodeSizeBytes(input.info.data_node.first);
  const SafeDistanceSearchInput search_input{data_size,
                                             input.last_user_index,
                                             input.min_distance,
                                             input.max_distance,
                                             input.memory_profile,
                                             input.memory_threshold,
                                             true,
                                             input.memory_profile.size()};
  return FindMaxSafeDistanceValue(search_input);
}

size_t GetIdealReleaseDistance(const AnfNodePtr &data_node, const CNodePtrList &exec_order,
                               const std::map<AnfNodePtr, size_t> &node_to_index, size_t last_user_index) {
  auto node_attr_dist = GetNodeReleaseDistance(data_node);
  if (node_attr_dist.has_value()) {
    return node_attr_dist.value();
  }

  const auto &config = HyperOffloadConfig::GetInstance();
  if (config.enable_adaptive_release_distance()) {
    auto it = node_to_index.find(data_node);
    size_t data_node_index = (it != node_to_index.end()) ? it->second : 0;
    return CalculateAdaptiveReleaseDistance(data_node, exec_order, data_node_index, last_user_index);
  }

  return config.release_distance();
}

size_t GetIdealPrefetchDistance(const AnfNodePtr &data_node, const CNodePtrList &exec_order,
                                const std::map<AnfNodePtr, size_t> &node_to_index, size_t first_use_index) {
  auto node_attr_dist = GetNodePrefetchDistance(data_node);
  if (node_attr_dist.has_value()) {
    return node_attr_dist.value();
  }

  const auto &config = HyperOffloadConfig::GetInstance();
  if (config.enable_adaptive_prefetch_distance()) {
    return CalculateAdaptivePrefetchDistance(data_node, exec_order, first_use_index);
  }

  return config.prefetch_distance();
}

void PhaseOneGetAdaptiveDistance(std::vector<OffloadInfo> &offload_infos, const CNodePtrList &exec_order,
                                 const std::map<AnfNodePtr, size_t> &node_to_index) {
  for (auto &info : offload_infos) {
    size_t last_user_index = GetLastUserIndexForOffloadInfo(info, node_to_index);
    info.optimized_release_distance =
      GetIdealReleaseDistance(info.data_node.first, exec_order, node_to_index, last_user_index);
  }
}

void PhaseTwoPositionBasedExpansion(std::vector<OffloadInfo> &offload_infos, const CNodePtrList &exec_order,
                                    const std::map<AnfNodePtr, size_t> &node_to_index) {
  if (offload_infos.empty()) {
    return;
  }

  std::vector<size_t> last_user_indices;
  for (const auto &info : offload_infos) {
    size_t last_user_index = GetLastUserIndexForOffloadInfo(info, node_to_index);
    last_user_indices.push_back(last_user_index);
  }

  size_t max_release_position = 0;
  for (size_t i = 0; i < offload_infos.size(); ++i) {
    size_t release_position = last_user_indices[i] + offload_infos[i].optimized_release_distance;
    max_release_position = std::max(max_release_position, release_position);
  }

  for (size_t i = 0; i < offload_infos.size(); ++i) {
    auto &info = offload_infos[i];
    size_t last_user_index = last_user_indices[i];
    size_t current_distance = info.optimized_release_distance;

    size_t position_gap = max_release_position - last_user_index;
    size_t expansion = position_gap / 2;

    if (expansion > 0) {
      size_t new_distance = current_distance + expansion;
      info.optimized_release_distance = new_distance;
    }
  }
}

void PhaseThreeDownwardAdjustmentForSafety(std::vector<OffloadInfo> &offload_infos, std::vector<size_t> &memory_profile,
                                           const CNodePtrList &exec_order,
                                           const std::map<AnfNodePtr, size_t> &node_to_index, size_t memory_threshold) {
  for (auto &info : offload_infos) {
    size_t last_user_index = GetLastUserIndexForOffloadInfo(info, node_to_index);
    size_t expanded_distance = info.optimized_release_distance;

    size_t simulated_peak =
      SimulateMemoryPeakWithReleaseDistance(info, 0, expanded_distance, last_user_index, memory_profile);
    if (simulated_peak <= memory_threshold) {
      UpdateMemoryProfileWithReleaseDistance(info, 0, expanded_distance, last_user_index, memory_profile);
      continue;
    }

    const SafeReleaseDistanceSearchInput search_input{
      info, 0, expanded_distance, last_user_index, memory_profile, memory_threshold};
    size_t safe_distance = FindMaxSafeReleaseDistanceValue(search_input);

    UpdateMemoryProfileWithReleaseDistance(info, 0, safe_distance, last_user_index, memory_profile);
    info.optimized_release_distance = safe_distance;
  }
}

}  // namespace

struct PrefetchInfo {
  KernelWithIndex data_node;
  size_t offload_info_index;
  size_t first_use_index;
  size_t free_data_position;
  size_t ideal_prefetch_distance;
  size_t expanded_prefetch_distance;
  size_t optimized_prefetch_distance;
};

size_t GetFirstUseIndexForOffloadInfo(const OffloadInfo &info, const std::map<AnfNodePtr, size_t> &node_to_index) {
  size_t first_use_index = std::numeric_limits<size_t>::max();

  for (const auto &replace_info : info.replace_info_list) {
    for (const auto &rest_node : replace_info.replace_rest_nodes) {
      auto rest_it = node_to_index.find(rest_node.first);
      if (rest_it != node_to_index.end()) {
        first_use_index = std::min(first_use_index, rest_it->second);
      }
    }
  }

  return (first_use_index == std::numeric_limits<size_t>::max()) ? 0 : first_use_index;
}

size_t GetFreeDataPositionForOffloadInfo(const OffloadInfo &info, const std::map<AnfNodePtr, size_t> &node_to_index) {
  size_t free_data_position = 0;
  for (const auto &replace_info : info.replace_info_list) {
    if (replace_info.free_data_after_node.first != nullptr) {
      auto it = node_to_index.find(replace_info.free_data_after_node.first);
      if (it != node_to_index.end()) {
        free_data_position = std::max(free_data_position, it->second);
      }
    }
  }
  return free_data_position;
}

size_t SimulateMemoryPeakWithPrefetchDistance(const PrefetchInfo &info, size_t old_distance, size_t new_distance,
                                              size_t first_use_index, const std::vector<size_t> &memory_profile) {
  if (memory_profile.empty() || first_use_index == 0) {
    return 0;
  }
  size_t data_size = GetDataNodeSizeBytes(info.data_node.first);
  size_t old_prefetch_index = CalculateFreeIndex(first_use_index, old_distance, memory_profile.size(), false);
  size_t new_prefetch_index = CalculateFreeIndex(first_use_index, new_distance, memory_profile.size(), false);
  return SimulateMemoryPeakWithDistanceChange(data_size, old_prefetch_index, new_prefetch_index, memory_profile, false);
}

void UpdateMemoryProfileWithPrefetchDistance(const PrefetchInfo &info, size_t old_distance, size_t new_distance,
                                             size_t first_use_index, std::vector<size_t> &memory_profile) {
  if (memory_profile.empty() || first_use_index == 0) {
    return;
  }
  size_t data_size = GetDataNodeSizeBytes(info.data_node.first);
  size_t old_prefetch_index = CalculateFreeIndex(first_use_index, old_distance, memory_profile.size(), false);
  size_t new_prefetch_index = CalculateFreeIndex(first_use_index, new_distance, memory_profile.size(), false);
  UpdateMemoryProfileWithDistanceChange(data_size, old_prefetch_index, new_prefetch_index, memory_profile, false);
}

struct SafePrefetchDistanceSearchInput {
  const PrefetchInfo &info;
  size_t min_distance;
  size_t max_distance;
  size_t first_use_index;
  const std::vector<size_t> &memory_profile;
  size_t memory_threshold;
};

size_t FindMaxSafePrefetchDistanceValue(const SafePrefetchDistanceSearchInput &input) {
  if (input.first_use_index == 0) {
    return input.min_distance;
  }
  size_t data_size = GetDataNodeSizeBytes(input.info.data_node.first);
  const SafeDistanceSearchInput search_input{data_size,
                                             input.first_use_index,
                                             input.min_distance,
                                             input.max_distance,
                                             input.memory_profile,
                                             input.memory_threshold,
                                             false,
                                             input.memory_profile.size()};
  return FindMaxSafeDistanceValue(search_input);
}

void PhaseOneGetIdealPrefetchDistance(std::vector<PrefetchInfo> &prefetch_infos,
                                      const std::vector<OffloadInfo> &offload_infos, const CNodePtrList &exec_order,
                                      const std::map<AnfNodePtr, size_t> &node_to_index) {
  prefetch_infos.clear();
  prefetch_infos.reserve(offload_infos.size());

  for (size_t i = 0; i < offload_infos.size(); ++i) {
    const auto &info = offload_infos[i];
    size_t first_use_index = GetFirstUseIndexForOffloadInfo(info, node_to_index);
    size_t free_data_position = GetFreeDataPositionForOffloadInfo(info, node_to_index);
    size_t ideal_distance = GetIdealPrefetchDistance(info.data_node.first, exec_order, node_to_index, first_use_index);

    PrefetchInfo prefetch_info;
    prefetch_info.data_node = info.data_node;
    prefetch_info.offload_info_index = i;
    prefetch_info.first_use_index = first_use_index;
    prefetch_info.free_data_position = free_data_position;
    prefetch_info.ideal_prefetch_distance = ideal_distance;
    prefetch_info.expanded_prefetch_distance = ideal_distance;
    prefetch_info.optimized_prefetch_distance = ideal_distance;
    prefetch_infos.push_back(prefetch_info);
  }
}

void PhaseTwoOrderedExpansion(std::vector<PrefetchInfo> &prefetch_infos) {
  if (prefetch_infos.empty()) {
    return;
  }

  std::sort(prefetch_infos.begin(), prefetch_infos.end(),
            [](const PrefetchInfo &a, const PrefetchInfo &b) { return a.first_use_index < b.first_use_index; });

  size_t min_ideal_position = std::numeric_limits<size_t>::max();
  for (const auto &info : prefetch_infos) {
    size_t ideal_position =
      (info.first_use_index > info.ideal_prefetch_distance) ? (info.first_use_index - info.ideal_prefetch_distance) : 0;
    min_ideal_position = std::min(min_ideal_position, ideal_position);
  }

  for (auto &info : prefetch_infos) {
    size_t ideal_position =
      (info.first_use_index > info.ideal_prefetch_distance) ? (info.first_use_index - info.ideal_prefetch_distance) : 0;
    size_t position_gap = (ideal_position > min_ideal_position) ? (ideal_position - min_ideal_position) : 0;
    size_t expansion = position_gap / 2;
    info.expanded_prefetch_distance = info.ideal_prefetch_distance + expansion;
  }

  for (auto &info : prefetch_infos) {
    size_t prefetch_position = (info.first_use_index > info.expanded_prefetch_distance)
                                 ? (info.first_use_index - info.expanded_prefetch_distance)
                                 : 0;
    if (prefetch_position <= info.free_data_position) {
      size_t adjustment = info.free_data_position + 1 - prefetch_position;
      if (info.expanded_prefetch_distance > adjustment) {
        info.expanded_prefetch_distance -= adjustment;
      } else {
        info.expanded_prefetch_distance = 0;
      }
    }
  }
}

void PhaseThreeDownwardAdjustmentForMemoryPrefetch(std::vector<PrefetchInfo> &prefetch_infos,
                                                   std::vector<size_t> &memory_profile, size_t memory_threshold) {
  for (auto &info : prefetch_infos) {
    size_t expanded_distance = info.expanded_prefetch_distance;
    size_t simulated_peak =
      SimulateMemoryPeakWithPrefetchDistance(info, 0, expanded_distance, info.first_use_index, memory_profile);
    if (simulated_peak <= memory_threshold) {
      UpdateMemoryProfileWithPrefetchDistance(info, 0, expanded_distance, info.first_use_index, memory_profile);
      info.optimized_prefetch_distance = expanded_distance;
      continue;
    }

    const SafePrefetchDistanceSearchInput search_input{
      info, 0, expanded_distance, info.first_use_index, memory_profile, memory_threshold};
    size_t safe_distance = FindMaxSafePrefetchDistanceValue(search_input);
    UpdateMemoryProfileWithPrefetchDistance(info, 0, safe_distance, info.first_use_index, memory_profile);
    info.optimized_prefetch_distance = safe_distance;
  }
}

void OptimizeReleaseDistanceByMemory(std::vector<OffloadInfo> &offload_infos, std::vector<size_t> &memory_profile,
                                     const CNodePtrList &exec_order,
                                     const std::map<AnfNodePtr, size_t> &node_to_index) {
  if (offload_infos.empty() || memory_profile.empty() || exec_order.empty()) {
    return;
  }

  const auto &config = HyperOffloadConfig::GetInstance();
  size_t memory_threshold = EffectiveMemoryThresholdBytes(config.max_device_memory());

  if (config.enable_adaptive_release_distance()) {
    PhaseOneGetAdaptiveDistance(offload_infos, exec_order, node_to_index);
    PhaseTwoPositionBasedExpansion(offload_infos, exec_order, node_to_index);
    PhaseThreeDownwardAdjustmentForSafety(offload_infos, memory_profile, exec_order, node_to_index, memory_threshold);

    for (auto &info : offload_infos) {
      if (info.optimized_release_distance > 0 && info.data_node.first != nullptr) {
        auto cnode = info.data_node.first->cast<CNodePtr>();
        if (cnode != nullptr) {
          cnode->AddAttr("release_distance", MakeValue<size_t>(info.optimized_release_distance));
        }
      }
    }
  }
}

void ApplyOptimizedPrefetchDistance(const PrefetchInfo &prefetch_info, const OffloadInfo &offload_info) {
  if (prefetch_info.optimized_prefetch_distance == 0 || offload_info.data_node.first == nullptr) {
    return;
  }
  auto cnode = offload_info.data_node.first->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }
  cnode->AddAttr("prefetch_distance", MakeValue<size_t>(prefetch_info.optimized_prefetch_distance));
}

void ApplyOptimizedPrefetchDistances(const std::vector<PrefetchInfo> &prefetch_infos,
                                     const std::vector<OffloadInfo> &offload_infos) {
  for (const auto &prefetch_info : prefetch_infos) {
    if (prefetch_info.offload_info_index >= offload_infos.size()) {
      continue;
    }
    ApplyOptimizedPrefetchDistance(prefetch_info, offload_infos[prefetch_info.offload_info_index]);
  }
}

void OptimizePrefetchDistanceByMemory(const std::vector<OffloadInfo> &offload_infos,
                                      std::vector<size_t> &memory_profile, const CNodePtrList &exec_order,
                                      const std::map<AnfNodePtr, size_t> &node_to_index) {
  if (offload_infos.empty() || memory_profile.empty() || exec_order.empty()) {
    return;
  }

  const auto &config = HyperOffloadConfig::GetInstance();
  size_t memory_threshold = EffectiveMemoryThresholdBytes(config.max_device_memory());

  if (!config.enable_adaptive_prefetch_distance()) {
    return;
  }

  std::vector<PrefetchInfo> prefetch_infos;
  PhaseOneGetIdealPrefetchDistance(prefetch_infos, offload_infos, exec_order, node_to_index);
  PhaseTwoOrderedExpansion(prefetch_infos);
  PhaseThreeDownwardAdjustmentForMemoryPrefetch(prefetch_infos, memory_profile, memory_threshold);
  ApplyOptimizedPrefetchDistances(prefetch_infos, offload_infos);
}

std::vector<KernelWithIndex> FindUsersAfterGap(const UserInfo &user_info,
                                               const std::map<AnfNodePtr, size_t> &node_to_index, size_t gap_end) {
  std::vector<KernelWithIndex> users_after_gap;
  for (const auto &user : user_info.users) {
    const auto user_it = node_to_index.find(user.first);
    if (user_it != node_to_index.end() && user_it->second >= gap_end) {
      users_after_gap.push_back(user);
    }
  }
  return users_after_gap;
}

std::optional<ReplaceInfo> GenerateReplaceInfoFromSelectedGap(const UserInfo &user_info, const GapRange &gap_range,
                                                              const CNodePtrList &execution_order,
                                                              const std::map<AnfNodePtr, size_t> &node_to_index) {
  const size_t gap_start = gap_range.first;
  const size_t gap_end = gap_range.second;
  if (gap_start == 0 || gap_start >= gap_end || gap_start > execution_order.size() ||
      gap_end >= execution_order.size()) {
    return std::nullopt;
  }
  const auto data_node_it = node_to_index.find(user_info.node.first);
  if (data_node_it == node_to_index.end() || data_node_it->second >= execution_order.size()) {
    return std::nullopt;
  }
  const auto &d2h_insert_after_node = execution_order[data_node_it->second];
  if (d2h_insert_after_node == nullptr) {
    return std::nullopt;
  }
  const auto &free_data_after_node = execution_order[gap_start - 1];
  if (free_data_after_node == nullptr) {
    return std::nullopt;
  }
  auto replace_rest_nodes = FindUsersAfterGap(user_info, node_to_index, gap_end);
  if (replace_rest_nodes.empty()) {
    return std::nullopt;
  }
  return ReplaceInfo{KernelWithIndex{d2h_insert_after_node, 0}, replace_rest_nodes,
                     KernelWithIndex{free_data_after_node, 0}};
}

OffloadInfoList GenerateOffloadInfoFromSelectedGaps(const UserInfoList &selected_user_infos,
                                                    const std::vector<GapRangeList> &selected_gap_ranges,
                                                    const CNodePtrList &execution_order,
                                                    const std::map<AnfNodePtr, size_t> &node_to_index,
                                                    const std::vector<size_t> &selected_peak_indices) {
  OffloadInfoList offload_info_list;
  if (selected_user_infos.size() != selected_gap_ranges.size()) {
    MS_LOG(WARNING) << "[HyperOffload] selected user count " << selected_user_infos.size()
                    << " does not match selected gap range count " << selected_gap_ranges.size();
    return offload_info_list;
  }
  if (selected_user_infos.size() != selected_peak_indices.size()) {
    MS_LOG(WARNING) << "[HyperOffload] selected user count " << selected_user_infos.size()
                    << " does not match selected peak index count " << selected_peak_indices.size();
    return offload_info_list;
  }
  for (size_t i = 0; i < selected_user_infos.size(); ++i) {
    const auto peak_covering_gap = FindPeakCoveringGap(selected_gap_ranges[i], selected_peak_indices[i]);
    if (!peak_covering_gap.has_value()) {
      continue;
    }
    const auto replace_info = GenerateReplaceInfoFromSelectedGap(selected_user_infos[i], peak_covering_gap.value(),
                                                                 execution_order, node_to_index);
    if (replace_info.has_value()) {
      offload_info_list.emplace_back(OffloadInfo{selected_user_infos[i].node, ReplaceInfoList{replace_info.value()}});
    }
  }
  return offload_info_list;
}

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
  const MemoryProfileBuildInputs memory_profile_inputs{
    user_info_list, node_to_index, execution_order.size(), parameter_memory_baseline, execution_order, graph};
  const MemoryProfileBuildOutputs memory_profile_outputs{&memory_profile, &all_candidates};
  BuildMemoryProfileAndCandidates(memory_profile_inputs, memory_profile_outputs);

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
  std::vector<GapRangeList> selected_gap_ranges;
  std::vector<size_t> selected_peak_indices;

  const PeakSelectionLoopInputs selection_inputs{all_candidates, execution_order, node_to_index, profile_size, max_mem};
  const PeakSelectionLoopOutputs selection_outputs{&working_profile,           &selected_user_infos,
                                                   &selected_producer_anchors, &successfully_offloaded,
                                                   &selected_gap_ranges,       &selected_peak_indices};
  RunPeakOffloadSelectionLoop(selection_inputs, selection_outputs);

  const auto final_peak_opt = FindGlobalPeak(working_profile);
  LogPeakReductionFailureIfNeeded(initial_peak_opt, final_peak_opt, max_mem, execution_order, selected_user_infos);

  offload_info_list = GenerateOffloadInfoFromSelectedGaps(selected_user_infos, selected_gap_ranges, execution_order,
                                                          node_to_index, selected_peak_indices);

  OptimizeReleaseDistanceByMemory(offload_info_list, working_profile, execution_order, node_to_index);
  OptimizePrefetchDistanceByMemory(offload_info_list, working_profile, execution_order, node_to_index);

  return offload_info_list;
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
