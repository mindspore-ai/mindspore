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

#ifndef MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_MEM_ESTIMATE_H_
#define MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_MEM_ESTIMATE_H_

#include <map>
#include <vector>

#include "ir/anf.h"
#include "utils/hyper_offload/hyper_offload_utils.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {

/**
 * @brief Candidate for offload selection from peak estimation.
 */
struct PeakEstimateCandidate {
  UserInfo user_info;
  size_t size;
  size_t creation_index;
  size_t last_usage_index;  // Aligned with profile modeling (OutputLastUsageMap after missing-table extension).
};

/**
 * @brief Build the modeled memory profile and peak-offload candidates.
 * @note The function prefers `memory_block.csv` when it is available. If the table cannot be loaded,
 *       it falls back to static tensor-size estimation and uses `parameter_memory_baseline` as the
 *       profile baseline.
 */
void BuildMemoryProfileAndCandidates(const UserInfoList &user_info_list,
                                     const std::map<AnfNodePtr, size_t> &node_to_index, size_t order_size,
                                     size_t parameter_memory_baseline, std::vector<size_t> *memory_profile,
                                     std::vector<PeakEstimateCandidate> *all_candidates,
                                     const CNodePtrList &index_order, const FuncGraphPtr &graph);

/**
 * @brief Extend UserInfo.users so ReplaceInfo/offload spans reach schedule last-use.
 */
void AugmentUserInfoListForOffloadSpan(UserInfoList *user_info_list, const CNodePtrList &index_order,
                                       const std::map<AnfNodePtr, size_t> &node_to_index, size_t order_size);

/**
 * @brief Reduce memory profile in [start_index, end_index) by size.
 */
void ReduceMemoryProfile(size_t start_index, size_t end_index, size_t size, std::vector<size_t> *memory_profile);

/**
 * @brief Compute total parameter memory used as the fallback profile baseline.
 */
size_t ComputeTotalParameterMemory(const FuncGraphPtr &graph);

/**
 * @brief Query the one-step workspace pulse used by peak modeling for a node.
 */
size_t GetNodeWorkspacePulseForProfile(const AnfNodePtr &node);

}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_MEM_ESTIMATE_H_
