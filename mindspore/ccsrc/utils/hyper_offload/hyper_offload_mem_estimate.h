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
  size_t last_usage_index;
};

/**
 * @brief Build memory profile (bytes per execution index) and offload candidates.
 * @param user_info_list User info from graph analysis
 * @param node_to_index Node to execution order index map
 * @param order_size Execution order size
 * @param parameter_memory_baseline Total parameter memory (bytes)
 * @param memory_profile Output: memory bytes at each index
 * @param all_candidates Output: offload candidates
 * @param index_order Execution order for indices (use mem_reuse when subgraph_multi_call)
 * @param graph FuncGraph for output/summary resolution
 * @param kernel_order Optional: if non-null, use for kernel iteration; else use kg->execution_order() when
 *                     KernelGraph, or index_order otherwise
 */
void BuildMemoryProfileAndCandidates(const UserInfoList &user_info_list,
                                     const std::map<AnfNodePtr, size_t> &node_to_index, size_t order_size,
                                     size_t parameter_memory_baseline, std::vector<size_t> *memory_profile,
                                     std::vector<PeakEstimateCandidate> *all_candidates,
                                     const CNodePtrList &index_order, const FuncGraphPtr &graph,
                                     const CNodePtrList *kernel_order = nullptr);

/**
 * @brief Reduce memory profile in [start_index, end_index) by size (simulate offload).
 */
void ReduceMemoryProfile(size_t start_index, size_t end_index, size_t size, std::vector<size_t> *memory_profile);

/**
 * @brief Compute total parameter memory (bytes) for graph.
 */
size_t ComputeTotalParameterMemory(const FuncGraphPtr &graph);

}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_MEM_ESTIMATE_H_
