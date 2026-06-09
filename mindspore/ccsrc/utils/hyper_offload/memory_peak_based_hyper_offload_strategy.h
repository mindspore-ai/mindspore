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

#ifndef MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_MEMORY_PEAK_BASED_HYPER_OFFLOAD_STRATEGY_H_
#define MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_MEMORY_PEAK_BASED_HYPER_OFFLOAD_STRATEGY_H_

#include <memory>
#include <vector>
#include "utils/hyper_offload/hyper_offload_strategy.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {

/**
 * @brief Memory-peak-based strategy for selecting offload candidates when memory exceeds threshold.
 */
class MemoryPeakBasedHyperOffloadStrategy : public HyperOffloadStrategy {
 public:
  MemoryPeakBasedHyperOffloadStrategy() = default;
  // Execute the memory-peak-based strategy.
  OffloadInfoList Run(const CNodePtrList &execution_order, const UserInfoList &user_info_list,
                      const FuncGraphPtr &graph = nullptr) override;
};
using MemoryPeakBasedHyperOffloadStrategyPtr = std::unique_ptr<MemoryPeakBasedHyperOffloadStrategy>;

// Optimize release_distance based on memory constraints (two-phase optimization)
void OptimizeReleaseDistanceByMemory(std::vector<OffloadInfo> &offload_infos, std::vector<size_t> &memory_profile,
                                     const CNodePtrList &exec_order, const std::map<AnfNodePtr, size_t> &node_to_index);

// Optimize prefetch_distance based on memory constraints (three-phase optimization)
void OptimizePrefetchDistanceByMemory(const std::vector<OffloadInfo> &offload_infos,
                                      std::vector<size_t> &memory_profile, const CNodePtrList &exec_order,
                                      const std::map<AnfNodePtr, size_t> &node_to_index);

}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_MEMORY_PEAK_BASED_HYPER_OFFLOAD_STRATEGY_H_
