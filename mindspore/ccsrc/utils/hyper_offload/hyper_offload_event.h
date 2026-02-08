/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_EVENT_H_
#define MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_EVENT_H_

#include "ir/anf.h"
#include "utils/hyper_offload/hyper_offload_operation.h"
#include "utils/hyper_offload/hyper_offload_utils.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
class HyperOffloadEventInjector {
 public:
  HyperOffloadEventInjector(const FuncGraphPtr &graph, const CNodePtrList &exec_order,
                            HyperOffloadOperations *operations);
  ~HyperOffloadEventInjector() = default;

  void Run();

 private:
  void AddEventNodesForSingleD2HNode(const CNodePtr &d2h_node, EventNodeFactory &event_factory, size_t node_index);
  void AddEventNodesForD2HNodes(EventNodeFactory &event_factory);
  void AddEventNodesForNonLastH2DNode(const CNodePtr &h2d_node, const AnfNodePtr &original_data, size_t first_user_idx,
                                      size_t last_user_idx, const CNodePtr &recv_node);
  void AddEventNodesForLastH2DNode(const CNodePtr &h2d_node, const AnfNodePtr &original_data, size_t first_user_idx,
                                   const CNodePtr &recv_node);
  void AddEventNodesForH2DNodes(EventNodeFactory &event_factory);

  using H2DGroupMap = std::unordered_map<AnfNodePtr, std::vector<std::pair<size_t, CNodePtr>>>;
  H2DGroupMap GroupH2DNodes() const;
  void ProcessH2DGroup(const AnfNodePtr &input_node, const std::vector<std::pair<size_t, CNodePtr>> &h2d_list,
                       EventNodeFactory &event_factory);
  std::pair<size_t, size_t> FindH2DUserIndices(const CNodePtr &h2d_node) const;

  FuncGraphPtr graph_;
  const CNodePtrList &exec_order_;
  HyperOffloadOperations *operations_;
};
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_EVENT_H_
