/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OFFLOADING_ACTIVATION_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OFFLOADING_ACTIVATION_H_

#include <memory>
#include <utility>
#include <vector>

#include "include/frontend/optimizer/optimizer.h"

namespace mindspore {
namespace parallel {
bool OffloadActivation(const FuncGraphPtr &func_graph);

class OffloadActivationOptimizer {
  struct OffloadInfo {
    OffloadInfo(const FuncGraphPtr &fw_graph, const CNodePtr &fw_node, const FuncGraphPtr &bw_graph,
                const CNodePtr &bw_node, size_t bw_input_idx, int64_t bw_prefetch)
        : fw_graph_(fw_graph),
          fw_node_(fw_node),
          bw_graph_(bw_graph),
          bw_node_(bw_node),
          bw_input_idx_(bw_input_idx),
          bw_prefetch_(bw_prefetch) {}
    FuncGraphPtr fw_graph_;
    CNodePtr fw_node_;
    FuncGraphPtr bw_graph_;
    CNodePtr bw_node_;
    size_t bw_input_idx_;
    int64_t bw_prefetch_;
  };
  using OffloadInfoPtr = std::shared_ptr<OffloadInfo>;

 public:
  bool Optimize(const FuncGraphPtr &func_graph);

 private:
  static FuncGraphPtr GetBackwardGraph(const FuncGraphPtr &func_graph);
  void GetFwBwGraphs();
  void AddOffloadForCommUser(const FuncGraphPtr &fw_graph);
  void GetActivationOffloadInfo(const FuncGraphPtr &fw_graph, const FuncGraphPtr &bw_grpah);
  void AddDependForMoveOut(const FuncGraphPtr &fw_graph, const CNodePtr &fw_node, const CNodePtr &move_out);
  void InsertMoveToForOffloadActivation(const OffloadInfoPtr &offload_info);
  void WarningOffloadOutsideLazyInline();

  FuncGraphManagerPtr manager_{nullptr};
  std::vector<std::pair<FuncGraphPtr, FuncGraphPtr>> fw_bw_graphs_;
  std::vector<OffloadInfoPtr> offload_infos_;
  HashMap<CNodePtr, HashMap<FuncGraphPtr, CNodePtr>> move_to_node_cache_;
  HashSet<CNodePtr> offload_in_lazy_inline_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OFFLOADING_ACTIVATION_H_
