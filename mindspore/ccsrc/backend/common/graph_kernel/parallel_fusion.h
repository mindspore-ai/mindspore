
/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_FUSION_H_

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "base/base.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/graph_kernel/parallel_cost_model.h"
#include "include/backend/kernel_graph.h"
#include "utils/ms_context.h"

namespace mindspore::graphkernel {
class ParallelInfo {
 public:
  ParallelInfo() = default;
  ParallelInfo(const AnfNodePtrList &nodes, const std::vector<DimInfoPtr> &dims, const FusionInfoPtr &fusion_info)
      : nodes_(nodes), dims_(dims), fusion_info_(fusion_info) {}
  ~ParallelInfo() = default;

  size_t GetSize() const {
    if (nodes_.size() != dims_.size()) {
      MS_LOG(EXCEPTION) << "Internal error in parallel info! nodes' size is different from dims' size: "
                        << nodes_.size() << " vs " << dims_.size();
    }
    return nodes_.size();
  }
  const AnfNodePtrList &nodes() const { return nodes_; }
  const std::vector<DimInfoPtr> &dims() const { return dims_; }
  const FusionInfoPtr &fusion_info() const { return fusion_info_; }

 private:
  AnfNodePtrList nodes_;
  std::vector<DimInfoPtr> dims_;
  FusionInfoPtr fusion_info_;
};

class ParallelConfig {
 public:
  ParallelConfig() = default;
  explicit ParallelConfig(size_t max_n) : max_num_for_fuse_(max_n) {}
  ~ParallelConfig() = default;
  size_t max_num_for_fuse() const { return max_num_for_fuse_; }

 private:
  size_t max_num_for_fuse_{10};  // Too many nodes to fuse together may produce bad result.
};

struct NodeRelation {
 public:
  NodeRelation() {}
  ~NodeRelation() = default;
  OrderedSet<AnfNodePtr> pres;
  OrderedSet<AnfNodePtr> nexts;
};

class ParallelOpFusion : public opt::Pass {
 public:
  ParallelOpFusion(const std::string &target, const ParallelConfig &config)
      : Pass("parallel_fusion"), target_(target), config_(config) {}
  ~ParallelOpFusion() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  std::tuple<AnfNodePtrList, std::vector<int>> GetAvaliableNodesByOffset(int start, const std::vector<size_t> &offsets,
                                                                         const std::vector<bool> &used,
                                                                         const AnfNodePtrList &nodes,
                                                                         const std::set<int> &excludes) const;

  std::tuple<std::vector<bool>, std::vector<ParallelInfo>> DoSearchInSortedCandidates(
    size_t origin_size, const AnfNodePtrList &candidates, std::map<AnfNodePtr, int> *origin_indices,
    std::map<AnfNodePtr, int> *sorted_indices);

  std::tuple<std::vector<bool>, std::vector<ParallelInfo>> SearchFuseNodesInCandidates(const AnfNodePtrList &cs);

  void SearchFuseNodesInParallelGroup(const std::vector<AnfNodePtrList> &group,
                                      std::vector<ParallelInfo> *parallel_infos);

  std::vector<ParallelInfo> SearchFusableParallelCNodes(const std::vector<std::vector<AnfNodePtrList>> &groups);

  void SetFusionInfoAttrToNode(const AnfNodePtr &node, const ParallelInfo &parallel_info);

  void SetFusedParallelOpAttrToReturnNode(const ParallelInfo &parallel_info);

  bool CreateParallelOpSubGraphs(const std::vector<ParallelInfo> &parallel_infos,
                                 const std::shared_ptr<session::KernelGraph> &kernel_graph);

  OrderedMap<AnfNodePtr, NodeRelation> GenAnalysisGraph(const AnfNodePtrList &nodes);
  std::vector<std::vector<AnfNodePtrList>> SearchParallelGroups(const OrderedMap<AnfNodePtr, NodeRelation> &node_rels);

  std::string target_;
  ParallelConfig config_;
  ParallelCostModelPtr cost_model_ptr_;
  std::set<AnfNodePtr> virtual_noout_nodes_;
  std::set<AnfNodePtr> ignore_noin_nodes_;
  unsigned int parallel_level_{0};
};
using ParallelOpFusionPtr = std::shared_ptr<ParallelOpFusion>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_FUSION_H_
