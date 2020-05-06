/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_GRAPH_H_
#define MINDSPORE_CCSRC_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_GRAPH_H_

#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "ir/anf.h"
#include "parallel/allreduce_fusion/allreduce_node.h"
#include "parallel/status.h"

namespace mindspore {
namespace parallel {
class AllreduceGraph {
 public:
  AllreduceGraph()
      : head_cnode_(nullptr),
        arnode_set_(),
        arnode_vec_(),
        cnode_set_(),
        para_cnode_map_(),
        para_cnodeset_map_(),
        cnode_paraset_map_(),
        cnode_arnode_map_(),
        max_(0) {}
  virtual ~AllreduceGraph() = default;
  Status AddNode(const CNodePtr &node, const AnfNodePtr &para);
  Status AddEdge(const CNodePtr &from, const CNodePtr &to, double dist);
  bool NodeInGraph(const CNodePtr &node) const;
  std::vector<AnfNodePtr> GetParaByCost(double from, double to);
  // Find the first several AllreduceNode whose depend_feat_size is less than to, the sum of whose parameter size is
  // over para_size.
  // Return the parameter AnfNodePtr vector corresponding to these AllreduceNodes and the smallest depend_feat_size.
  // If the sum of left AllreduceNode's parameter size is less than para_size, the returned depend_feat_size must be 0.
  std::pair<std::vector<AnfNodePtr>, double> GetParaByParaSize(double to, double para_size);
  // If one parameter is used by multiple AllreduceNode, parameter belong to the last node for backward computation
  // is saved by the corresponding AllreduceNode, parameters belong to other AllreduceNode are removed.
  // Called during precise optimization, not implemented temporarily.
  void SortArnode();
  Status RemoveExtraParas();
  void PrintCNodeSet() const;
  void PrintAllredueGraphInfo() const;
  void PrintArnodeVec() const;
  void PrintArnodeSet() const;
  const std::unordered_set<CNodePtr> &cnode_set() const { return cnode_set_; }
  CNodePtr head_cnode() const { return head_cnode_; }
  Status set_head_cnode(const CNodePtr &node);
  double max() const { return max_; }

 private:
  CNodePtr head_cnode_;
  std::set<AllreduceNodePtr> arnode_set_;
  std::vector<AllreduceNode> arnode_vec_;
  std::unordered_set<CNodePtr> cnode_set_;
  // If One ParameterPtr is used by multiple CNode, the last node for backward computation is saved.
  std::unordered_map<AnfNodePtr, std::vector<CNodePtr>> para_cnode_map_;
  // One ParameterPtr may be used by multiple CNode
  std::unordered_map<AnfNodePtr, std::unordered_set<CNodePtr>> para_cnodeset_map_;
  // Multiple Parameter may be inputs to the same CNode
  std::unordered_map<CNodePtr, std::unordered_set<AnfNodePtr>> cnode_paraset_map_;
  std::unordered_map<CNodePtr, AllreduceNodePtr> cnode_arnode_map_;
  double max_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_GRAPH_H_
