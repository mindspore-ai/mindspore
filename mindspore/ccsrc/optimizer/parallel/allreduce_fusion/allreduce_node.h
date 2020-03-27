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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_NODE_H_
#define MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_NODE_H_

#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <vector>
#include "ir/anf.h"
#include "optimizer/parallel/status.h"

namespace mindspore {
namespace parallel {
class AllreduceNode;
using AllreduceNodePtr = std::shared_ptr<AllreduceNode>;

class AllreduceNode {
 public:
  AllreduceNode()
      : cnode_ptr_(nullptr), prev_(), next_(), paras_(), para_size_map_(), curr_para_size_(0), depend_feat_size_(0) {}
  Status Init(const CNodePtr& cnode_ptr);
  Status AddPara(const AnfNodePtr& node_ptr);
  Status RemovePara(const AnfNodePtr& node_ptr);
  double curr_para_size() const { return curr_para_size_; }
  virtual ~AllreduceNode() = default;
  Status AddPrev(const AllreduceNodePtr& prev_node, double dist);
  Status AddNext(const AllreduceNodePtr& next_node);
  double depend_feat_size() const { return depend_feat_size_; }
  void ToString() const;
  bool operator<(const AllreduceNode& node) const { return depend_feat_size_ < node.depend_feat_size(); }

 private:
  CNodePtr cnode_ptr_;
  std::vector<AllreduceNodePtr> prev_;
  std::vector<AllreduceNodePtr> next_;
  std::unordered_set<AnfNodePtr> paras_;
  std::unordered_map<AnfNodePtr, double> para_size_map_;
  double curr_para_size_;
  double depend_feat_size_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_ALLREDUCE_FUSION_ALLREDUCE_NODE_H_
