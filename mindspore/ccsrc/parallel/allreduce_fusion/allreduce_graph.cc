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

#include "parallel/allreduce_fusion/allreduce_graph.h"
#include <algorithm>
#include <functional>
#include "ir/anf.h"
#include "parallel/allreduce_fusion/allreduce_node.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status AllreduceGraph::AddNode(const CNodePtr& node, const AnfNodePtr& para) {
  auto arnode = std::make_shared<AllreduceNode>(AllreduceNode());
  if (arnode->Init(node) != SUCCESS) {
    MS_LOG(ERROR) << "AllreduceNode Init failed";
    return FAILED;
  }
  if (arnode->AddPara(para) != SUCCESS) {
    MS_LOG(ERROR) << "AllreduceNode AddPara failed";
    return FAILED;
  }
  cnode_arnode_map_[node] = arnode;

  auto arnode_emplace_return = arnode_set_.insert(arnode);
  if (!arnode_emplace_return.second) {
    MS_LOG(INFO) << "node: " << node->DebugString() << "'s arnode has already been added!";
  }
  auto cnode_emplace_return = cnode_set_.emplace(node);
  if (!cnode_emplace_return.second) {
    MS_LOG(INFO) << "node: " << node->DebugString() << " has already been added!";
  }
  cnode_emplace_return = para_cnodeset_map_[para].emplace(node);
  if (!cnode_emplace_return.second) {
    MS_LOG(INFO) << "node: " << node->DebugString() << " already in para: " << para->fullname_with_scope()
                 << "'s cnodeset!";
  }
  auto para_emplace_return = cnode_paraset_map_[node].emplace(para);
  if (!para_emplace_return.second) {
    MS_LOG(INFO) << "para: " << para->fullname_with_scope() << " already in node: " << node->DebugString()
                 << "'s paraset!";
  }
  return SUCCESS;
}

Status AllreduceGraph::AddEdge(const CNodePtr& from, const CNodePtr& to, double dist) {
  auto from_arnode_iter = cnode_arnode_map_.find(from);
  if (from_arnode_iter == cnode_arnode_map_.end()) {
    MS_LOG(ERROR) << "cnode from: " << from->DebugString() << "has not been added";
    PrintCNodeSet();
    return FAILED;
  }
  auto to_arnode_iter = cnode_arnode_map_.find(to);
  if (to_arnode_iter == cnode_arnode_map_.end()) {
    MS_LOG(ERROR) << "cnode to: " << to->DebugString() << "has not been added";
    PrintCNodeSet();
    return FAILED;
  }
  auto from_arnode = from_arnode_iter->second;
  auto to_arnode = to_arnode_iter->second;
  if (from_arnode->AddNext(to_arnode) != SUCCESS) {
    MS_LOG(ERROR) << "from_arnode AddNext failed";
    return FAILED;
  }
  if (to_arnode->AddPrev(from_arnode, dist) != SUCCESS) {
    MS_LOG(ERROR) << "to_arnode AddPrev failed";
    return FAILED;
  }
  max_ = std::max(max_, to_arnode->depend_feat_size());
  MS_LOG(DEBUG) << "from " << from->DebugString() << ", to " << to->DebugString();
  MS_LOG(DEBUG) << "from depend_feat_size: " << from_arnode->depend_feat_size()
                << ", to depend_feat_size: " << to_arnode->depend_feat_size();
  return SUCCESS;
}

bool AllreduceGraph::NodeInGraph(const CNodePtr& node) const {
  auto cnode_iter = cnode_set_.find(node);
  return !(cnode_iter == cnode_set_.end());
}

std::vector<AnfNodePtr> AllreduceGraph::GetParaByCost(double from, double to) {
  std::vector<AnfNodePtr> nodes;
  for (auto& cnode_arnode : cnode_arnode_map_) {
    MS_LOG(DEBUG) << "cnode: " << cnode_arnode.first->DebugString()
                  << ", depend_feat_size: " << cnode_arnode.second->depend_feat_size()
                  << " curr_para_size: " << cnode_arnode.second->curr_para_size();
    if ((cnode_arnode.second->depend_feat_size() <= to) && (cnode_arnode.second->depend_feat_size() > from)) {
      (void)nodes.insert(nodes.end(), cnode_paraset_map_[cnode_arnode.first].begin(),
                         cnode_paraset_map_[cnode_arnode.first].end());
    }
  }
  return nodes;
}

std::pair<std::vector<AnfNodePtr>, double> AllreduceGraph::GetParaByParaSize(double to, double para_size) {
  std::vector<AnfNodePtr> nodes;
  double cur_para_size = 0;
  double from = to;
  for (auto& arnode : arnode_vec_) {
    if (arnode.depend_feat_size() >= to) {
      continue;
    }
    if (para_size > 0 && cur_para_size >= para_size && arnode.depend_feat_size() < from) {
      return std::make_pair(nodes, from);
    }
    (void)nodes.insert(nodes.end(), arnode.paras().begin(), arnode.paras().end());
    cur_para_size += arnode.curr_para_size();
    from = arnode.depend_feat_size();
  }
  MS_LOG(INFO) << "GetParaByParaSize has reached head node! para_size: " << para_size
               << " cur_para_size: " << cur_para_size << " from: " << from;
  return std::make_pair(nodes, from);
}

void AllreduceGraph::PrintCNodeSet() const {
  MS_LOG(INFO) << "CNodeSet:";
  for (auto& cnode : cnode_set_) {
    MS_LOG(INFO) << cnode->DebugString();
  }
}

void AllreduceGraph::PrintAllredueGraphInfo() const {
  MS_LOG(INFO) << "max: " << max_;
  for (auto& cnode_arnode : cnode_arnode_map_) {
    MS_LOG(INFO) << "cnode: " << cnode_arnode.first->DebugString();
    MS_LOG(INFO) << "arnode info: ";
    cnode_arnode.second->ToString();
  }
}

void AllreduceGraph::PrintArnodeVec() const {
  MS_LOG(INFO) << "ArnodeVec:";
  for (auto& arnode : arnode_vec_) {
    arnode.ToString();
  }
}

void AllreduceGraph::PrintArnodeSet() const {
  MS_LOG(INFO) << "ArnodeSet:";
  for (auto& arnode : arnode_set_) {
    arnode->ToString();
  }
}

void AllreduceGraph::SortArnode() {
  arnode_vec_.clear();
  for (auto& node : arnode_set_) {
    arnode_vec_.emplace_back(*node);
  }
  std::sort(arnode_vec_.begin(), arnode_vec_.end(), std::greater<>());
}

Status AllreduceGraph::RemoveExtraParas() {
  std::unordered_set<AnfNodePtr> para_map;
  for (auto& node : arnode_vec_) {
    for (auto& para : node.paras()) {
      auto emplac_result = para_map.emplace(para);
      if (!emplac_result.second) {
        MS_LOG(DEBUG) << "parameter: " << para->fullname_with_scope() << "in arnode";
        if (node.RemovePara(para) != SUCCESS) {
          MS_LOG(ERROR) << "remove para failed";
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

Status AllreduceGraph::set_head_cnode(const CNodePtr& node) {
  auto arnode = std::make_shared<AllreduceNode>(AllreduceNode());
  if (arnode->Init(node) != SUCCESS) {
    MS_LOG(ERROR) << "AllreduceNode Init failed";
  }
  head_cnode_ = node;
  cnode_arnode_map_[node] = arnode;
  auto arnode_emplace_return = arnode_set_.insert(arnode);
  if (!arnode_emplace_return.second) {
    MS_LOG(WARNING) << "node: " << node->DebugString() << "'s arnode has already been added!";
  }
  auto cnode_emplace_return = cnode_set_.emplace(node);
  if (!cnode_emplace_return.second) {
    MS_LOG(WARNING) << "node: " << node->DebugString() << " has already been added!";
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
