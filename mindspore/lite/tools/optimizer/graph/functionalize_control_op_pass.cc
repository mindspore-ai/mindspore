/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *conv_activation_fusion.h
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/optimizer/graph/functionalize_control_op_pass.h"
#include <algorithm>
#include <deque>
#include "tools/optimizer/graph/functionalize_while.h"
#include "tools/optimizer/graph/functionalize_cond.h"
#include "include/errorcode.h"

namespace mindspore::opt {

FuncGraphPtr FunctionalizeControlOpPass::NewFuncGraph(const std::string &subgraph_name, const FmkType &fmk_type) {
  auto fg = std::make_shared<FuncGraph>();
  if (fg == nullptr) {
    MS_LOG(ERROR) << "new func_graph failed.";
    return nullptr;
  }
  fg->set_attr("graph_name", MakeValue(subgraph_name));
  fg->set_attr("fmk", MakeValue(static_cast<int>(fmk_type)));
  return fg;
}

std::string FunctionalizeControlOpPass::NodeClusterName(const AnfNodePtr &node) {
  std::string cluster_name{};
  // tf node name use '/' split node name
  auto cnode = utils::cast<CNodePtr>(node);
  size_t pos = cnode->fullname_with_scope().rfind('/');
  if (pos != std::string::npos) {
    cluster_name = cnode->fullname_with_scope().substr(0, pos);
  } else {
    cluster_name = cnode->fullname_with_scope();
  }
  return cluster_name;
}

void FunctionalizeControlOpPass::InitNodeClusters(const FuncGraphPtr &func_graph) {
  for (auto &node : func_graph->nodes()) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cluster_name = NodeClusterName(node);
    auto cluster_pos = WhichCluster(cluster_name);
    if (cluster_pos == node_clusters_.size()) {
      std::vector<AnfNodePtr> node_list{node};
      node_clusters_.emplace_back(std::make_pair(cluster_name, node_list));
    } else {
      node_clusters_[cluster_pos].second.push_back(node);
    }
  }
  // sort node_clusters_
  std::sort(node_clusters_.begin(), node_clusters_.end(),
            [](std::pair<std::string, std::vector<AnfNodePtr>> a, std::pair<std::string, std::vector<AnfNodePtr>> b) {
              if (a.first.size() != b.first.size()) {
                return a.first.size() > b.first.size();
              } else {
                return a.first > b.first;
              }
            });
}

size_t FunctionalizeControlOpPass::WhichCluster(const std::string &cluster_name) {
  size_t pos = node_clusters_.size();
  for (size_t i = 0; i < pos; ++i) {
    if (node_clusters_[i].first == cluster_name) {
      return i;
    }
  }
  return pos;
}

STATUS FunctionalizeControlOpPass::BuildWhileSubgraph(const FuncGraphPtr &func_graph) {
  int ret = RET_OK;
  for (auto &node_cluster : node_clusters_) {
    for (auto &node : node_cluster.second) {
      if (IsLoopCond(node)) {
        loop_cond_nodes_.push_back(node->cast<CNodePtr>());
        FunctionalizeWhile fw(node_cluster.second, node->cast<CNodePtr>(), func_graph);
        ret = fw.Process();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "run functionalize while failed, ret: " << ret;
          return ret;
        }
        break;
      }
    }
  }
  return ret;
}

STATUS FunctionalizeControlOpPass::BuildIfSubgraph(const FuncGraphPtr &func_graph) {
  int ret = RET_OK;
  auto nodes = func_graph->nodes();
  for (auto &node : nodes) {
    if (!IsMerge(node)) {
      continue;
    }
    auto cnode = utils::cast<CNodePtr>(node);
    FunctionalizeCond fc(func_graph, cnode);
    ret = fc.Process();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run functionalize cond failed, ret: " << ret;
      return ret;
    }
  }

  return ret;
}

bool FunctionalizeControlOpPass::Run(const FuncGraphPtr &func_graph) {
  // use name to find the frame
  InitNodeClusters(func_graph);
  if (BuildWhileSubgraph(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "build while subgraph failed.";
    return false;
  }
  if (BuildIfSubgraph(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "build while subgraph failed.";
    return false;
  }
  return true;
}

CNodePtr FunctionalizeControlOpPass::BelongToWhichNode(const CNodePtr &node, const AimFunc &aim_func,
                                                       const FilterFunc &filter_func) {
  if (node == nullptr) {
    return nullptr;
  }
  if (aim_func(node)) {
    return node;
  }
  CNodePtr aim_node = nullptr;
  std::deque<AnfNodePtr> todo(256);
  todo.clear();
  for (auto &input_node : node->inputs()) {
    if (aim_func(input_node)) {
      aim_node = utils::cast<CNodePtr>(input_node);
      todo.clear();
      break;
    }
    todo.push_back(input_node);
  }

  while (!todo.empty()) {
    AnfNodePtr todo_node = todo.front();
    todo.pop_front();
    if (aim_func(todo_node)) {
      if (filter_func == nullptr) {
        aim_node = utils::cast<CNodePtr>(todo_node);
        todo.clear();
        break;
      } else if (filter_func(todo_node)) {
        aim_node = utils::cast<CNodePtr>(todo_node);
        todo.clear();
        break;
      }
    }
    if (utils::isa<CNodePtr>(todo_node)) {
      auto cnode = utils::cast<CNodePtr>(todo_node);
      for (size_t i = 0; i < cnode->inputs().size(); i++) {
        todo.push_back(cnode->input(i));
      }
    }
  }
  if (aim_node == nullptr) {
    MS_LOG(WARNING) << "not found belonging enter node.";
    return nullptr;
  }

  return aim_node;
}
}  // namespace mindspore::opt
