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
#include "frontend/optimizer/py_pass.h"
#include <unordered_set>
#include <deque>
#include <algorithm>
#include <utility>
#include <vector>

#include "ir/func_graph.h"
#include "ir/manager.h"
#include "pipeline/jit/parse/parse_base.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace opt {
namespace python_pass {
namespace internal {
std::string GetNodeRepr(AnfNodePtr node) {
  if (node != nullptr) {
    if (node->isa<CNode>()) {
      std::string repr = "(";
      auto const &inputs = node->cast<CNodePtr>()->inputs();
      for (auto &input : inputs) {
        repr += " ";
        repr += GetNodeRepr(input);
        repr += " ";
      }
      repr += ")";
      return repr;
    }
    if (node->isa<ValueNode>()) {
      return GetValueNode(node)->ToString();
    }
    return node->ToString();
  }
  return "";
}

void ResolveFuncGraph_(const FuncGraphPtr &fg) {
  auto manager = Manage(fg, false);
  parse::python_adapter::set_use_signature_in_resolve(false);
  parse::ResolveAll(manager);
  parse::python_adapter::set_use_signature_in_resolve(true);
}

bool Match(const AnfNodePtr &pattern, const AnfNodePtr &node, const NodeEquivPtr &equiv_ptr) {
  if (node == nullptr) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(pattern);
  if (pattern->isa<ValueNode>()) {
    if (!node->isa<ValueNode>()) {
      return false;
    }
    if (GetNodeRepr(pattern) == GetNodeRepr(node)) {
      // add to equiv_ptr
      equiv_ptr->insert(std::make_pair(GetValueNode(pattern)->ToString(), node));
      return true;
    }
    return false;
  } else if (pattern->isa<Parameter>()) {
    MS_LOG(DEBUG) << pattern->ToString() + "\n";
    // add to equiv_ptr
    equiv_ptr->insert(std::make_pair(pattern->ToString(), node));
    return true;
  } else if (pattern->isa<CNode>()) {
    // match every single sub ANode
    if (!node->isa<CNode>()) {
      return false;
    }
    auto pattern_inputs = pattern->cast<CNodePtr>()->inputs();
    auto node_inputs = node->cast<CNodePtr>()->inputs();
    if (pattern_inputs.size() != node_inputs.size()) {
      return false;
    }
    for (auto p_item = pattern_inputs.begin(), node_item = node_inputs.begin(); p_item != pattern_inputs.end();
         p_item++, node_item++) {
      auto res = Match(*p_item, *node_item, equiv_ptr);
      if (!res) {
        return false;
      }
    }
    return true;
  }
  MS_LOG(EXCEPTION) << "Unexpected condition, (" + pattern->ToString() + " , " + node->ToString() + ")\n";
}

AnfNodePtr BuildTarget(const FuncGraphPtr &func_graph, const AnfNodePtr cur_raw_dst_node_,
                       const NodeEquivPtr &equiv_ptr) {
  if (cur_raw_dst_node_->isa<Parameter>()) {
    auto sub_pair = equiv_ptr->find(cur_raw_dst_node_->ToString());
    if (sub_pair != equiv_ptr->end()) {
      return sub_pair->second;
    }
    MS_LOG(EXCEPTION) << "cur_raw_dst_node_ : " + internal::GetNodeRepr(cur_raw_dst_node_) + "\n";
  } else if (cur_raw_dst_node_->isa<ValueNode>()) {
    // check primitive ValueNode
    auto sub_pair = equiv_ptr->find(cur_raw_dst_node_->cast<ValueNodePtr>()->value()->ToString());
    if (sub_pair != equiv_ptr->end()) {
      return sub_pair->second;
    }
    return cur_raw_dst_node_;
  } else if (cur_raw_dst_node_->isa<CNode>()) {
    std::vector<AnfNodePtr> new_inputs;
    auto inputs = cur_raw_dst_node_->cast<CNodePtr>()->inputs();
    for (auto sub_node = inputs.begin(); sub_node != inputs.end(); sub_node++) {
      auto subed = internal::BuildTarget(func_graph, *sub_node, equiv_ptr);
      new_inputs.push_back(subed);
    }
    return func_graph->NewCNode(new_inputs);
  }
  MS_LOG(EXCEPTION) << "Unexpected node type, got : " + internal::GetNodeRepr(cur_raw_dst_node_);
}

bool isTraversable(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (node->isa<CNode>() || node->isa<Parameter>()) {
    return true;
  }
  if (IsValueNode<FuncGraph>(node) || IsValueNode<RefKey>(node)) {
    return true;
  }
  return false;
}
}  // namespace internal

void PythonPass::Build(const py::function &src, const py::function &dst) {
  // 1. get FuncGraph from py::function
  auto src_fg_ = parse::ParsePythonCode(src);
  auto dst_fg_ = parse::ParsePythonCode(dst);
  if (src_fg_ == nullptr || dst_fg_ == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to parse python code.\n";
  }
  // 2. Resolve
  internal::ResolveFuncGraph_(src_fg_);
  internal::ResolveFuncGraph_(dst_fg_);
  // 3. from FuncGraphPtr to ValueNode
  src_node_ = src_fg_->output();
  dst_node_ = dst_fg_->output();
}

PythonPass::PythonPass(const std::string &name, const py::function &src, const py::function &dst, bool run_only_once,
                       bool multigraph)
    : name_(name), run_only_once_(run_only_once), multigraph_(multigraph) {
  Build(src, dst);
}

AnfNodePtr PythonPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  auto equiv_ptr = std::make_shared<NodeEquiv>();
  bool is_a_match = internal::Match(src_node_, node, equiv_ptr);
  if (is_a_match) {
    auto new_node = internal::BuildTarget(func_graph, dst_node_, equiv_ptr);
    MS_LOG(DEBUG) << "To be replaced node: " + internal::GetNodeRepr(new_node) + "\n";
    return new_node;
  }
  return nullptr;
}

bool PythonPass::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);
  auto seen = NewSeenGeneration();
  // 1024 is for the initial capacity of deque
  std::deque<AnfNodePtr> todo(1024);
  todo.push_back(func_graph->output());
  bool changes = false;

  auto &all_nodes = manager->all_nodes();
  while (!todo.empty()) {
    AnfNodePtr node = todo.front();
    todo.pop_front();

    // check whether this node has been matched.
    if (node == nullptr || node->seen_ == seen || !internal::isTraversable(node) || !all_nodes.contains(node)) {
      continue;
    }
    node->seen_ = seen;

    // select nodes that this transform can be applied.
    AnfNodePtr new_node = Run(func_graph, node);
    bool change = (new_node != nullptr);
    if (new_node != nullptr && new_node != node) {
      (void)manager->Replace(node, new_node);
    } else if (new_node == nullptr) {
      new_node = node;
    }
    if (run_only_once_) {
      return change;
    }

    // find success, and add them to todo list
    if (IsValueNode<FuncGraph>(node)) {
      todo.push_back(GetValueNode<FuncGraphPtr>(node)->output());
    }

    if (node->isa<CNode>()) {
      auto &inputs = node->cast<CNodePtr>()->inputs();
      (void)std::copy(inputs.begin(), inputs.end(), std::back_inserter(todo));
    }

    auto &node_users = manager->node_users();
    if (change && node_users.find(node) != node_users.end()) {
      for (auto &use : node_users[node]) {
        auto use_node = use.first;
        if (use_node == nullptr) {
          continue;
        }
        todo.push_back(use_node);
        if (use_node->seen_ == seen) {
          use_node->seen_--;
        }
      }
    }
  }
  return changes;
}
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
