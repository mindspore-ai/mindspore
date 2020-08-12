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
#include "pybind_api/ir/primitive_py.h"
#include "pipeline/jit/parse/parse_base.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace opt {
namespace python_pass {
namespace internal {
AnfNodePtr ProcessSinglePattern(const PatternPtr &pattern, const MatchResultPtr &res);

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

bool IsTraversable(const AnfNodePtr &node) {
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

AnfNodePtr BuildPrimitive(const PatternPtr &pattern, const MatchResultPtr &res) {
  // Build up AnfNode from primitive
  auto prim_pattern = pattern->cast<IsPrimTypeOfPtr>();
  MS_EXCEPTION_IF_NULL(prim_pattern);
  PrimitivePyPtr prim = prim_pattern->matched_primitive();
  MS_EXCEPTION_IF_NULL(prim);
  // Make value node out of primitives
  return std::make_shared<ValueNode>(prim);
}

AnfNodePtr BuildNewTensor(const PatternPtr &pattern, const MatchResultPtr &res) {
  // Build a ValueNode from TensorPtr
  auto new_tensor_pattern = pattern->cast<NewTensorPtr>();
  MS_EXCEPTION_IF_NULL(new_tensor_pattern);
  auto input_tensor = new_tensor_pattern->input_tensor();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return std::make_shared<ValueNode>(input_tensor);
}

AnfNodePtr BuildPrimitiveValueNode(const PatternPtr &pattern, const MatchResultPtr &res) {
  auto call_with_pattern = pattern->cast<CallWithPtr>();
  MS_EXCEPTION_IF_NULL(call_with_pattern);
  auto prim = call_with_pattern->prim_value();
  if (prim != nullptr) {
    return std::make_shared<ValueNode>(prim);
  }
  auto prim_pattern = call_with_pattern->prim_pattern();
  MS_EXCEPTION_IF_NULL(prim_pattern);
  return ProcessSinglePattern(prim_pattern, res);
}

AnfNodePtr ProcessSinglePattern(const PatternPtr &pattern, const MatchResultPtr &res) {
  if (pattern->should_replace()) {
    // Find replacement in the MatchResult
    auto target_node = res->get_node(pattern);
    if (target_node == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot find target node in pattern match result, pattern: " + pattern->unique_name() + "\n";
    }
    return target_node;
  }
  // Build up new node from pattern
  if (pattern->isa<IsPrimTypeOf>()) {
    return BuildPrimitive(pattern, res);
  } else if (pattern->isa<NewTensor>()) {
    return BuildNewTensor(pattern, res);
  } else if (pattern->isa<CallWith>()) {
    return BuildPrimitiveValueNode(pattern, res);
  }
  return nullptr;
}

AnfNodePtr BuildTarget(const PatternPtr &pattern, const FuncGraphPtr &func_graph, const MatchResultPtr &res) {
  auto target_inputs = pattern->inputs();
  if (target_inputs.size() == 0) {
    return ProcessSinglePattern(pattern, res);
  }
  // Build up the AnfNode in a recursive manner
  std::vector<AnfNodePtr> new_inputs;
  auto prim_value_node = ProcessSinglePattern(pattern, res);
  MS_EXCEPTION_IF_NULL(prim_value_node);
  new_inputs.push_back(prim_value_node);
  for (auto &iter : target_inputs) {
    if (iter == pattern) {
      MS_LOG(EXCEPTION) << "Circle references: Pattern takes itself as input. Got pattern: " + pattern->unique_name() +
                             "\n";
    }
    new_inputs.push_back(BuildTarget(iter, func_graph, res));
  }
  return func_graph->NewCNode(new_inputs);
}
}  // namespace internal

AnfNodePtr PythonPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(src_pattern_);
  MS_EXCEPTION_IF_NULL(dst_pattern_);
  auto res = src_pattern_->match(node);
  if (res != nullptr) {
    res->dump();
    MS_LOG(WARNING) << "Matched pattern: " + src_pattern_->unique_name();
    auto new_node = internal::BuildTarget(dst_pattern_, func_graph, res);
    dst_pattern_->reset();
    MS_LOG(DEBUG) << "To be replaced node: " + internal::GetNodeRepr(new_node) + "\n";
    return new_node;
  }
  src_pattern_->reset();
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
    // Check whether this node has been matched.
    if (node == nullptr || node->seen_ == seen || !internal::IsTraversable(node) || !all_nodes.contains(node)) {
      continue;
    }
    node->seen_ = seen;
    // Select nodes that this transform can be applied.
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
    // Find success, and add them to todo list
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
