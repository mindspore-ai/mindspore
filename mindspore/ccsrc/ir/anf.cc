/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "ir/anf.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "ir/func_graph.h"
#include "ir/primitive_base.h"

namespace mindspore {
// namespace to support intermediate representation definition
CNode::CNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph)
    : AnfNode(func_graph), inputs_(inputs), stop_gradient_(false) {}

// Check if CNode is an apply with the specific Primitive.
bool CNode::IsApply(const PrimitivePtr &value) const {
  if (value == nullptr) {
    return false;
  }

  if (inputs_.size() != 0 && IsValueNode<Primitive>(inputs_[0])) {
    PrimitivePtr fn_value = GetValueNode<PrimitivePtr>(inputs_[0]);
    if (fn_value->Hash() == value->Hash() && fn_value->name() == value->name()) {
      return true;
    }
  }

  return false;
}

void CNode::set_input(size_t i, const AnfNodePtr &new_input) { inputs_[i] = new_input; }

std::string CNode::DebugString(int recursive_level) const {
  std::ostringstream buffer;
  if (recursive_level > 0) {
    if (func_graph() != nullptr) {
      buffer << func_graph()->ToString() << ":";
    }
    buffer << ToString() << "{";
    bool is_first_node = true;
    int idx = 0;
    for (auto &node : inputs_) {
      MS_EXCEPTION_IF_NULL(node);
      if (is_first_node) {
        is_first_node = false;
      } else {
        buffer << ", ";
      }
      buffer << "[" << idx << "]: " << node->DebugString(recursive_level - 1);
      idx++;
    }
    buffer << "}";
  } else {
    buffer << ToString();
  }
  return buffer.str();
}

std::string ValueNode::ToString() const {
  MS_EXCEPTION_IF_NULL(value_);
  if (value_->isa<FuncGraph>()) {
    return value_->cast<FuncGraphPtr>()->ToString();
  }
  std::ostringstream buffer;
  buffer << AnfNode::ToString();
  buffer << "(" << value_->ToString() << ")";
  return buffer.str();
}

std::string ValueNode::DebugString(int) const {
  MS_EXCEPTION_IF_NULL(value_);
  std::ostringstream buffer;
  buffer << "ValueNode<" << value_->type_name() << "> " << value_->ToString();
  return buffer.str();
}

std::string ValueNode::fullname_with_scope() {
  if (!fullname_with_scope_.empty()) {
    return fullname_with_scope_;
  }

  MS_EXCEPTION_IF_NULL(scope());
  fullname_with_scope_ = scope()->name() + "/" + "data-" + id_generator::get_id(shared_from_base<ValueNode>());
  return fullname_with_scope_;
}

bool IsPrimitiveCNode(const AnfNodePtr &node, const PrimitivePtr &value) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode != nullptr) {
    return cnode->IsApply(value);
  }
  return false;
}

PrimitivePtr GetCNodePrimitive(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode != nullptr) {
    if (cnode->size() > 0) {
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      return prim;
    }
  }
  return nullptr;
}

std::string GetCNodeFuncName(const CNodePtr cnode) {
  if (cnode->inputs().empty()) {
    return "";
  }

  AnfNodePtr valuenode = cnode->input(0);
  if (valuenode->isa<ValueNode>()) {
    auto value = GetValueNode(valuenode);
    // check whether the valuenode is primitive
    if (value->isa<Primitive>()) {
      return value->cast<PrimitivePtr>()->name();
    }
    return value->ToString();
  }
  return "";
}

bool IsPrimitive(const AnfNodePtr &node, const PrimitivePtr &value) {
  if (IsValueNode<Primitive>(node)) {
    PrimitivePtr fn_value = GetValueNode<PrimitivePtr>(node);
    MS_EXCEPTION_IF_NULL(value);
    if (fn_value->Hash() == value->Hash() && fn_value->name() == value->name()) {
      return true;
    }
  }
  return false;
}

size_t NewSeenGeneration() {
  static size_t seen_generation = 0;
  return ++seen_generation;
}

namespace id_generator {
static std::unordered_map<std::string, int> node_ids;
std::string get_id(const AnfNodePtr &node) {
  auto type_name = node->type_name();
  if (node_ids.find(type_name) == node_ids.end()) {
    node_ids[type_name] = 0;
  } else {
    node_ids[type_name]++;
  }
  return std::to_string(node_ids[type_name]);
}

void reset_id() { node_ids.clear(); }
}  // namespace id_generator
}  // namespace mindspore
