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
#include "frontend/optimizer/pattern.h"
#include "pybind_api/api_register.h"
#include "pybind_api/export_flags.h"

namespace mindspore {
namespace opt {
namespace python_pass {
int Pattern::g_id_ = 0;

MatchResultPtr IsPrimTypeOf::match(const AnfNodePtr &node) {
  if (!IsValueNode<Primitive>(node)) {
    return nullptr;
  }
  MatchResultPtr res = std::make_shared<MatchResult>();
  if (IsValueNode<Primitive>(node)) {
    // iterate over all primitives
    for (auto &iter : primitives_) {
      if (IsPrimitive(node, iter) || iter->name() == "*") {
        matched_prim_ = iter;
        res->add_entry(shared_from_base<IsPrimTypeOf>(), node);
        return res;
      }
    }
  }
  return nullptr;
}

MatchResultPtr CallWith::match(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node)) {
    return nullptr;
  }
  MatchResultPtr res = std::make_shared<MatchResult>();
  // IsPrimitiveCNode
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Check Primitive ValueNode
  if (prim_pattern_ != nullptr) {
    // Passed in prim_pattern
    auto prim_value_res = prim_pattern_->match(cnode->input(0));
    if (prim_value_res == nullptr) {
      return nullptr;
    }
    res->merge(prim_value_res);
  } else if (prim_ != nullptr) {
    // Passed in primitive/primitive str
    if (!IsPrimitive(cnode->input(0), prim_)) {
      return nullptr;
    }
  } else {
    MS_LOG(EXCEPTION) << "Uninitialized CallWith pattern.";
  }
  // Check inputs
  auto p_inputs_size = inputs_.size();
  auto node_inputs_size = cnode->size() - 1;
  if (p_inputs_size != 0 && p_inputs_size != node_inputs_size) {
    return nullptr;
  }
  // If inputs is not specified, add node without looking into its inputs
  if (p_inputs_size == 0) {
    res->add_entry(shared_from_base<CallWith>(), cnode->input(0));
    return res;
  }
  bool failed = false;
  for (std::size_t i = 0; i < node_inputs_size; i++) {
    auto pattern = inputs_[i];
    auto input = cnode->input(i + 1);
    auto input_match_result = pattern->match(input);
    if (input_match_result == nullptr) {
      failed = true;
      break;
    }
    res->merge(input_match_result);
  }
  if (!failed) {
    res->add_entry(shared_from_base<CallWith>(), cnode->input(0));
    return res;
  }
  return nullptr;
}

MatchResultPtr IsIn::match(const AnfNodePtr &node) {
  for (auto &iter : patterns_) {
    auto res = iter->match(node);
    if (res != nullptr) {
      return res;
    }
  }
  return nullptr;
}

MatchResultPtr IsNot::match(const AnfNodePtr &node) {
  for (auto &iter : patterns_) {
    auto res = iter->match(node);
    if (res != nullptr) {
      return nullptr;
    }
  }
  auto res = std::make_shared<MatchResult>();
  res->add_entry(shared_from_base<IsNot>(), node);
  return res;
}

MatchResultPtr AnyPattern::match(const AnfNodePtr &node) {
  MatchResultPtr res = std::make_shared<MatchResult>();
  res->add_entry(shared_from_base<AnyPattern>(), node);
  return res;
}

AnfNodePtr MatchResult::get_node(const PatternPtr &pattern) {
  auto entry = match_result_.find(pattern);
  if (entry == match_result_.end()) {
    return nullptr;
  }
  return entry->second;
}

void MatchResult::merge(const MatchResultPtr &other_result) {
  auto other_result_map = other_result->_result();
  // add/update entries in other_result
  for (auto &iter : other_result_map) {
    match_result_[iter.first] = iter.second;
  }
}

REGISTER_PYBIND_DEFINE(
  Pattern, ([](const py::module *m) {
    (void)py::class_<Pattern, std::shared_ptr<Pattern>>(*m, "Pattern").def(py::init<>());
    (void)py::class_<IsIn, std::shared_ptr<IsIn>, Pattern>(*m, "IsIn_").def(py::init<vector<PatternPtr>>());
    (void)py::class_<IsPrimTypeOf, std::shared_ptr<IsPrimTypeOf>, Pattern>(*m, "IsPrimTypeOf_", py::dynamic_attr())
      .def(py::init<vector<PrimitivePyPtr>, string, bool>())
      .def(py::init<vector<string>, string, bool>());
    (void)py::class_<CallWith, std::shared_ptr<CallWith>, Pattern>(*m, "CallWith_")
      .def(py::init<PatternPtr, vector<PatternPtr>, bool>())
      .def(py::init<PrimitivePyPtr, vector<PatternPtr>, bool>())
      .def(py::init<string, vector<PatternPtr>, bool>());
    (void)py::class_<IsNot, std::shared_ptr<IsNot>, Pattern>(*m, "IsNot_").def(py::init<vector<PatternPtr>>());
    (void)py::class_<AnyPattern, std::shared_ptr<AnyPattern>, Pattern>(*m, "AnyPattern").def(py::init<>());
    (void)py::class_<NewTensor, std::shared_ptr<NewTensor>, Pattern>(*m, "NewTensor_")
      .def(py::init<tensor::TensorPtr>());
  }));
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
