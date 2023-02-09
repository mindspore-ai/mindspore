/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
namespace opt {
namespace python_pass {
int64_t Pattern::g_id_ = 0;

MatchResultPtr Prim::match(const AnfNodePtr &node) {
  if (!IsValueNode<Primitive>(node)) {
    return nullptr;
  }
  MatchResultPtr res = std::make_shared<MatchResult>();
  // iterate over all primitives
  for (auto &iter : primitives_) {
    if (IsPrimitive(node, iter) || iter->name() == "*") {
      matched_prim_ = iter;
      res->add_entry(shared_from_base<Prim>(), node);
      return res;
    }
  }
  return nullptr;
}

MatchResultPtr Call::match(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node)) {
    return nullptr;
  }
  MatchResultPtr res = std::make_shared<MatchResult>();
  // IsPrimitiveCNode
  auto cnode = node->cast_ptr<CNode>();
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
    res->add_entry(shared_from_base<Call>(), cnode->input(0));
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
    res->add_entry(shared_from_base<Call>(), cnode->input(0));
    return res;
  }
  return nullptr;
}

MatchResultPtr OneOf::match(const AnfNodePtr &node) {
  for (auto &iter : patterns_) {
    auto res = iter->match(node);
    if (res != nullptr) {
      res->add_entry(shared_from_base<OneOf>(), node);
      return res;
    }
  }
  return nullptr;
}

MatchResultPtr NoneOf::match(const AnfNodePtr &node) {
  for (auto &iter : patterns_) {
    auto match_res = iter->match(node);
    if (match_res != nullptr) {
      return nullptr;
    }
  }
  auto res = std::make_shared<MatchResult>();
  res->add_entry(shared_from_base<NoneOf>(), node);
  return res;
}

MatchResultPtr Any::match(const AnfNodePtr &node) {
  MatchResultPtr res = std::make_shared<MatchResult>();
  res->add_entry(shared_from_base<Any>(), node);
  return res;
}

MatchResultPtr Imm::match(const AnfNodePtr &node) {
  auto value_ptr = GetValuePtr<Int32Imm>(node);
  if (value_ptr == nullptr) {
    return nullptr;
  }
  if (value_ptr->value() != value_) {
    return nullptr;
  }
  MatchResultPtr res = std::make_shared<MatchResult>();
  res->add_entry(shared_from_base<Imm>(), node);
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
  auto other_result_map = other_result->result();
  // add/update entries in other_result
  for (auto &iter : other_result_map) {
    match_result_[iter.first] = iter.second;
  }
}

void RegPattern(const py::module *m) {
  (void)py::class_<Pattern, std::shared_ptr<Pattern>>(*m, "Pattern").def(py::init<>());
  (void)py::class_<OneOf, std::shared_ptr<OneOf>, Pattern>(*m, "OneOf_").def(py::init<vector<PatternPtr>>());
  (void)py::class_<Prim, std::shared_ptr<Prim>, Pattern>(*m, "Prim_", py::dynamic_attr())
    .def(py::init<vector<py::object>, string>());
  (void)py::class_<Call, std::shared_ptr<Call>, Pattern>(*m, "Call_")
    .def(py::init<PatternPtr, vector<PatternPtr>>())
    .def(py::init<py::object, vector<PatternPtr>>());
  (void)py::class_<NoneOf, std::shared_ptr<NoneOf>, Pattern>(*m, "NoneOf_").def(py::init<vector<PatternPtr>>());
  (void)py::class_<Any, std::shared_ptr<Any>, Pattern>(*m, "Any").def(py::init<>());
  (void)py::class_<NewTensor, std::shared_ptr<NewTensor>, Pattern>(*m, "NewTensor_").def(py::init<tensor::TensorPtr>());
  (void)py::class_<NewParameter, std::shared_ptr<NewParameter>, Pattern>(*m, "NewParameter_")
    .def(py::init<string, tensor::TensorPtr, bool, bool>());
  (void)py::class_<Imm, std::shared_ptr<Imm>, Pattern>(*m, "Imm").def(py::init<int64_t>());
}
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
