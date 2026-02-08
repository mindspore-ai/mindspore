/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/graph_fusion/split_model/area.h"
#include <algorithm>
#include <sstream>
#include "abstract/symbolic_shape/int_symbol.h"

namespace mindspore::graphkernel::inner {
namespace {
bool ShapeEqual(const NodePtr &a, const NodePtr &b, bool skip_leading_one = true) {
  MS_EXCEPTION_IF_NULL(a);
  MS_EXCEPTION_IF_NULL(b);
  auto l = a->shape.size() < b->shape.size() ? b : a;
  auto s = a->shape.size() < b->shape.size() ? a : b;
  auto l_shape = l->shape;
  auto s_shape = s->shape;
  auto l_symbol_shape = l->symbolic_shape;
  auto s_symbol_shape = s->symbolic_shape;
  bool use_symbol = (l_symbol_shape != nullptr && s_symbol_shape != nullptr);
  if (IsDynamicRank(l_shape)) {
    return use_symbol ? (l_symbol_shape == s_symbol_shape) : false;
  }
  auto diff = l_shape.size() - s_shape.size();
  if (diff != 0 && !skip_leading_one) {
    // shapes with different rank
    return false;
  }
  // check leading one
  for (size_t i = 0; i < diff; ++i) {
    if (l_shape[i] == 1 || (l_shape[i] < 0 && l_symbol_shape != nullptr && l_symbol_shape->item(i)->EqualsTo(kSym1))) {
      continue;
    }
    return false;
  }
  // check other dimensions
  for (size_t i = 0; i < s_shape.size(); ++i) {
    auto il = i + diff;
    if (l_shape[il] < 0 || s_shape[i] < 0) {
      if (use_symbol && l_symbol_shape->item(il)->EqualsTo(s_symbol_shape->item(i))) {
        continue;
      }
      return false;
    } else if (l_shape[il] != s_shape[i]) {
      return false;
    }
  }
  return true;
}

EdgeRelation GetRelation(const PrimOpPtr &node, const NodePtr &input) {
  if (node->compute_type() != NodePattern::ELEMWISE) {
    return node->compute_type() == NodePattern::BROADCAST ? EdgeRelation::BROADCAST : EdgeRelation::INJECTIVE;
  }
  if (node->inputs().size() == 1) {
    // single input elemwise op has no broadcast
    return EdgeRelation::INJECTIVE;
  }
  if (IsDynamic(input->shape)) {
    if (std::all_of(node->inputs().begin(), node->inputs().end(),
                    [input](const NodePtr &inp) { return inp == input; })) {
      return EdgeRelation::INJECTIVE;
    }
  }
  // naively set the edge relation to "broadcast" if the result shape is not equal to the input shape.
  return ShapeEqual(node, input) ? EdgeRelation::INJECTIVE : EdgeRelation::BROADCAST;
}

bool SameArea(const AreaWithRelation &a, const AreaWithRelation &b) { return a.first == b.first; }

bool AreaWithRelationCmp(const AreaWithRelation &a, const AreaWithRelation &b) {
  // for same areas, put the area with greater EdgeRelation in front when sorting.
  // compare the areas with unique id, instead of Area pointer, to avoid random result.
  return SameArea(a, b) ? (a.second > b.second) : (a.first->id() < b.first->id());
}
}  // namespace

Area::Area(size_t id, const PrimOpPtr &prim_op, bool is_output, const HashMap<NodePtr, AreaPtr> &node_area_map)
    : hd_(new NodeHandle(this, prim_op)), unique_id_(id), is_output_(is_output), ops_(1, prim_op) {
  // link inputs of the handle node
  auto init_pattern = pattern();
  for (auto &inp : prim_op->inputs()) {
    auto input_relation = GetRelation(prim_op, inp);
    if (init_pattern == NodePattern::ELEMWISE && input_relation == EdgeRelation::BROADCAST) {
      hd_->compute_type_ = NodePattern::BROADCAST;
    }
    if (auto inp_area_iter = node_area_map.find(inp); inp_area_iter != node_area_map.end()) {
      (void)inputs_with_relation_.emplace_back(std::make_pair(inp_area_iter->second, input_relation));
    }
  }
  // ELEMWISE if op has one variable input, other inputs are const input with shape [1]
  // e.g. Cast(out_0, 43)
  //      Add(param0, const)
  if (hd_->compute_type_ == NodePattern::BROADCAST && init_pattern == NodePattern::ELEMWISE) {
    size_t scalar_input_num = 0;
    auto input_num = prim_op->inputs().size();
    for (size_t i = 0; i < input_num; ++i) {
      auto inp = prim_op->inputs()[i];
      if (inp != nullptr && inp->tensor_size() == 1 &&
          (inp->NodeType() == NType::Tensor || inp->NodeType() == NType::Scalar)) {
        scalar_input_num++;
      }
    }
    if (scalar_input_num + 1 == input_num) {
      hd_->compute_type_ = NodePattern::ELEMWISE;
      if (!inputs_with_relation_.empty()) {
        inputs_with_relation_[0].second = EdgeRelation::INJECTIVE;
      }
    }
  }
  MakeUniqueAndSyncInputs();
}

std::vector<AreaPtr> Area::inputs() const {
  std::vector<AreaPtr> result;
  (void)std::transform(inputs_with_relation_.begin(), inputs_with_relation_.end(), std::back_inserter(result),
                       [](const AreaWithRelation &inp) { return inp.first; });
  return result;
}

std::vector<AreaPtr> Area::users() const {
  std::vector<AreaPtr> result;
  (void)std::transform(hd_->users().begin(), hd_->users().end(), std::back_inserter(result), [](const auto &u) {
    Node *node = u.first;
    return node->As<NodeHandle>()->area();
  });
  return result;
}

std::vector<AreaWithRelation> Area::users_with_relation() const {
  std::vector<AreaWithRelation> result;
  (void)std::transform(hd_->users().begin(), hd_->users().end(), std::back_inserter(result), [](const auto &u) {
    Node *node = u.first;
    auto area = node->As<NodeHandle>()->area();
    // the input edge of area is unique
    const auto relation = area->input_relation(*(u.second.begin()));
    return std::make_pair(area, relation);
  });
  return result;
}

int64_t Area::compute_size() const {
  auto op = dom();
  MS_EXCEPTION_IF_NULL(op);
  return SizeToLong(op->tensor_size());
}

bool Area::ComputeSizeEqual(const AreaPtr &other) const {
  if (other == nullptr) {
    return false;
  }
  auto op = dom();
  auto other_op = other->dom();
  if (op == nullptr || other_op == nullptr) {
    return false;
  }
  auto op_shape = op->shape;
  auto other_op_shape = other_op->shape;
  if (!IsDynamic(op_shape) && !IsDynamic(other_op_shape)) {
    return compute_size() == other->compute_size();
  }
  return ShapeEqual(op, other_op);
}

std::string Area::ToString() const {
  std::ostringstream oss;
  bool is_first = true;
  oss << "<";
  for (auto op : ops_) {
    if (is_first) {
      is_first = false;
      oss << id() << ":";
    } else {
      oss << "-";
    }
    oss << op->debug_name();
  }
  oss << ">";
  return oss.str();
}

void Area::MakeUniqueAndSyncInputs() {
  // remove the repeated inputs, keep the area with greater EdgeRelation.
  std::sort(inputs_with_relation_.begin(), inputs_with_relation_.end(), AreaWithRelationCmp);
  (void)inputs_with_relation_.erase(std::unique(inputs_with_relation_.begin(), inputs_with_relation_.end(), SameArea),
                                    inputs_with_relation_.cend());
  // sync the inputs to NodeHandle to maintain users
  this->hd_->ClearInputs();
  (void)std::for_each(inputs_with_relation_.begin(), inputs_with_relation_.end(),
                      [this](const AreaWithRelation &inp) { this->hd_->AddInput(inp.first->hd_); });
}

void Area::UpdateUsersRelation(const AreaPtr &input_area) {
  auto &user_node_with_index = input_area->hd_->users();
  std::vector<AreaPtr> user_areas;
  for (auto &[user_hd, index] : user_node_with_index) {
    (void)user_areas.emplace_back(user_hd->As<NodeHandle>()->area());
    const auto idx = *(index.begin());
    user_areas.back()->inputs_with_relation_[idx].first = this->shared_from_this();
  }
  // the inputs should be updated outside the above for-loop,
  // since the users cannot be updated while traversing.
  for (auto user : user_areas) {
    user->MakeUniqueAndSyncInputs();
  }
}

void Area::FuseInput(const AreaPtr &input_area) {
  auto iter = std::find_if(inputs_with_relation_.begin(), inputs_with_relation_.end(),
                           [&input_area](const AreaWithRelation &a) { return a.first == input_area; });
  if (iter == inputs_with_relation_.end()) {
    MS_LOG(EXCEPTION) << "The area " << input_area->ToString() << " should be the input of area " << this->ToString();
  }
  auto input_idx = LongToSize(iter - inputs_with_relation_.begin());

  if (input_area->is_output_) {
    is_output_ = true;
  }

  // Update ops, and discard the input_area's ops.
  // The dominant node is ops[0], keep the dominant with greater pattern.
  if (pattern() < input_area->pattern()) {
    ops_.swap(input_area->ops_);
  }
  (void)ops_.insert(ops_.cend(), input_area->ops_.cbegin(), input_area->ops_.cend());

  // update area pattern
  hd_->compute_type_ = std::max(pattern(), input_area->pattern());
  if ((pattern() == NodePattern::ELEMWISE) && (input_relation(input_idx) == EdgeRelation::BROADCAST)) {
    hd_->compute_type_ = NodePattern::BROADCAST;
  }

  // update inputs and relations
  (void)inputs_with_relation_.erase(iter);
  (void)inputs_with_relation_.insert(inputs_with_relation_.cend(), input_area->inputs_with_relation_.cbegin(),
                                     input_area->inputs_with_relation_.cend());
  MakeUniqueAndSyncInputs();
  UpdateUsersRelation(input_area);

  // clear the input_area.
  input_area->ops_.clear();
  input_area->inputs_with_relation_.clear();
  input_area->hd_->ClearInputs();
}
}  // namespace mindspore::graphkernel::inner
