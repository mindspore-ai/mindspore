/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/graph_fusion/split_model/split_model.h"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "abstract/symbolic_shape/int_symbol.h"

namespace mindspore::graphkernel::inner {
ReachTable::ReachTable(size_t size) : size_(size), reach_(size, std::vector<bool>(size, false)) {
  for (size_t i = 0; i < size_; ++i) {
    reach_[i][i] = true;
    (void)alive_.insert(i);
  }
}

void ReachTable::Link(size_t from, size_t to) {
  // if there's an edge <from, to>, the `from` can reach to `to`'s succeeding areas.
  // so we connect `from` to all succeeding areas of `to`.
  for (const size_t suc : alive_) {
    if (Reachable(to, suc)) {
      reach_[from][suc] = true;
    }
  }
}

void ReachTable::FuseArea(size_t target, size_t other) {
  // if `suc` is the succeeding nodes of other_node,
  // link the target_node's previous nodes to `suc`.
  for (const size_t suc : alive_) {
    if (Reachable(other, suc) && !Reachable(target, suc)) {
      for (const size_t pre : alive_) {
        if (Reachable(pre, target)) {
          reach_[pre][suc] = true;
        }
      }
    }
  }
  // if `pre` is the previous nodes of other_node,
  // link `pre` to target_node's succeeding nodes.
  for (const size_t pre : alive_) {
    if (Reachable(pre, other) && !Reachable(pre, target)) {
      for (const size_t suc : alive_) {
        if (Reachable(target, suc)) {
          reach_[pre][suc] = true;
        }
      }
    }
  }
  // discard other_node.
  (void)alive_.erase(other);
}

bool ReachTable::HasCircle(const AreaPtr &a, const AreaPtr &b) const {
  // a is the input of b
  if (Reachable(a->id(), b->id())) {
    // use `inputs_with_relation` instead of `inputs` to avoid generating a new vector.
    for (auto &inp : b->inputs_with_relation()) {
      if (inp.first != a && Reachable(a->id(), inp.first->id())) {
        return true;
      }
    }
  } else {
    // b is the input of a
    for (auto &inp : a->inputs_with_relation()) {
      if (inp.first != b && Reachable(b->id(), inp.first->id())) {
        return true;
      }
    }
  }
  return false;
}

AreaPtr SplitModel::NewArea(const PrimOpPtr &op, bool is_output) {
  auto new_area = std::make_shared<Area>(cur_area_id_++, op->As<PrimOp>(), is_output, node_area_map_);
  (void)areas_.emplace_back(new_area);
  node_area_map_[op] = new_area;
  SetDefaultAreaMode(new_area);
  UpdateAreaOutput(new_area);
  return new_area;
}

void SplitModel::AlignShape(const LiteGraphPtr &litegraph) const {
  for (auto &inp : litegraph->inputs()) {
    if (inp->shape.empty()) {
      inp->shape.push_back(1LL);
      if (inp->symbolic_shape != nullptr && inp->symbolic_shape->size() == 0) {
        inp->symbolic_shape = ListSymbol::Make({kSym1});
      }
    }
  }
  auto check_pattern = [](const NodePtr &op) {
    auto pn = op->As<PrimOp>()->compute_type();
    return pn == NodePattern::ELEMWISE || pn == NodePattern::BROADCAST || pn == NodePattern::REDUCE;
  };
  for (auto &op : litegraph->ops()) {
    if (!check_pattern(op)) {
      if (op->shape.empty()) {
        op->shape.push_back(1LL);
        if (op->symbolic_shape != nullptr && op->symbolic_shape->size() == 0) {
          op->symbolic_shape = ListSymbol::Make({kSym1});
        }
      }
      continue;
    }
    auto cur_shape_size = op->shape.size();
    for (auto &inp : op->inputs()) {
      if (inp->shape.size() > cur_shape_size) {
        cur_shape_size = inp->shape.size();
      }
    }
    if (cur_shape_size > op->shape.size()) {
      auto num = cur_shape_size - op->shape.size();
      (void)op->shape.insert(op->shape.cbegin(), num, 1LL);
      if (op->symbolic_shape != nullptr) {
        auto symbols = op->symbolic_shape->symbols();
        (void)symbols.insert(symbols.begin(), num, kSym1);
        op->symbolic_shape = ListSymbol::Make(std::move(symbols));
      }
    }
  }
}

void SplitModel::InitGraph(const LiteGraphPtr &litegraph) {
  AlignShape(litegraph);
  auto &outputs = litegraph->GetOutputs();
  HashSet<NodePtr> outputs_set(outputs.begin(), outputs.end());
  for (const auto &op : litegraph->ops()) {
    if (op->NodeType() != NType::Primitive) {
      MS_LOG(EXCEPTION) << "Op " << op->debug_name() << " should be a Primitive node, but got " << op->NodeType();
    }
    bool is_output = (outputs_set.count(op) > 0);
    (void)NewArea(op->As<PrimOp>(), is_output);
  }

  // Initialize reach table in reversed topological order
  reach_table_ = std::make_shared<ReachTable>(litegraph->ops().size());
  MS_EXCEPTION_IF_NULL(reach_table_);
  for (auto iter = areas_.rbegin(); iter != areas_.rend(); ++iter) {
    auto users = (*iter)->users();
    for (auto &user : users) {
      reach_table_->Link((*iter)->id(), user->id());
    }
  }
}

void SplitModel::AddPattern(const std::shared_ptr<FusePattern> &pn, bool enable) {
  auto &flags = GraphKernelFlags::GetInstance();
  if (!flags.enable_fusion_pattern_only.empty()) {
    if (std::none_of(flags.enable_fusion_pattern_only.begin(), flags.enable_fusion_pattern_only.end(),
                     [&pn](std::string only) { return pn->name().find(only) != std::string::npos; })) {
      enable = false;
      MS_LOG(INFO) << "Disable pattern by only flag: " << pn->name();
    }
  } else if (std::any_of(flags.disable_fusion_pattern.begin(), flags.disable_fusion_pattern.end(),
                         [&pn](std::string dis) { return pn->name().find(dis) != std::string::npos; })) {
    enable = false;
    MS_LOG(INFO) << "Disable pattern by disable flag: " << pn->name();
  }
  if (enable) {
    MS_LOG(INFO) << "Enable pattern: " << pn->name();
  }
  (void)patterns_.emplace_back(std::make_pair(pn, enable));
  patterns_.back().first->SetCircleChecker(reach_table_);
}

void SplitModel::LimitAreaSize(const AreaPtr &dom, std::vector<AreaPtr> *areas) const {
  uint64_t max_size = GraphKernelFlags::GetInstance().composite_op_limit_size;
  auto dom_size = dom->size();
  for (auto a = areas->begin(); a != areas->end(); ++a) {
    dom_size += (*a)->size();
  }
  if (dom_size <= max_size) {
    return;
  }
  // fuse the smaller area in priority
  std::sort(areas->begin(), areas->end(), [](const AreaPtr &a, const AreaPtr &b) { return a->size() < b->size(); });
  auto iter = std::find_if(areas->begin(), areas->end(), [cur_size = dom->size(), max_size](const AreaPtr &a) mutable {
    cur_size += a->size();
    return cur_size > max_size;
  });
  (void)areas->erase(iter, areas->cend());
}

void SplitModel::FuseAreas(const AreaPtr &dom, const std::vector<AreaPtr> &areas, FuseDirection direction) {
  if (areas.empty()) {
    return;
  }
  auto target = dom;
  if (direction == FuseDirection::BACKWARD) {
    for (auto a : areas) {
      // always use back node to fuse the front node.
      std::swap(target, a);
      target->FuseInput(a);
      reach_table_->FuseArea(target->id(), a->id());
    }
    for (auto &op : target->ops()) {
      node_area_map_[op] = target;
    }
  } else {
    for (auto a : areas) {
      for (auto &op : a->ops()) {
        node_area_map_[op] = target;
      }
      target->FuseInput(a);
      reach_table_->FuseArea(target->id(), a->id());
    }
  }
  if (target->pattern() > NodePattern::RESHAPE) {
    target->SetMode(AreaMode::COMPOSITE);
  }
  UpdateAreaOutput(target);
}

bool SplitModel::RunOnePattern(const FusePatternPtr &pattern) {
  // in one step, we only match the adjacent areas of the "area",
  // so if matched, we should handle the same area again in the next step
  bool changed = false;
  for (auto iter = areas_.begin(); iter != areas_.end();) {
    auto area = *iter;
    if (!area->IsAlive()) {
      iter = areas_.erase(iter);
      continue;
    }
    if (pattern->Run(area)) {
      MS_LOG(DEBUG) << "Area " << area->ToString() << " matches " << pattern->ToString();
      LimitAreaSize(area, &pattern->fused_areas_);
      if (!pattern->fused_areas_.empty()) {
        FuseAreas(area, pattern->fused_areas_, pattern->direction());
        changed = true;
        continue;
      }
    }
    ++iter;
  }
  return changed;
}

void SplitModel::RunFusePatterns() {
  // process one pattern for all areas before process next pattern.
  for (auto &[pattern, enable] : patterns_) {
    if (!enable) {
      continue;
    }
    MS_LOG(DEBUG) << "Run pattern " << pattern->name();
    (void)RunOnePattern(pattern);
  }
  // remove the areas that is fused
  for (auto iter = areas_.begin(); iter != areas_.end();) {
    if (!(*iter)->IsAlive()) {
      iter = areas_.erase(iter);
    } else {
      ++iter;
    }
  }
}

void SplitModel::UpdateAreaOutput(const AreaPtr &area) const {
  auto &area_outputs = area->area_outputs();
  area_outputs.clear();
  for (auto &ops : area->ops()) {
    for (auto &users : ops->users()) {
      auto iter = node_area_map_.find(users.first->shared_from_this());
      if (iter == node_area_map_.end() || iter->second.get() != area.get()) {
        (void)area_outputs.emplace_back(ops);
        break;
      }
    }
  }
}

void SplitModel::Run(const LiteGraphPtr &litegraph) {
  InitGraph(litegraph);
  InitFusePatterns();
  RunFusePatterns();
}
}  // namespace mindspore::graphkernel::inner
