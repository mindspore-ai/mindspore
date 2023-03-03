/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/split_model/fuse_pattern.h"
#include <sstream>

namespace mindspore::graphkernel::inner {
bool FuseReshape::Match(const AreaPtr &dom) {
  min_area_ = nullptr;
  // Reshape nodes have at most one user, which is graranteed by the pass "shape_ops_splitter".
  for (auto &user : dom->users()) {
    if (user->pattern() <= NodePattern::BROADCAST && !HasCircle(dom, user)) {
      KeepMinimumArea(user, FuseDirection::BACKWARD);
    }
  }

  for (auto &inp : dom->inputs()) {
    if (inp->pattern() <= NodePattern::BROADCAST && !HasCircle(inp, dom)) {
      KeepMinimumArea(inp, FuseDirection::FORWARD);
    }
  }
  if (min_area_ == nullptr) {
    return false;
  }
  (void)fused_areas_.emplace_back(min_area_);
  return true;
}

void FuseReshape::KeepMinimumArea(const AreaPtr &a, FuseDirection dir) {
  if (min_area_ == nullptr || a->pattern() < min_area_->pattern()) {
    min_area_ = a;
    direction_ = dir;
  }
}

bool FuseIsolateReshape::Match(const AreaPtr &dom) {
  for (auto &user : dom->users()) {
    if (user->mode() == AreaMode::COMPOSITE && !HasCircle(dom, user)) {
      (void)fused_areas_.emplace_back(user);
      direction_ = FuseDirection::BACKWARD;
      return true;
    }
  }
  for (auto &inp : dom->inputs()) {
    if (inp->mode() == AreaMode::COMPOSITE && !HasCircle(inp, dom)) {
      (void)fused_areas_.emplace_back(inp);
      direction_ = FuseDirection::FORWARD;
      return true;
    }
  }
  return false;
}

bool FuseElemwiseBroadcastFwd::Check(const AreaPtr &dom) {
  if (dom->pattern() != NodePattern::ELEMWISE && dom->pattern() != NodePattern::BROADCAST) {
    return false;
  }
  return fuse_type_ == FuseType::kWidth || dom->input_num() == 1;
}

bool FuseElemwiseBroadcastFwd::Match(const AreaPtr &dom) {
  for (auto &[a, r] : dom->inputs_with_relation()) {
    // depth match only support one to one pattern
    if (fuse_type_ == FuseType::kDepth && a->user_num() != 1) {
      continue;
    }
    if (a->pattern() <= NodePattern::BROADCAST && r == EdgeRelation::INJECTIVE) {
      // it's unnecessary to check circle for depth match
      if (fuse_type_ == FuseType::kWidth && HasCircle(a, dom)) {
        continue;
      }
      if (a->compute_size() == dom->compute_size()) {
        (void)fused_areas_.emplace_back(a);
      }
    }
  }
  return !fused_areas_.empty();
}

bool FuseReduceFwd::Check(const AreaPtr &dom) {
  if (dom->pattern() != NodePattern::REDUCE) {
    return false;
  }
  return fuse_type_ == FuseType::kWidth || dom->input_num() == 1;
}

bool FuseReduceFwd::Match(const AreaPtr &dom) {
  for (auto &[a, r] : dom->inputs_with_relation()) {
    if (fuse_type_ == FuseType::kDepth && a->user_num() != 1) {
      continue;
    }
    if (a->size() > size_limit_) {
      continue;
    }
    if (a->pattern() <= NodePattern::ELEMWISE && r == EdgeRelation::INJECTIVE) {
      // it's unnecessary to check circle for depth match
      if (fuse_type_ == FuseType::kWidth && HasCircle(a, dom)) {
        continue;
      }
      (void)fused_areas_.emplace_back(a);
    }
  }
  return !fused_areas_.empty();
}

bool FuseElemwiseBroadcastBwd::Check(const AreaPtr &dom) {
  if (dom->pattern() != NodePattern::ELEMWISE && dom->pattern() != NodePattern::BROADCAST) {
    return false;
  }
  if (dom->is_output()) {
    return false;
  }
  if (fuse_type_ == FuseType::kDepth && dom->user_num() > 1) {
    return false;
  }
  return dom->size() <= size_limit_;
}

bool FuseElemwiseBroadcastBwd::Match(const AreaPtr &dom) {
  // this pattern is to fuse ALL users of dom area,
  // since the broadcast node should not be an output when it fuse nodes in backward.
  for (auto &[a, r] : dom->users_with_relation()) {
    if (fuse_type_ == FuseType::kDepth && a->input_num() != 1) {
      return false;
    }
    if (a->pattern() > NodePattern::REDUCE) {
      return false;
    }
    if (fuse_type_ == FuseType::kWidth) {
      if (!fused_areas_.empty() && fused_areas_[0]->compute_size() != a->compute_size()) {
        return false;
      }
      if (HasCircle(dom, a)) {
        continue;
      }
    }
    if (a->pattern() == NodePattern::REDUCE) {
      // elemwise + reduce
      if (dom->pattern() == NodePattern::ELEMWISE && r == EdgeRelation::INJECTIVE) {
        (void)fused_areas_.emplace_back(a);
      } else {
        return false;
      }
    } else {  // a->pattern() < NodePattern::REDUCE
      (void)fused_areas_.emplace_back(a);
    }
  }
  return fused_areas_.size() == dom->user_num();
}

std::string FusePattern::ToString() const {
  std::ostringstream oss;
  if (direction_ == FuseDirection::FORWARD) {
    oss << "Forward{";
  } else {
    oss << "Backward{";
  }
  bool first = true;
  for (auto &area : fused_areas_) {
    if (first) {
      first = false;
    } else {
      oss << ",";
    }
    oss << area->ToString();
  }
  oss << "}";
  return oss.str();
}

bool FuseVirtualNode::Match(const AreaPtr &area) {
  fused_areas_ = area->inputs();
  return true;
}
}  // namespace mindspore::graphkernel::inner
