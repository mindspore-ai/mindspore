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
#include "backend/optimizer/graph_kernel/split_model/split_model_cpu.h"
#include <memory>

namespace mindspore::graphkernel::inner {
enum class FuseType { kWidth, kDepth };
class FuseReshape : public FusePattern {
 public:
  FuseReshape() : FusePattern("reshape") {}
  ~FuseReshape() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return dom->pattern() == NodePattern::RESHAPE; }
  bool Match(const AreaPtr &dom) override {
    min_area_ = nullptr;
    // Reshape nodes have at most one user, which is graranteed by the pass "shape_ops_splitter".
    for (auto &user : dom->users()) {
      if (user->pattern() <= NodePattern::BROADCAST && !HasCircle(dom, user)) {
        KeepMinimumArea(user, FuseDirection::BACKWARD);
      }
    }

    for (auto &inp : dom->inputs()) {
      if (inp->is_output() || inp->user_num() > 1) {
        continue;
      }
      if (inp->pattern() <= NodePattern::BROADCAST && !HasCircle(inp, dom)) {
        KeepMinimumArea(inp, FuseDirection::FORWARD);
      }
    }
    if (min_area_ == nullptr) return false;
    (void)fused_areas_.emplace_back(min_area_);
    return true;
  }

  void KeepMinimumArea(const AreaPtr &a, FuseDirection dir) {
    if (min_area_ == nullptr || a->pattern() < min_area_->pattern()) {
      min_area_ = a;
      direction_ = dir;
    }
  }
  AreaPtr min_area_;
};

class FuseElemwiseBroadcastFwd : public FusePattern {
 public:
  explicit FuseElemwiseBroadcastFwd(FuseType fuse_type) : FusePattern("elemwise_broadcast_fwd"), fuse_type_(fuse_type) {
    direction_ = FuseDirection::FORWARD;
    name_ += (fuse_type == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseElemwiseBroadcastFwd() = default;
  static FusePatternPtr CreateDepthMatcher() { return std::make_shared<FuseElemwiseBroadcastFwd>(FuseType::kDepth); }
  static FusePatternPtr CreateWidthMatcher() { return std::make_shared<FuseElemwiseBroadcastFwd>(FuseType::kWidth); }

 protected:
  bool Check(const AreaPtr &dom) override {
    if (dom->pattern() != NodePattern::ELEMWISE && dom->pattern() != NodePattern::BROADCAST) {
      return false;
    }
    return fuse_type_ == FuseType::kWidth || dom->input_num() == 1;
  }
  bool Match(const AreaPtr &dom) override {
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
        if (a->dom()->shape == dom->dom()->shape) {
          (void)fused_areas_.emplace_back(a);
        }
      }
    }
    return !fused_areas_.empty();
  }

  FuseType fuse_type_;
};

class FuseReduceFwd : public FusePattern {
 public:
  FuseReduceFwd(FuseType fuse_type, size_t size_limit)
      : FusePattern("reduce_fwd"), fuse_type_(fuse_type), size_limit_(size_limit) {
    direction_ = FuseDirection::FORWARD;
    name_ += (fuse_type == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseReduceFwd() = default;
  static FusePatternPtr CreateDepthMatcher(size_t size_limit) {
    return std::make_shared<FuseReduceFwd>(FuseType::kDepth, size_limit);
  }
  static FusePatternPtr CreateWidthMatcher(size_t size_limit) {
    return std::make_shared<FuseReduceFwd>(FuseType::kWidth, size_limit);
  }

 protected:
  bool Check(const AreaPtr &dom) override {
    if (dom->pattern() != NodePattern::REDUCE) {
      return false;
    }
    return fuse_type_ == FuseType::kWidth || dom->input_num() == 1;
  }
  bool Match(const AreaPtr &dom) override {
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

  FuseType fuse_type_;
  size_t size_limit_;
};

class FuseElemwiseBroadcastBwd : public FusePattern {
 public:
  FuseElemwiseBroadcastBwd(FuseType fuse_type, size_t size_limit)
      : FusePattern("elemwise_broadcast_bwd"), fuse_type_(fuse_type), size_limit_(size_limit) {
    direction_ = FuseDirection::BACKWARD;
    name_ += (fuse_type == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseElemwiseBroadcastBwd() = default;
  static FusePatternPtr CreateDepthMatcher(size_t size_limit) {
    return std::make_shared<FuseElemwiseBroadcastBwd>(FuseType::kDepth, size_limit);
  }
  static FusePatternPtr CreateWidthMatcher(size_t size_limit) {
    return std::make_shared<FuseElemwiseBroadcastBwd>(FuseType::kWidth, size_limit);
  }

 protected:
  bool Check(const AreaPtr &dom) override {
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
  bool Match(const AreaPtr &dom) override {
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
        if (!fused_areas_.empty() && fused_areas_[0]->dom()->shape != a->dom()->shape) {
          return false;
        }
        if (HasCircle(dom, a)) {
          return false;
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

  FuseType fuse_type_;
  size_t size_limit_;
};

constexpr size_t kReduceFusionDepth = 20;
constexpr size_t kBroadcastFusionDepth = 20;

void SplitModelCpu::InitFusePatterns() {
  AddPattern(std::make_shared<FuseVirtualNode>(), true);
  AddPattern(std::make_shared<FuseReshape>(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateDepthMatcher(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateWidthMatcher(), true);
  AddPattern(FuseReduceFwd::CreateDepthMatcher(kReduceFusionDepth), true);
  AddPattern(FuseReduceFwd::CreateWidthMatcher(kReduceFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateDepthMatcher(kBroadcastFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateWidthMatcher(kBroadcastFusionDepth), true);
}

AreaMode SplitModelCpu::GetDefaultAreaMode(const PrimOpPtr &) const { return AreaMode::COMPOSITE; }
}  // namespace mindspore::graphkernel::inner
