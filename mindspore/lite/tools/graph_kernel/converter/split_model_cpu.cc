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
#include "tools/graph_kernel/converter/split_model_cpu.h"
#include <memory>
#include "utils/ms_context.h"

namespace mindspore::graphkernel::inner {
SPLIT_MODEL_REGISTER(kCPUDevice, SplitModelCpu);
constexpr size_t kReduceFusionDepth = 20;
constexpr size_t kBroadcastFusionDepth = 20;

class FuseElemwiseFwd : public FusePattern {
 public:
  explicit FuseElemwiseFwd(FuseType fuse_type) : FusePattern("elemwise_fwd"), fuse_type_(fuse_type) {
    direction_ = FuseDirection::FORWARD;
    name_ += (fuse_type == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseElemwiseFwd() = default;
  static FusePatternPtr CreateDepthMatcher() { return std::make_shared<FuseElemwiseFwd>(FuseType::kDepth); }
  static FusePatternPtr CreateWidthMatcher() { return std::make_shared<FuseElemwiseFwd>(FuseType::kWidth); }

 protected:
  bool Check(const AreaPtr &dom) override {
    if (dom->pattern() != NodePattern::ELEMWISE) {
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
      if (a->pattern() <= NodePattern::ELEMWISE && r == EdgeRelation::INJECTIVE) {
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

  FuseType fuse_type_;
};

class FuseConv : public FusePattern {
 public:
  FuseConv() : FusePattern("conv") { direction_ = FuseDirection::BACKWARD; }
  ~FuseConv() = default;

 protected:
  bool Check(const AreaPtr &dom) override {
    if (dom->dom()->op() != "Conv2D") {
      return false;
    }
    return true;
  }
  bool Match(const AreaPtr &dom) override {
    for (auto d : dom->users_with_relation()) {
      auto a = d.first;
      if (HasCircle(dom, a)) {
        continue;
      }
      if (a->pattern() < NodePattern::BROADCAST ||
          (a->pattern() == NodePattern::BROADCAST && a->dom()->shape == dom->dom()->shape)) {
        (void)fused_areas_.emplace_back(a);
      }
    }
    return !fused_areas_.empty();
  }

  FuseType fuse_type_;
};

void SplitModelCpu::InitFusePatterns() {
  AddPattern(std::make_shared<FuseVirtualNode>(), true);
  AddPattern(std::make_shared<FuseReshape>(), true);
  AddPattern(FuseElemwiseFwd::CreateDepthMatcher(), true);
  AddPattern(FuseElemwiseFwd::CreateWidthMatcher(), true);
  AddPattern(std::make_shared<FuseConv>(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateDepthMatcher(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateWidthMatcher(), true);
  AddPattern(FuseReduceFwd::CreateDepthMatcher(kReduceFusionDepth), true);
  AddPattern(FuseReduceFwd::CreateWidthMatcher(kReduceFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateDepthMatcher(kBroadcastFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateWidthMatcher(kBroadcastFusionDepth), true);
  AddPattern(std::make_shared<FuseIsolateReshape>(), true);
}

AreaMode SplitModelCpu::GetDefaultAreaMode(const PrimOpPtr &) const { return AreaMode::COMPOSITE; }
}  // namespace mindspore::graphkernel::inner
