/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/graph_kernel/converter/split_model_ascend.h"
#include <memory>
#include "utils/ms_context.h"

namespace mindspore::graphkernel::inner {
SPLIT_MODEL_REGISTER("Ascend", SplitModelAscend);

constexpr size_t kReduceFusionDepth = 10;
constexpr size_t kBroadcastFusionDepth = 6;

class FuseLayerNorm : public FusePattern {
 public:
  FuseLayerNorm() : FusePattern("layer_norm") { direction_ = FuseDirection::BACKWARD; }
  ~FuseLayerNorm() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return (dom->dom()->op() == "ReduceSum"); }
  bool Match(const AreaPtr &dom) override {
    constexpr size_t c1 = 1;
    constexpr size_t c2 = 2;
    auto users = dom->users();
    if (users.size() != c1 || users[0]->pattern() != NodePattern::BROADCAST) {
      return false;
    }
    auto user_users = users[0]->users();
    if (user_users.size() != c2) {
      return false;
    }
    if ((user_users[0]->pattern() == NodePattern::REDUCE && user_users[1]->pattern() == NodePattern::BROADCAST) ||
        (user_users[0]->pattern() == NodePattern::BROADCAST && user_users[1]->pattern() == NodePattern::REDUCE)) {
      (void)fused_areas_.emplace_back(users[0]);
      (void)fused_areas_.emplace_back(user_users[0]);
      (void)fused_areas_.emplace_back(user_users[1]);
    }
    return !fused_areas_.empty();
  }
};

void SplitModelAscend::InitFusePatterns() {
  AddPattern(std::make_shared<FuseVirtualNode>(), true);
  AddPattern(std::make_shared<ascend::FuseMatMul>(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateDepthMatcher(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateWidthMatcher(), true);
  AddPattern(FuseReduceFwd::CreateDepthMatcher(kReduceFusionDepth), true);
  AddPattern(FuseReduceFwd::CreateWidthMatcher(kReduceFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateDepthMatcher(kBroadcastFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateWidthMatcher(kBroadcastFusionDepth), true);
  AddPattern(std::make_shared<FuseLayerNorm>(), true);
}

AreaMode SplitModelAscend::GetDefaultAreaMode(const PrimOpPtr &node) const {
  if (node != nullptr && node->op() == "MatMul") {
    return AreaMode::COMPOSITE;
  }
  return AreaMode::BASIC;
}
}  // namespace mindspore::graphkernel::inner
