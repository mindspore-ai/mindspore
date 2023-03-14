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
#include "backend/common/graph_kernel/adapter/split_model_ascend.h"
#include <memory>
#include <string>
#include "utils/ms_context.h"

namespace mindspore::graphkernel::inner {
namespace ascend {
constexpr size_t kReduceFusionDepth = 10;
constexpr size_t kBroadcastFusionDepth = 6;

class FuseReduceBwd : public FusePattern {
 public:
  FuseReduceBwd() : FusePattern("reduce_bwd") { direction_ = FuseDirection::BACKWARD; }
  ~FuseReduceBwd() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return dom->IsAlive() && dom->pattern() == NodePattern::REDUCE; }
  bool Match(const AreaPtr &dom) override {
    auto op_attrs = dom->dom()->attrs();
    if (op_attrs.find("reduce_output_fuse") == op_attrs.end()) {
      return false;
    }
    for (auto &[a, r] : dom->users_with_relation()) {
      if (a->pattern() <= NodePattern::BROADCAST && r == EdgeRelation::INJECTIVE && !HasCircle(dom, a)) {
        (void)fused_areas_.emplace_back(a);
      }
    }
    return !fused_areas_.empty();
  }
};

class FuseTransdata : public FusePattern {
 public:
  FuseTransdata() : FusePattern("transdata") {}
  ~FuseTransdata() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return dom->IsAlive() && dom->dom()->op() == kTransDataOpName; }
  bool Match(const AreaPtr &dom) override {
    for (auto &a : dom->inputs()) {
      if (a->IsAlive() && Supported(dom, a) && !HasCircle(a, dom)) {
        (void)fused_areas_.emplace_back(a);
      }
    }
    return !fused_areas_.empty();
  }

 private:
  bool NeedPad(const DShape &in_shape, const DShape &out_shape) const {
    const size_t min_rank = 2;
    const int64_t block_sz = 16;
    return !(in_shape.size() >= min_rank && out_shape.size() >= min_rank &&
             in_shape[in_shape.size() - kIndex1] == block_sz && in_shape[in_shape.size() - kIndex2] == block_sz &&
             out_shape[out_shape.size() - kIndex1] == block_sz && out_shape[out_shape.size() - kIndex2] == block_sz);
  }
  bool Supported(const AreaPtr &dom, const AreaPtr &a) const {
    if (dom->size() != 1 || dom->dom()->inputs().empty() || NeedPad(dom->dom()->input(0)->shape, dom->dom()->shape)) {
      return false;
    }
    if (a->dom()->op() == kMatMulOpName) {
      return true;
    }
    if (a->pattern() > NodePattern::BROADCAST) {
      return false;
    }
    auto op_attrs = dom->dom()->attrs();
    if (op_attrs.find(kAttrSrcFormat) == op_attrs.end() || op_attrs.find(kAttrDstFormat) == op_attrs.end()) {
      MS_LOG(ERROR) << "For '" << dom->dom()->op() << "', can not find the attr '" << kAttrSrcFormat << "' or '"
                    << kAttrDstFormat << "'";
      return false;
    }
    auto src_format = GetValue<std::string>(op_attrs[kAttrSrcFormat]);
    auto dst_format = GetValue<std::string>(op_attrs[kAttrDstFormat]);
    if (src_format == kOpFormat_FRAC_NZ && (dst_format == kOpFormat_DEFAULT || dst_format == kOpFormat_NCHW)) {
      return true;
    }
    return (src_format == kOpFormat_DEFAULT || src_format == kOpFormat_NCHW) && dst_format == kOpFormat_FRAC_NZ &&
           a->size() == 1 && a->dom()->op() == kCastOpName && !a->is_output();
  }
};
}  // namespace ascend

void SplitModelAscend::InitFusePatterns() {
  AddPattern(std::make_shared<FuseVirtualNode>(), true);
  AddPattern(std::make_shared<FuseReshape>(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateDepthMatcher(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateWidthMatcher(), true);
  AddPattern(FuseReduceFwd::CreateDepthMatcher(inner::ascend::kReduceFusionDepth), true);
  AddPattern(FuseReduceFwd::CreateWidthMatcher(inner::ascend::kReduceFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateDepthMatcher(inner::ascend::kBroadcastFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateWidthMatcher(inner::ascend::kBroadcastFusionDepth), true);
  AddPattern(std::make_shared<inner::ascend::FuseMatMul>(), true);
  AddPattern(std::make_shared<inner::ascend::FuseReduceBwd>(), true);
  AddPattern(std::make_shared<inner::ascend::FuseTransdata>(), true);
}

AreaMode SplitModelAscend::GetDefaultAreaMode(const PrimOpPtr &node) const {
  if (node != nullptr && node->op() == kReshapeOpName) {
    return AreaMode::BASIC;
  }
  return AreaMode::COMPOSITE;
}

SPLIT_MODEL_REGISTER(kAscendDevice, SplitModelAscend);
}  // namespace mindspore::graphkernel::inner
