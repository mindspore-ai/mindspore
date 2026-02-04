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
#include "backend/ms_backend/graph_fusion/adapter/split_model_ascend.h"
#include <memory>
#include <string>
#include <algorithm>
#include <functional>
#include "primitive/array_op_name.h"
#include "primitive/math_op_name.h"
#include "primitive/other_op_name.h"  // collective communication ops
#include "primitive/sequence_ops.h"
#include "primitive/nn_optimizer_op_name.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"

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

class FuseSlice : public FusePattern {
 public:
  FuseSlice() : FusePattern("slice") { direction_ = FuseDirection::BACKWARD; }
  ~FuseSlice() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return dom->dom()->op() == "Slice" || dom->dom()->op() == "StridedSlice"; }
  bool Match(const AreaPtr &dom) override {
    for (auto &[a, r] : dom->users_with_relation()) {
      if (a->pattern() < NodePattern::BROADCAST && r == EdgeRelation::INJECTIVE && !HasCircle(dom, a)) {
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

class FuseElemAny : public FusePattern {
 public:
  FuseElemAny() : FusePattern("elemany_addn") {}
  ~FuseElemAny() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return dom->dom()->op() == "ElemAny"; }
  bool Match(const AreaPtr &dom) override {
    for (auto &[a, r] : dom->inputs_with_relation()) {
      if (a->pattern() <= NodePattern::BROADCAST && r == EdgeRelation::INJECTIVE && !HasCircle(dom, a)) {
        (void)fused_areas_.emplace_back(a);
      }
    }
    return !fused_areas_.empty();
  }
};
}  // namespace ascend

namespace dvm {
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
      if (a->pattern() <= NodePattern::BROADCAST) {
        if (r != EdgeRelation::INJECTIVE && (a->user_num() != 1 || a->is_output())) {
          continue;
        }
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

class FuseMatMul : public FusePattern {
 public:
  FuseMatMul() : FusePattern("matmul_depth") { direction_ = FuseDirection::BACKWARD; }
  ~FuseMatMul() = default;

 protected:
  bool Check(const AreaPtr &dom) override {
    const int64_t kGroupTypeK = 2;
    const size_t kGmmSize = 2;
    if (dom->size() == 1) {
      return dom->dom()->op() == kMatMulOpName || dom->dom()->op() == kBatchMatMulOpName;
    } else if (dom->size() == kGmmSize) {
      if (dom->ops()[0]->op() == ops::kNameGroupedMatmul && dom->ops()[1]->op() == kTupleGetItemOpName) {
        auto group_type_value = dom->dom()->attrs().at("group_type");
        MS_EXCEPTION_IF_NULL(group_type_value);
        auto group_type = GetValue<int64_t>(group_type_value);
        return group_type == kGroupTypeK;
      }
    }
    return false;
  }

  bool IsSameShapeSize(int64_t size, const NodePtrList &output_nodes) {
    for (auto &node : output_nodes) {
      if (std::accumulate(node->shape.begin(), node->shape.end(), static_cast<int64_t>(1),
                          std::multiplies<int64_t>()) != size) {
        return false;
      }
    }
    return true;
  }

  bool Match(const AreaPtr &dom) override {
    constexpr size_t MAX_FUSE_NUM = 5;
    size_t current_size = 0;
    auto output_shape = dom->ops().back()->shape;
    int64_t matmul_output_size =
      std::accumulate(output_shape.begin(), output_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    if (output_shape.back() == 1) {
      return false;
    }
    auto dom_users = dom->users();
    std::sort(dom_users.begin(), dom_users.end(),
              [](const AreaPtr &a, const AreaPtr &b) { return a->area_outputs().size() < b->area_outputs().size(); });
    for (auto &a : dom_users) {
      if (current_size + a->area_outputs().size() > MAX_FUSE_NUM) {
        break;
      }
      if (a->size() == 1 && a->dom()->op() == kReshapeOpName) {
        continue;
      }

      bool fuse_flag = (dom->dom()->op() == kMatMulOpName || dom->dom()->op() == kBatchMatMulOpName ||
                        dom->dom()->op() == ops::kNameGroupedMatmul);
      if (std::any_of(a->ops().begin(), a->ops().end(),
                      [](const PrimOpPtr &op) { return op->op() == kReshapeOpName; })) {
        fuse_flag = fuse_flag && (a->pattern() < NodePattern::BROADCAST);
      } else {
        fuse_flag = fuse_flag && (a->pattern() <= NodePattern::BROADCAST);
      }
      if (fuse_flag && !HasCircle(dom, a) && IsSameShapeSize(matmul_output_size, a->area_outputs())) {
        (void)fused_areas_.emplace_back(a);
        current_size += a->area_outputs().size();
      }
    }
    return !fused_areas_.empty();
  }
};

class FuseAllReduceFwd : public FusePattern {
 public:
  FuseAllReduceFwd() : FusePattern("allreduce_fwd") { direction_ = FuseDirection::FORWARD; }
  ~FuseAllReduceFwd() = default;

  bool Check(const AreaPtr &dom) override { return dom->size() == 1 && (dom->dom()->op() == kAllReduceOpName); }

  bool Match(const AreaPtr &dom) override {
    for (auto &[a, r] : dom->inputs_with_relation()) {
      if (a->user_num() != 1) {
        continue;
      }
      if (!HasCircle(a, dom) && r == EdgeRelation::INJECTIVE && a->size() == 1 && a->dom()->op() == kMatMulOpName) {
        (void)fused_areas_.emplace_back(a);
      }
    }
    return !fused_areas_.empty();
  }
};

class FuseAllReduceBwd : public FusePattern {
 public:
  FuseAllReduceBwd() : FusePattern("allreduce_bwd") { direction_ = FuseDirection::BACKWARD; }
  ~FuseAllReduceBwd() = default;

  bool Check(const AreaPtr &dom) override {
    auto ops = dom->ops();
    return std::any_of(ops.begin(), ops.end(), [](const PrimOpPtr op) { return op->op() == kAllReduceOpName; });
  }

  bool Match(const AreaPtr &dom) override {
    const auto &users = dom->users();
    for (auto &a : users) {
      if (a->pattern() < NodePattern::BROADCAST && !HasCircle(dom, a)) {
        (void)fused_areas_.emplace_back(a);
      }
    }
    return !fused_areas_.empty();
  }
};

// bind the virtual nodes to their inputs
class FuseVirtualNode : public FusePattern {
 public:
  FuseVirtualNode() : FusePattern("fuse_virtual_node") { direction_ = FuseDirection::FORWARD; }
  ~FuseVirtualNode() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return dom->pattern() == NodePattern::VIRTUAL; }
  bool Match(const AreaPtr &dom) override {
    for (auto &inp : dom->inputs()) {
      if (inp != nullptr && inp->dom()->op() != kTransposeOpName) {
        (void)fused_areas_.emplace_back(inp);
      }
    }
    return !fused_areas_.empty();
  }
};
}  // namespace dvm

void SplitModelAscend::InitFusePatterns() {
  is_dvm_ = (GraphKernelFlags::GetInstance().kernel_generator == "DVM");
  if (is_dvm_) {
    // fuse pattern for dvm
    AddPattern(std::make_shared<inner::dvm::FuseVirtualNode>(), true);
    AddPattern(std::make_shared<FuseReshape>(), true);
    AddPattern(FuseElemwiseBroadcastFwd::CreateDepthMatcher(), true);
    AddPattern(FuseElemwiseBroadcastFwd::CreateWidthMatcher(), true);
    AddPattern(inner::dvm::FuseReduceFwd::CreateDepthMatcher(inner::ascend::kReduceFusionDepth), true);
    AddPattern(inner::dvm::FuseReduceFwd::CreateWidthMatcher(inner::ascend::kReduceFusionDepth), true);
    AddPattern(FuseElemwiseBroadcastBwd::CreateDepthMatcher(inner::ascend::kBroadcastFusionDepth), true);
    AddPattern(FuseElemwiseBroadcastBwd::CreateWidthMatcher(inner::ascend::kBroadcastFusionDepth), true);
    AddPattern(std::make_shared<inner::ascend::FuseElemAny>(), true);
    AddPattern(std::make_shared<inner::ascend::FuseSlice>(), true);
    if (!graphkernel::GraphKernelFlags::GetInstance().disable_matmul_post_fusion) {
      AddPattern(std::make_shared<inner::dvm::FuseMatMul>(), true);
    }
    if (graphkernel::GraphKernelFlags::GetInstance().enable_allreduce_prologue_fusion) {
      AddPattern(std::make_shared<inner::dvm::FuseAllReduceFwd>(), true);
    }
    if (graphkernel::GraphKernelFlags::GetInstance().enable_allreduce_epilogue_fusion) {
      AddPattern(std::make_shared<inner::dvm::FuseAllReduceBwd>(), true);
    }
  } else {
    // fuse pattern for akg
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
}

AreaMode SplitModelAscend::GetDefaultAreaMode(const PrimOpPtr &node) const {
  if (node != nullptr) {
    auto node_name = node->op();
    if (node_name == kReshapeOpName || node_name == kAssignOpName) {
      return AreaMode::BASIC;
    }
    if (is_dvm_ && (node_name == kTransposeOpName || node_name == kCastOpName)) {
      return AreaMode::BASIC;
    }
  }
  return AreaMode::COMPOSITE;
}
}  // namespace mindspore::graphkernel::inner
