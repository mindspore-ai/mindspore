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

#include "plugin/device/ascend/optimizer/mindir/maxpool_with_argmax_unify_mindir.h"
#include <memory>
#include <vector>
#include "backend/common/optimizer/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMaxPoolGradWithArgmaxInputTensorNum = 3;
constexpr size_t kMaxPoolGradWithArgmaxInputNum = 4;
constexpr size_t kMaxPoolWithArgmaxShape = 4;
constexpr size_t kAlignBytes = 16;
constexpr auto kX1 = "X1";
constexpr auto kX2 = "X2";
constexpr auto kMaxPoolIndex = "index0";
constexpr auto kMMaxPool = "m_max_pool";
constexpr auto kRMaxPool = "r_max_pool";
constexpr auto kMMaxpoolWithArgmax = "m_maxpool_with_argmax";
constexpr auto kMTupleGetitem0 = "m_tuple_getitem0";
constexpr auto kMMaxpoolGradWithArgmax = "m_maxpool_grad_with_argmax";
constexpr auto kRMaxpoolGradWithArgmax = "r_maxpool_grad_with_argmax";

bool IsC(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    return in->isa<ValueNode>();
  }
  return false;
}

CNodePtr GetMaxPoolWithArgmax(const CNodePtr &maxpool_grad_with_argmax) {
  CheckCNodeInputSize(maxpool_grad_with_argmax, kMaxPoolGradWithArgmaxInputTensorNum);
  auto tuple_getitem0_anf = maxpool_grad_with_argmax->input(kIndex3);
  MS_EXCEPTION_IF_NULL(tuple_getitem0_anf);
  return tuple_getitem0_anf->cast<CNodePtr>();
}

AnfNodePtr BuildMaxPoolWithArgmax(const PatternMap &m, const AnfNodePtr &) {
  auto node = m.Get(kMMaxPool);
  MS_EXCEPTION_IF_NULL(node);
  auto maxpool_with_argmax = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(maxpool_with_argmax);
  if (common::AnfAlgo::IsDynamicShape(maxpool_with_argmax)) {
    MS_LOG(EXCEPTION) << "MaxPoolWithArgmax don't support dynamic shape, node: "
                      << maxpool_with_argmax->fullname_with_scope();
  }

  TypeId argmax_dtype = kNumberTypeUInt16;
  auto ksize = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(maxpool_with_argmax, kAttrKernelSize);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(maxpool_with_argmax, 0UL);
  auto argmax_shape = output_shape;
  if (argmax_shape.size() != kMaxPoolWithArgmaxShape || ksize.size() != kMaxPoolWithArgmaxShape) {
    MS_LOG(EXCEPTION) << "Argmax or kernel_size's shape dim should be equal to 4, but got argmax dim: "
                      << argmax_shape.size() << ", kernel_size dim: " << ksize.size() << trace::DumpSourceLines(node);
  }
  argmax_shape[kDim2] = ksize[kDim1] * ksize[kDim2];
  argmax_shape[kDim3] =
    (output_shape[kDim2] * output_shape[kDim3] + SizeToLong(kAlignBytes) - 1) / SizeToLong(kAlignBytes) + 1;
  auto types = {common::AnfAlgo::GetOutputInferDataType(maxpool_with_argmax, 0), argmax_dtype};
  auto shapes = {output_shape, argmax_shape};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, maxpool_with_argmax.get());

  return maxpool_with_argmax;
}

AnfNodePtr BuildMaxPoolGradWithArgmax(const PatternMap &m, const AnfNodePtr &) {
  auto node = m.Get(kMMaxpoolGradWithArgmax);
  MS_EXCEPTION_IF_NULL(node);
  auto maxpool_grad_with_argmax = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(maxpool_grad_with_argmax);
  if (common::AnfAlgo::IsDynamicShape(maxpool_grad_with_argmax)) {
    MS_LOG(EXCEPTION) << "MaxPoolGradWithArgmax don't support dynamic shape, node: "
                      << maxpool_grad_with_argmax->fullname_with_scope();
  }
  auto tuple_getitem0_anf = GetMaxPoolWithArgmax(maxpool_grad_with_argmax);
  MS_EXCEPTION_IF_NULL(tuple_getitem0_anf);

  TypeId argmax_dtype = kNumberTypeUInt16;
  auto ksize = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(maxpool_grad_with_argmax, kAttrKernelSize);
  auto argmax_shape = common::AnfAlgo::GetOutputInferShape(tuple_getitem0_anf, 0UL);
  if (argmax_shape.size() != kMaxPoolWithArgmaxShape || ksize.size() != kMaxPoolWithArgmaxShape) {
    MS_LOG(EXCEPTION) << "Argmax or kernel_size's shape dim should be equal to 4, but got argmax dim: "
                      << argmax_shape.size() << ", kernel_size dim: " << ksize.size() << trace::DumpSourceLines(node);
  }
  argmax_shape[kDim3] =
    (argmax_shape[kDim2] * argmax_shape[kDim3] + SizeToLong(kAlignBytes) - 1) / SizeToLong(kAlignBytes) + 1;
  argmax_shape[kDim2] = ksize[kDim1] * ksize[kDim2];
  common::AnfAlgo::SetOutputInferTypeAndShape({argmax_dtype}, {argmax_shape}, tuple_getitem0_anf.get());

  return maxpool_grad_with_argmax;
}
}  // namespace

void MaxPoolWithArgmaxUnifyMindIR::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddVar(kX1).AddCNode(kMMaxPool, {prim::kPrimMaxPoolWithArgmax, kX1});
}

void MaxPoolWithArgmaxUnifyMindIR::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(kRMaxPool, {prim::kPrimMaxPoolWithArgmax, kX1}, BuildMaxPoolWithArgmax);
}

void MaxPoolGradWithArgmaxUnifyMindIR::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern)
    .AddVar(kX1)
    .AddVar(kX2)
    .AddVar(kMaxPoolIndex, IsC)
    .AddCNode(kMMaxpoolWithArgmax, {prim::kPrimMaxPoolWithArgmax, kX1})
    .AddCNode(kMTupleGetitem0, {prim::kPrimTupleGetItem, kMMaxpoolWithArgmax, kMaxPoolIndex})
    .AddCNode(kMMaxpoolGradWithArgmax, {prim::kPrimMaxPoolGradWithArgmax, kX1, kX2, kMTupleGetitem0});
}
void MaxPoolGradWithArgmaxUnifyMindIR::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern)
    .AddCNode(kRMaxpoolGradWithArgmax, {prim::kPrimMaxPoolGradWithArgmax, kX1, kX2, kMTupleGetitem0},
              BuildMaxPoolGradWithArgmax);
}
}  // namespace opt
}  // namespace mindspore
