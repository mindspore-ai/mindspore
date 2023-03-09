/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/conv_transpose_to_conv_bp.h"
#include <memory>
#include <vector>
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "utils/trace_base.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCNodePrimitiveIdx = 0;
constexpr auto kXs = "Xs";
constexpr auto kMConv2dTrans = "m_conv2d_trans";
constexpr auto kRConv2dBp = "r_conv2d_bp";

AnfNodePtr BuildConv2DBackpropInput(const PatternMap &m, const AnfNodePtr &default_node) {
  auto node = m.Get(kMConv2dTrans);
  auto conv_transpose = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(conv_transpose);

  if (conv_transpose->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Cnode inputs should not be empty, cnode: " << node->DebugString()
                      << trace::DumpSourceLines(conv_transpose);
  }

  auto prim = GetValueNode<PrimitivePtr>(conv_transpose->input(kCNodePrimitiveIdx));
  MS_EXCEPTION_IF_NULL(prim);
  prim->Named::operator=(Named(kConv2DBackpropInputOpName));

  return node;
}
}  // namespace

bool ConvTransposeToConvBackpropInputPass::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &,
                                                           const AnfNodePtr &) const {
  return true;
}

void ConvTransposeToConvBackpropInputPass::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddSeqVar(kXs).AddCNode(kMConv2dTrans, {prim::kPrimConv2DTranspose, kXs});
}

void ConvTransposeToConvBackpropInputPass::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(kRConv2dBp, {prim::kPrimConv2DBackpropInput, kXs}, BuildConv2DBackpropInput);
}
}  // namespace opt
}  // namespace mindspore
