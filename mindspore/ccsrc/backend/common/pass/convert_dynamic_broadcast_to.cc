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

#include <memory>
#include "backend/common/pass/convert_dynamic_broadcast_to.h"
#include "mindspore/core/ops/array_ops.h"
#include "ir/anf.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
const auto kA = "A";
const auto kVs = "Vs";
const auto kMBroadcastTo = "m_broadcast_to";
const auto kRBroadcastTo = "r_broadcast_to";
constexpr size_t kCNodePrimitiveIdx = 0;
AnfNodePtr BuildBroadcastTo(const PatternMap &m, const AnfNodePtr &) {
  auto node = m.Get(kMBroadcastTo);
  MS_EXCEPTION_IF_NULL(node);
  auto broadcast_to_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(broadcast_to_node);

  auto broadcast_to_op_name = prim::kPrimBroadcastTo->name();
  auto prim = GetValueNode<PrimitivePtr>(broadcast_to_node->input(kCNodePrimitiveIdx));
  MS_EXCEPTION_IF_NULL(prim);
  prim->Named::operator=(Named(broadcast_to_op_name));

  return node;
}
}  // namespace

bool ConvertDynamicBroadcastTo::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &,
                                                const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::IsDynamicShape(node)) {
    return true;
  }
  return false;
}

void ConvertDynamicBroadcastTo::DefineSrcPattern(SrcPattern *src_pattern) {
  (void)(*src_pattern).AddVar(kA).AddSeqVar(kVs).AddCNode(kMBroadcastTo, {prim::kPrimDynamicBroadcastTo, kA, kVs});
}

void ConvertDynamicBroadcastTo::DefineDstPattern(DstPattern *dst_pattern) {
  (void)(*dst_pattern).AddCNode(kRBroadcastTo, {prim::kPrimBroadcastTo, kA, kVs}, BuildBroadcastTo);
}
}  // namespace opt
}  // namespace mindspore
