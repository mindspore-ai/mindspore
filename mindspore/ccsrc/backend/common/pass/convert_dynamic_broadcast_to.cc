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
AnfNodePtr BuildBroadcastTo(const PatternMap &m, const AnfNodePtr &) {
  auto node = m.Get(kMBroadcastTo);
  MS_EXCEPTION_IF_NULL(node);
  auto broadcast_to_op_name = prim::kPrimBroadcastTo->name();
  auto ori_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(ori_cnode);
  auto input_x = common::AnfAlgo::GetInputNode(ori_cnode, 0);
  auto func_graph = node->func_graph();
  CNodePtr broadcast_to_node =
    opt::NewCNode({NewValueNode(std::make_shared<Primitive>(broadcast_to_op_name)), input_x}, func_graph, {node});
  MS_EXCEPTION_IF_NULL(broadcast_to_node);
  MS_EXCEPTION_IF_NULL(node->abstract());
  MS_EXCEPTION_IF_NULL(node->abstract()->BuildShape());
  broadcast_to_node->set_abstract(node->abstract());
  auto shape_ptr = node->abstract()->BuildShape()->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(shape_ptr->shape()), broadcast_to_node);
  return broadcast_to_node;
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
  (*src_pattern).AddVar(kA).AddSeqVar(kVs).AddCNode(kMBroadcastTo, {prim::kPrimDynamicBroadcastTo, kA, kVs});
}

void ConvertDynamicBroadcastTo::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(kRBroadcastTo, {prim::kPrimBroadcastTo, kA}, BuildBroadcastTo);
}
}  // namespace opt
}  // namespace mindspore
