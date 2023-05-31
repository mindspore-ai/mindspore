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

#include "backend/common/pass/batch_matmul_attr_fusion.h"

#include <memory>

#include "mindspore/core/ops/math_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef BatchMatMulAttrFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimBatchMatMul, Xs});
}

const AnfNodePtr BatchMatMulAttrFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  constexpr auto kTransposeA = "transpose_a";
  constexpr auto kTransposeB = "transpose_b";
  constexpr auto kTransposeX1 = "transpose_x1";
  constexpr auto kTransposeX2 = "transpose_x2";

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  bool transpose_a = common::AnfAlgo::GetNodeAttr<bool>(cnode, kTransposeA);
  bool transpose_b = common::AnfAlgo::GetNodeAttr<bool>(cnode, kTransposeB);

  common::AnfAlgo::SetNodeAttr(kTransposeX1, MakeValue(transpose_a), cnode);
  common::AnfAlgo::SetNodeAttr(kTransposeX2, MakeValue(transpose_b), cnode);

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
