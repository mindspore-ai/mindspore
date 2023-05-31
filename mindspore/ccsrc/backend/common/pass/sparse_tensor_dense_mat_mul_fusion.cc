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

#include "backend/common/pass/sparse_tensor_dense_mat_mul_fusion.h"

#include <memory>

#include "mindspore/core/ops/sparse_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef SparseTensorDenseMatMulFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimSparseTensorDenseMatmul, Xs});
}

const AnfNodePtr SparseTensorDenseMatMulFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                        const EquivPtr &) const {
  constexpr auto kAdjointA = "adjoint_a";
  constexpr auto kAdjointB = "adjoint_b";
  constexpr auto kAdjointSt = "adjoint_st";
  constexpr auto kAdjointDt = "adjoint_dt";

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  bool adjoint_st = common::AnfAlgo::GetNodeAttr<bool>(cnode, kAdjointSt);
  bool adjoint_dt = common::AnfAlgo::GetNodeAttr<bool>(cnode, kAdjointDt);

  common::AnfAlgo::SetNodeAttr(kAdjointA, MakeValue(adjoint_st), cnode);
  common::AnfAlgo::SetNodeAttr(kAdjointB, MakeValue(adjoint_dt), cnode);

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
