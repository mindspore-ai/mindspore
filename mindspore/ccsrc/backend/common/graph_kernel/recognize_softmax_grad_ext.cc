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

#include "backend/common/graph_kernel/recognize_softmax_grad_ext.h"
#include <memory>
#include "ops/math_ops.h"

namespace mindspore::graphkernel {
namespace {
void SetNodeNotClusterable(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  cnode->AddAttr("keep_basic", MakeValue(true));
}
}  // namespace

const BaseRef RecognizeSoftmaxGradExt::DefinePattern() const {
  VarPtr x0 = std::make_shared<Var>();
  VarPtr x1 = std::make_shared<Var>();
  VarPtr axis = std::make_shared<Var>();
  VarPtr const_v = std::make_shared<Var>();
  VectorRef y0({mul1_, x0, x1});
  VectorRef y1({reduce_sum_, y0, axis});
  VectorRef y2({sub_, x1, y1});
  VectorRef y3({mul2_, x0, y2});
  VectorRef y4({prim::kPrimMul, const_v, y3});
  return y4;
}

const AnfNodePtr RecognizeSoftmaxGradExt::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto mul1 = opt::GetAnfNodeByVar(equiv, mul1_);
  auto mul2 = opt::GetAnfNodeByVar(equiv, mul2_);
  auto sub = opt::GetAnfNodeByVar(equiv, sub_);
  auto reduce_sum = opt::GetAnfNodeByVar(equiv, reduce_sum_);
  SetNodeNotClusterable(node);
  SetNodeNotClusterable(mul1);
  SetNodeNotClusterable(mul2);
  SetNodeNotClusterable(sub);
  SetNodeNotClusterable(reduce_sum);
  return node;
}
}  // namespace mindspore::graphkernel
