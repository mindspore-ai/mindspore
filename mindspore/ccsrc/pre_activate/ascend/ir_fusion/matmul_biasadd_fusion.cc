/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "pre_activate/ascend/ir_fusion/matmul_biasadd_fusion.h"
#include <memory>
#include "pre_activate/common/helper.h"
#include "session/anf_runtime_algorithm.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMatMulInputIndex = 1;
constexpr size_t kBiasInputIndex = 2;
}  // namespace

const BaseRef MatmulBiasaddFusion::DefinePattern() const {
  VarPtr X0 = std::make_shared<Var>();
  VarPtr X1 = std::make_shared<Var>();
  VarPtr X2 = std::make_shared<Var>();
  const auto prim_bias_add = std::make_shared<Primitive>(kBiasAddOpName);
  return VectorRef({prim_bias_add, VectorRef({prim::kPrimMatMul, X0, X1}), X2});
}

const AnfNodePtr MatmulBiasaddFusion::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  CheckCNodeInputSize(cnode, kBiasAddInputNum);
  AnfNodePtr matmul = cnode->input(kMatMulInputIndex);
  MS_EXCEPTION_IF_NULL(matmul);
  auto matmul_cnode = matmul->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(matmul_cnode);
  matmul_cnode->add_input(cnode->input(kBiasInputIndex));
  AnfAlgo::SetNodeAttr(kAttrHasBias, MakeValue(true), matmul);
  return matmul;
}
}  // namespace opt
}  // namespace mindspore
