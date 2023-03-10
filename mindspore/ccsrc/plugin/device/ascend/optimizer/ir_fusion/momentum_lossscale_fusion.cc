/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/momentum_lossscale_fusion.h"
#include <memory>
#include <vector>
#include <string>
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAccumIndex = 1;
bool CheckValueNodeInputOfMul(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    return false;
  }
  if (common::AnfAlgo::IsDynamicShape(node)) {
    return false;
  }
  auto mul_input_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  return mul_input_shape.empty() || (mul_input_shape.size() == 1 && mul_input_shape[0] == 1);
}
}  // namespace

const BaseRef MomentumLossscaleFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr X0 = std::make_shared<Var>();
  VarPtr X1 = std::make_shared<Var>();
  VarPtr X2 = std::make_shared<Var>();
  VarPtr X4 = std::make_shared<Var>();
  // UpdateState node
  VarPtr X5 = std::make_shared<Var>();
  return VectorRef({prim::kPrimApplyMomentumD, X0, X1, X2, VectorRef({prim::kPrimMul, Xs}), X4, X5});
}

const AnfNodePtr MomentumLossscaleFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  CheckCNodeInputSize(cnode, kApplyMomentumInputTensorNum);
  AnfNodePtr mul = cnode->input(kIndex4);
  MS_EXCEPTION_IF_NULL(mul);
  auto mul_cnode = mul->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_cnode);
  CheckCNodeInputSize(mul_cnode, kMulInputTensorNum);
  size_t value_node_index = 0;
  // All real inputs include 1prim + x*TensorInput
  for (size_t i = 1; i < kMulInputTensorNum + 1; ++i) {
    if (CheckValueNodeInputOfMul(mul_cnode->input(i))) {
      value_node_index = i;
      break;
    }
  }
  if (value_node_index == 0) {
    MS_LOG(DEBUG) << "The Mul " << mul->DebugString() << " to be fused must has a scalar constant input";
    return nullptr;
  }
  auto new_prim = std::make_shared<Primitive>(kFusedMulApplyMomentumOpName);
  auto depend_prim = NewValueNode(prim::kPrimDepend);
  auto depend = func_graph->NewCNode({depend_prim, cnode->input(kIndex5), cnode->input(kIndex6)});  // depend on monad
  depend->set_abstract(cnode->input(kIndex5)->abstract());
  depend->set_scope(cnode->input(kIndex5)->scope());
  std::vector<AnfNodePtr> new_node_inputs{NewValueNode(new_prim),
                                          cnode->input(kIndex1),
                                          cnode->input(kIndex2),
                                          cnode->input(kIndex3),
                                          mul_cnode->input(kMulInputTensorNum + 1 - value_node_index),
                                          depend,
                                          mul_cnode->input(value_node_index)};
  auto new_node = NewCNode(new_node_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  common::AnfAlgo::CopyNodeAttrs(node, new_node);
  auto input_names_value = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(new_node, kAttrInputNames);
  input_names_value[kIndex3] = "x1";
  (void)input_names_value.emplace_back("x2");
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names_value), new_node);
  new_node->set_abstract(node->abstract());
  new_node->set_scope(node->scope());
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
