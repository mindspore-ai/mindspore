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
#include "pre_activate/ascend/ir_fusion/softmax_grad_ext_fusion.h"
#include <memory>
#include "session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef SoftmaxGradExtFusion::DefinePattern() const {
  VectorRef mul({prim::kPrimMul, input1_, input0_});
  VectorRef sum({sum_var_, mul});
  VectorRef sub({prim::kPrimSub, input0_, sum});
  VectorRef mul1({prim::kPrimMul, input2_, input1_});
  VectorRef mul_grad({prim::kPrimMul, mul1, sub});
  return mul_grad;
}

const AnfNodePtr SoftmaxGradExtFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(node);
  auto input0 = GetAnfNodeByVar(equiv, input0_);
  auto input1 = GetAnfNodeByVar(equiv, input1_);
  auto input2 = GetAnfNodeByVar(equiv, input2_);
  auto sum = GetAnfNodeByVar(equiv, sum_var_);

  auto prim = std::make_shared<Primitive>(kSoftmaxGradExtOpName);
  auto fusion_node = graph->NewCNode({NewValueNode(prim), input0, input1, input2});
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_scope(node->scope());
  fusion_node->set_abstract(node->abstract());
  AnfAlgo::CopyNodeAttr(kAttrKeepDims, sum, fusion_node);
  AnfAlgo::CopyNodeAttr(kAttrAxis, sum, fusion_node);
  return fusion_node;
}
}  // namespace opt
}  // namespace mindspore
