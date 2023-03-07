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
#include "plugin/device/ascend/optimizer/ir_fusion/softmax_grad_ext_fusion.h"
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

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

const BaseRef SoftmaxGradExtFusionV2::DefinePattern() const {
  VectorRef mul({prim::kPrimMul, input1_, input0_});
  VectorRef sum({sum_var_, mul});
  VectorRef sub({prim::kPrimSub, input0_, sum});
  VectorRef mul1({prim::kPrimMul, input1_, sub});
  VectorRef mul_grad({prim::kPrimMul, input2_, mul1});
  return mul_grad;
}

const BaseRef SoftmaxGradExtFusionV3::DefinePattern() const {
  VectorRef mul({prim::kPrimMul, input1_, input0_});
  VectorRef sum({sum_var_, mul});
  VectorRef sub({prim::kPrimSub, input0_, sum});
  VectorRef mul1({prim::kPrimMul, input1_, sub});
  VectorRef mul_grad({prim::kPrimMul, mul1, input2_});
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
  if (!GetBoolAttr(sum, kAttrKeepDims)) {
    MS_LOG(INFO) << "sum's attr keep_dims should be true if do fusion";
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kSoftmaxGradExtOpName);
  auto fusion_node = NewCNode({NewValueNode(prim), input0, input1, input2}, graph);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_scope(node->scope());
  fusion_node->set_abstract(node->abstract());
  common::AnfAlgo::CopyNodeAttr(kAttrKeepDims, "keepdims", sum, fusion_node);
  common::AnfAlgo::CopyNodeAttr(kAttrAxis, sum, fusion_node);
  return fusion_node;
}
}  // namespace opt
}  // namespace mindspore
