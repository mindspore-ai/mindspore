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
#include "src/gllo/fusion/conv_biasadd_fusion.h"
#include <memory>
#include "schema/inner/model_generated.h"
#include "src/ir/primitive_t_value.h"
#include "mindspore/ccsrc/utils/utils.h"
#include "src/gllo/common/utils.h"

namespace mindspore {
namespace opt {

const BaseRef ConvBiasaddFusion::DefinePattern() const {
  MS_LOG(DEBUG) << "Enter pattern";

  VarPtr X = std::make_shared<Var>();
  VarPtr W = std::make_shared<Var>();
  VarPtr B = std::make_shared<Var>();
  CheckIfVarIsNull(X);
  CheckIfVarIsNull(W);
  CheckIfVarIsNull(B);

  auto prim1 = new schema::PrimitiveT();
  prim1->value.type = schema::PrimitiveType_BiasAdd;
  auto prim11 = std::make_shared<lite::PrimitiveTValue>(prim1);

  auto prim2 = new schema::PrimitiveT();
  prim2->value.type = schema::PrimitiveType_Conv2D;
  auto prim22 = std::make_shared<lite::PrimitiveTValue>(prim2);

  return VectorRef({prim11, VectorRef({prim22, X, W}), B});
}

const AnfNodePtr ConvBiasaddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  MS_LOG(DEBUG) << "Enter pass process";
  CheckIfFuncGraphIsNull(func_graph);

  CheckIfAnfNodeIsNull(node);
  auto cnode = node->cast<CNodePtr>();
  CheckIfCNodeIsNull(cnode);
  CheckInputSize(cnode, 3);  // [op, conv_node, bias_node]

  AnfNodePtr conv_node_anf = cnode->input(1);
  CheckIfAnfNodeIsNull(conv_node_anf);
  auto conv_node = conv_node_anf->cast<CNodePtr>();
  CheckIfCNodeIsNull(conv_node);
  CheckInputSize(conv_node, 3);  // [op, X, W]

  conv_node->add_input(cnode->input(2));

  auto primitive = (lite::PrimitiveTValue *)(conv_node->input(0)->cast<ValueNodePtr>()->value().get());
  primitive->GetPrimitiveT()->value.AsConv2D()->hasBias = true;

  return conv_node_anf;
}

}  // namespace opt
}  // namespace mindspore

