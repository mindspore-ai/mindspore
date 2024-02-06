/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/shape_getitem_reshape_fusion.h"
#include "mindspore/core/ops/array_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace opt {
const BaseRef ShapeGetItemReshapeFusion::DefinePattern() const {
  //  a = Shape(X)
  //  b = RealGetItem(a, 0)
  //  c = RealGetItem(a, 1)
  //  d = RealMakeTuple(b, c, var)
  //  out = Reshape(Y, d)
  VectorRef shape = VectorRef({std::make_shared<Primitive>("Shape"), x_});
  VectorRef get_0 = VectorRef({prim::kPrimRealTupleGetItem, shape, index0_});
  VectorRef get_1 = VectorRef({prim::kPrimRealTupleGetItem, shape, index1_});
  VectorRef make_tuple = VectorRef({prim::kPrimRealMakeTuple, get_0, get_1, var_});
  VectorRef out = VectorRef({std::make_shared<Primitive>("Reshape"), y_, make_tuple});
  return out;
}

const AnfNodePtr ShapeGetItemReshapeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  auto real_maktuple = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 1);
  auto var = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(real_maktuple), 2);
  int64_t value = AnfUtils::GetIntValue(utils::cast<ValueNodePtr>(var)->value());
  auto prim = std::make_shared<Primitive>("ReshapeExt");
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), utils::cast<AnfNodePtr>((*equiv)[x_]),
                                    utils::cast<AnfNodePtr>((*equiv)[y_]), utils::cast<AnfNodePtr>((*equiv)[var_])};
  auto new_node = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->AddAttr("var", utils::cast<ValueNodePtr>(var)->value());
  new_node->set_abstract(node->abstract());

  return new_node;
}
}  // namespace opt
}  // namespace mindspore
