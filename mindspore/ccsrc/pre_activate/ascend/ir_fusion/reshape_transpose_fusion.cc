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

#include "pre_activate/ascend/ir_fusion/reshape_transpose_fusion.h"
#include <memory>
#include "session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "pre_activate/common/helper.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
const BaseRef ReshapeTransposeFusion::DefinePattern() const {
  const auto prim_reshape = std::make_shared<Primitive>(prim::kPrimReshape->name());
  VectorRef reshape({prim_reshape, input_varptr_});

  return VectorRef({prim::kPrimTranspose, reshape});
}

const AnfNodePtr ReshapeTransposeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  auto transpose_cnode = CheckAnfNodeIfCNodeAndInputSize(node, kBackendReshapeInputNum);
  MS_EXCEPTION_IF_NULL(transpose_cnode);
  auto reshape_cnode = CheckAnfNodeIfCNodeAndInputSize(transpose_cnode->input(1), kBackendReshapeInputNum);
  MS_EXCEPTION_IF_NULL(reshape_cnode);
  auto prim = std::make_shared<Primitive>(kConfusionTransposeDOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), utils::cast<AnfNodePtr>((*equiv)[input_varptr_])};
  auto new_node = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_abstract(node->abstract());

  AnfAlgo::CopyNodeAttrs(reshape_cnode, new_node);
  AnfAlgo::CopyNodeAttr(kAttrPerm, transpose_cnode, new_node);
  AnfAlgo::SetNodeAttr(kAttrTransposeFirst, MakeValue(false), new_node);

  return new_node;
}
}  // namespace opt
}  // namespace mindspore
