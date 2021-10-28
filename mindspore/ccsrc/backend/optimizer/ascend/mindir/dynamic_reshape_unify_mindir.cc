/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/mindir/dynamic_reshape_unify_mindir.h"

#include <vector>
#include <memory>

#include "utils/utils.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
size_t kDynamicReshapeInputNum = 2;

AnfNodePtr CreateDynamicReshape(const FuncGraphPtr &graph, const CNodePtr &reshape_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(reshape_node);
  const auto &reshape_node_inputs = reshape_node->inputs();
  CheckCNodeInputSize(reshape_node, kDynamicReshapeInputNum);
  std::vector<AnfNodePtr> dynamic_reshape_inputs = {NewValueNode(std::make_shared<Primitive>(kDynamicReshapeOpName)),
                                                    reshape_node_inputs[kDim1], reshape_node_inputs[kDim2]};
  auto dynamic_reshape_node = graph->NewCNode(dynamic_reshape_inputs);
  MS_EXCEPTION_IF_NULL(dynamic_reshape_node);
  dynamic_reshape_node->set_scope(reshape_node->scope());
  auto types = {AnfAlgo::GetOutputInferDataType(reshape_node, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(reshape_node, 0)};
  AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, dynamic_reshape_node.get());
  AnfAlgo::CopyNodeAttrs(reshape_node, dynamic_reshape_node);
  return dynamic_reshape_node;
}
}  // namespace

const BaseRef DynamicReshapeUnifyMindIR::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kReshapeOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr DynamicReshapeUnifyMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() < kDynamicReshapeInputNum + 1) {
    return nullptr;
  }
  auto shp_input = cnode->input(kDynamicReshapeInputNum);
  if (shp_input->isa<ValueNode>()) {
    return nullptr;
  }
  return CreateDynamicReshape(func_graph, cnode);
}
}  // namespace opt
}  // namespace mindspore
