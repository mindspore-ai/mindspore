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
#include "backend/optimizer/ascend/ir_fission/tensor_scatter_update_fission.h"
#include <vector>
#include <memory>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
CNodePtr CreateTensorMove(const FuncGraphPtr &graph, const CNodePtr &tensor_scatter_update) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(tensor_scatter_update);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(kTensorMoveOpName)),
                                    tensor_scatter_update->input(1)};
  auto tensor_move = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tensor_move);
  tensor_move->set_scope(tensor_scatter_update->scope());
  tensor_move->set_abstract(tensor_scatter_update->abstract());
  AnfAlgo::SetNodeAttr(kAttrUseLocking, MakeValue(false), tensor_move);
  return tensor_move;
}

CNodePtr CreateScatterNdUpdate(const FuncGraphPtr &graph, const CNodePtr &tensor_scatter_update,
                               const CNodePtr &tensor_move) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(tensor_scatter_update);
  MS_EXCEPTION_IF_NULL(tensor_move);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(kScatterNdUpdateOpName)), tensor_move,
                                    tensor_scatter_update->input(2), tensor_scatter_update->input(3)};
  auto scatter_nd_update = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(scatter_nd_update);
  scatter_nd_update->set_scope(tensor_scatter_update->scope());
  scatter_nd_update->set_abstract(tensor_scatter_update->abstract());
  return scatter_nd_update;
}
}  // namespace

const BaseRef TensorScatterUpdateFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kTensorScatterUpdateOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr TensorScatterUpdateFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto tensor_scatter_update = node->cast<CNodePtr>();
  constexpr size_t INPUT_NUM = 4;
  if (tensor_scatter_update == nullptr || tensor_scatter_update->size() != INPUT_NUM) {
    return nullptr;
  }
  auto tensor_move = CreateTensorMove(func_graph, tensor_scatter_update);
  return CreateScatterNdUpdate(func_graph, tensor_scatter_update, tensor_move);
}
}  // namespace opt
}  // namespace mindspore
