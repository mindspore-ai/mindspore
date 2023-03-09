/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/tensor_scatter_fission.h"

#include <vector>
#include <memory>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "ops/core_ops.h"
#include "include/backend/optimizer/helper.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kTensorScatterInputSize = 3;
}  // namespace

const AnfNodePtr TensorScatterFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = CheckAnfNodeIfCNodeAndInputSize(node, kTensorScatterInputSize);
  // create TensorMove
  auto tensor_move_inputs =
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kTensorMoveOpName)), cnode->input(kIndex1)};
  auto tensor_move = NewCNode(tensor_move_inputs, graph);
  tensor_move->set_scope(node->scope());
  MS_EXCEPTION_IF_NULL(cnode->input(kIndex1));
  tensor_move->set_abstract(cnode->input(kIndex1)->abstract());
  // create ScatterNd node
  auto scatter_nd_inputs =
    std::vector<AnfNodePtr>{GetScatterNdPrimNode(), tensor_move, cnode->input(kIndex2), cnode->input(kIndex3)};
  auto scatter_nd_node = NewCNode(scatter_nd_inputs, graph);
  scatter_nd_node->set_scope(node->scope());
  scatter_nd_node->set_abstract(node->abstract());
  common::AnfAlgo::SetNodeAttr(kAttrUseLocking, MakeValue(false), scatter_nd_node);
  if (common::AnfAlgo::HasNodeAttr(kAttrCustAicpu, cnode)) {
    common::AnfAlgo::CopyNodeAttr(kAttrCustAicpu, cnode, scatter_nd_node);
  }
  return scatter_nd_node;
}

ValueNodePtr TensorScatterUpdateFission::GetScatterNdPrimNode() const {
  return NewValueNode(std::make_shared<Primitive>(prim::kPrimScatterNdUpdate->name()));
}

const BaseRef TensorScatterUpdateFission::DefinePattern() const {
  VarPtr input = std::make_shared<Var>();
  VarPtr indices = std::make_shared<Var>();
  VarPtr updates = std::make_shared<Var>();
  return VectorRef({prim::kPrimTensorScatterUpdate, input, indices, updates});
}

ValueNodePtr TensorScatterAddFission::GetScatterNdPrimNode() const {
  return NewValueNode(std::make_shared<Primitive>(prim::kPrimScatterNdAdd->name()));
}

const BaseRef TensorScatterAddFission::DefinePattern() const {
  VarPtr input = std::make_shared<Var>();
  VarPtr indices = std::make_shared<Var>();
  VarPtr updates = std::make_shared<Var>();
  return VectorRef({prim::kPrimTensorScatterAdd, input, indices, updates});
}

ValueNodePtr TensorScatterSubFission::GetScatterNdPrimNode() const {
  return NewValueNode(std::make_shared<Primitive>(prim::kPrimScatterNdSub->name()));
}

const BaseRef TensorScatterSubFission::DefinePattern() const {
  VarPtr input = std::make_shared<Var>();
  VarPtr indices = std::make_shared<Var>();
  VarPtr updates = std::make_shared<Var>();
  return VectorRef({prim::kPrimTensorScatterSub, input, indices, updates});
}

ValueNodePtr TensorScatterMaxFission::GetScatterNdPrimNode() const {
  return NewValueNode(std::make_shared<Primitive>(prim::kPrimScatterNdMax->name()));
}

const BaseRef TensorScatterMaxFission::DefinePattern() const {
  VarPtr input = std::make_shared<Var>();
  VarPtr indices = std::make_shared<Var>();
  VarPtr updates = std::make_shared<Var>();
  return VectorRef({prim::kPrimTensorScatterMax, input, indices, updates});
}

ValueNodePtr TensorScatterMinFission::GetScatterNdPrimNode() const {
  return NewValueNode(std::make_shared<Primitive>(prim::kPrimScatterNdMin->name()));
}

const BaseRef TensorScatterMinFission::DefinePattern() const {
  VarPtr input = std::make_shared<Var>();
  VarPtr indices = std::make_shared<Var>();
  VarPtr updates = std::make_shared<Var>();
  return VectorRef({prim::kPrimTensorScatterMin, input, indices, updates});
}
}  // namespace opt
}  // namespace mindspore
