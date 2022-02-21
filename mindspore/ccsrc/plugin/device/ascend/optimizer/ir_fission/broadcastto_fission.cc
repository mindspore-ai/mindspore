
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
#include "plugin/device/ascend/optimizer/ir_fission/broadcastto_fission.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
CNodePtr AddCastNode(const FuncGraphPtr &func_graph, const TypeId dst_type, const CNodePtr &input_node,
                     const bool fir_flag) {
  std::vector<AnfNodePtr> new_cast_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name()))};
  std::vector<size_t> shape;
  if (fir_flag) {
    new_cast_inputs.emplace_back(input_node->inputs()[kIndex1]);
    shape = common::AnfAlgo::GetOutputInferShape(input_node->inputs()[kIndex1], 0);
  } else {
    new_cast_inputs.emplace_back(input_node);
    shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  }
  CNodePtr new_cast = NewCNode(new_cast_inputs, func_graph);
  new_cast->set_scope(input_node->scope());
  new_cast->set_abstract(input_node->abstract());
  common::AnfAlgo::SetNodeAttr(kAttrDstType, MakeValue(static_cast<size_t>(dst_type)), new_cast);
  common::AnfAlgo::SetOutputInferTypeAndShape({dst_type}, {shape}, new_cast.get());
  return new_cast;
}
}  // namespace
const BaseRef BroadcasttoFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto broadcastto_prim = std::make_shared<Primitive>(prim::kPrimBroadcastTo->name());
  return VectorRef({broadcastto_prim, Xs});
}

CNodePtr AddBroadCastToNode(const FuncGraphPtr &func_graph, const CNodePtr &input_node,
                            const std::vector<int64_t> &broad_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto input_type = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  std::vector<AnfNodePtr> broadcastto_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimBroadcastTo->name()))};
  broadcastto_inputs.emplace_back(input_node);
  CNodePtr broadcastto_node = NewCNode(broadcastto_inputs, func_graph);
  broadcastto_node->set_scope(input_node->scope());
  broadcastto_node->set_abstract(input_node->abstract());
  std::vector<size_t> out_shape;
  std::transform(broad_shape.begin(), broad_shape.end(), std::back_inserter(out_shape), SizeToLong);
  common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue<std::vector<int64_t>>(broad_shape), broadcastto_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({input_type}, {out_shape}, broadcastto_node.get());
  return broadcastto_node;
}

const AnfNodePtr BroadcasttoFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  auto input_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  if (input_type != kNumberTypeBool) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto broad_shape = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrShape);
  auto cast_to_node = AddCastNode(graph, kNumberTypeInt8, cnode, true);
  auto broadcastto_node = AddBroadCastToNode(graph, cast_to_node, broad_shape);
  auto out_node = AddCastNode(graph, kNumberTypeBool, broadcastto_node, false);
  return out_node;
}
}  // namespace opt
}  // namespace mindspore
