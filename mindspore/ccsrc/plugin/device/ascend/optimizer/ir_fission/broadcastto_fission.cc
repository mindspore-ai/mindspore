
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
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
static const size_t kDynamicBroadcastToInputNum = 2;
const BaseRef BroadcasttoFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto broadcastto_prim = std::make_shared<Primitive>(prim::kPrimBroadcastTo->name());
  return VectorRef({broadcastto_prim, Xs});
}

CNodePtr BroadcasttoFission::AddBroadCastToNode(const FuncGraphPtr &func_graph, const CNodePtr &input_node,
                                                const std::vector<int64_t> &broad_shape) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto input_type = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  std::vector<AnfNodePtr> broadcastto_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimBroadcastTo->name()))};
  broadcastto_inputs.emplace_back(input_node);
  CNodePtr broadcastto_node = NewCNode(broadcastto_inputs, func_graph);
  broadcastto_node->set_scope(input_node->scope());
  broadcastto_node->set_abstract(input_node->abstract());
  common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue<std::vector<int64_t>>(broad_shape), broadcastto_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({input_type}, {broad_shape}, broadcastto_node.get());
  return broadcastto_node;
}

const AnfNodePtr BroadcasttoFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto input_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  if (input_type != kNumberTypeBool) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "BroadcastTo don't support dynamic shape, node: " << cnode->fullname_with_scope();
  }
  auto broad_shape = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrShape);
  auto cast_to_node = AddCastNode(graph, kNumberTypeInt8, cnode, true);
  auto broadcastto_node = AddBroadCastToNode(graph, cast_to_node, broad_shape);
  if (common::AnfAlgo::HasNodeAttr(kAttrCustAicpu, cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue<std::string>(prim::kPrimBroadcastTo->name()),
                                 broadcastto_node);
  }
  auto out_node = AddCastNode(graph, kNumberTypeBool, broadcastto_node, false);
  return out_node;
}

const BaseRef DynamicBroadcastToFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto dynamic_broadcastto_prim = std::make_shared<Primitive>(prim::kPrimDynamicBroadcastTo->name());
  return VectorRef({dynamic_broadcastto_prim, Xs});
}

CNodePtr DynamicBroadcastToFission::AddDynamicBroadCastToNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                              const CNodePtr &input0_node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_type = common::AnfAlgo::GetOutputInferDataType(input0_node, 0);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(cnode, 0);
  std::vector<AnfNodePtr> dynamic_broadcastto_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimDynamicBroadcastTo->name())), input0_node,
    cnode->inputs()[kIndex2]};

  CNodePtr broadcastto_node = NewCNode(dynamic_broadcastto_inputs, func_graph);
  common::AnfAlgo::CopyNodeAttrs(cnode, broadcastto_node);
  broadcastto_node->set_scope(cnode->scope());
  broadcastto_node->set_abstract(cnode->abstract());
  common::AnfAlgo::SetOutputInferTypeAndShape({input_type}, {output_shape}, broadcastto_node.get());
  return broadcastto_node;
}

const AnfNodePtr DynamicBroadcastToFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto input_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  if (input_type != kNumberTypeBool) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::GetInputTensorNum(cnode) != kDynamicBroadcastToInputNum) {
    MS_LOG(EXCEPTION) << "DynamicBroadcastTo only support 2 inputs, but got "
                      << common::AnfAlgo::GetInputTensorNum(cnode) << ", node: " << cnode->fullname_with_scope();
  }
  auto cast_to_node = AddCastNode(graph, kNumberTypeInt8, cnode, true);
  auto dynamic_broadcastto_node = AddDynamicBroadCastToNode(graph, cnode, cast_to_node);
  if (common::AnfAlgo::HasNodeAttr(kAttrCustAicpu, cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue<std::string>(prim::kPrimDynamicBroadcastTo->name()),
                                 dynamic_broadcastto_node);
  }
  auto out_node = AddCastNode(graph, kNumberTypeBool, dynamic_broadcastto_node, false);
  return out_node;
}
}  // namespace opt
}  // namespace mindspore
