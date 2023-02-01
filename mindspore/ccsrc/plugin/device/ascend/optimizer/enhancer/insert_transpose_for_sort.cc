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

#include "plugin/device/ascend/optimizer/enhancer/insert_transpose_for_sort.h"
#include <algorithm>
#include <string>
#include "plugin/device/ascend/optimizer/create_node_helper.h"

namespace mindspore {
namespace opt {
const size_t kSortOutputNum = 2;
const BaseRef InsertTransposeForSort::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kSortOpName);
  return VectorRef({prim, Xs});
}

abstract::BaseShapePtr InferTransposeOutputShape(const abstract::BaseShapePtr &shape,
                                                 const std::vector<int64_t> &perm) {
  auto shapeptr = shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shapeptr);
  if (shapeptr->shape().size() != perm.size()) {
    MS_LOG(EXCEPTION) << "Length of input shape and perm must be same, bug got input shape: " << shape
                      << ", perm: " << perm;
  }
  ShapeVector in_shape = shapeptr->shape();
  ShapeVector out_shape;
  for (int64_t i : perm) {
    auto idx = LongToSize(i);
    out_shape.push_back(in_shape[idx]);
  }

  abstract::ShapePtr out_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
  return out_shape_ptr;
}

CNodePtr InsertForInput(const FuncGraphPtr &func_graph, const CNodePtr &node, const std::vector<int64_t> &perm) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> new_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(node)};

  auto in_node = common::AnfAlgo::GetInputNode(node, 0);
  auto type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  auto in_shape = common::AnfAlgo::GetPrevNodeOutputDetailShape(node, 0);
  auto transpose_out_shape = InferTransposeOutputShape(in_shape, perm);

  auto ori_out_types = AnfAlgo::GetAllOutputInferDataTypes(node);
  auto perm_value_input = CreatePermValueNode(func_graph, perm);
  std::vector<AnfNodePtr> trans_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimTranspose->name()))};
  (void)trans_inputs.push_back(in_node);
  (void)trans_inputs.push_back(perm_value_input);
  auto transpose = func_graph->NewCNode(trans_inputs);
  MS_EXCEPTION_IF_NULL(transpose);
  common::AnfAlgo::SetOutputTypeAndDetailShape({type}, {transpose_out_shape}, transpose.get());
  std::vector<std::string> transpose_input_names{"x", "perm"};
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(transpose_input_names), transpose);
  transpose = CreateNodeHelper::CreateNodeWithCheck(transpose)->cast<CNodePtr>();
  new_inputs.push_back(transpose);

  CNodePtr new_cnode = nullptr;
  // cnode changed so make a new cnode to differ from original one.
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  if (kernel_graph == nullptr) {
    new_cnode = std::make_shared<CNode>(*node);
  } else {
    new_cnode = kernel_graph->NewCNode(node);
  }
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_inputs(new_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(ori_out_types, {transpose_out_shape, transpose_out_shape},
                                               new_cnode.get());
  return new_cnode;
}

AnfNodePtr InsertForOutput(const FuncGraphPtr &func_graph, const CNodePtr &orig_node, const CNodePtr &node,
                           const std::vector<int64_t> &perm) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(orig_node);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto update_states = common::AnfAlgo::GetUpdateStateUsers(manager, orig_node);
  for (auto &update_state : update_states) {
    manager->SetEdge(update_state.first, update_state.second, node);
  }
  if (manager->node_users()[orig_node].empty()) {
    return node;
  }

  std::vector<AnfNodePtr> tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  auto out_num = AnfAlgo::GetOutputElementNum(node);

  for (size_t output_idx = 0; output_idx < out_num; output_idx++) {
    auto tuple_getitem = CreatTupleGetItemNode(func_graph, node, output_idx);
    std::vector<AnfNodePtr> transpose_inputs;
    auto prim = std::make_shared<Primitive>(prim::kPrimTranspose->name());
    auto perm_value_input = CreatePermValueNode(func_graph, perm);
    (void)transpose_inputs.push_back(NewValueNode(prim));
    (void)transpose_inputs.push_back(tuple_getitem);
    (void)transpose_inputs.push_back(perm_value_input);

    auto shape = common::AnfAlgo::GetOutputDetailShape(node, output_idx);
    auto type = common::AnfAlgo::GetOutputInferDataType(node, output_idx);
    auto transpose_out_shape = InferTransposeOutputShape(shape, perm);

    CNodePtr transpose = func_graph->NewCNode(transpose_inputs);
    MS_EXCEPTION_IF_NULL(transpose);
    common::AnfAlgo::SetOutputTypeAndDetailShape({type}, {transpose_out_shape}, transpose.get());
    std::vector<std::string> transpose_input_names{"x", "perm"};
    common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(transpose_input_names), transpose);
    transpose = CreateNodeHelper::CreateNodeWithCheck(transpose)->cast<CNodePtr>();
    tuple_inputs.push_back(transpose);
  }
  auto make_tuple = func_graph->NewCNode(tuple_inputs);
  return make_tuple;
}

AnfNodePtr InsertTranspose(const FuncGraphPtr &func_graph, const CNodePtr &node, const ShapeVector &in_shape,
                           int64_t axis) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  int64_t axis_pos = axis < 0 ? (SizeToLong(in_shape.size()) + axis) : axis;
  if (axis_pos < 0 || axis_pos >= SizeToLong(in_shape.size())) {
    MS_LOG(EXCEPTION) << "Axis attr value [" << axis << "] of node " << node->fullname_with_scope() << " is invalid";
  }
  std::vector<int64_t> perm(in_shape.size(), 0);
  int64_t i = 0;
  std::generate(perm.begin(), perm.end(), [&] { return i++; });
  std::reverse(perm.begin() + axis_pos, perm.end());

  auto new_sort = InsertForInput(func_graph, node, perm);
  common::AnfAlgo::SetNodeAttr("axis", MakeValue(IntToLong(-1)), new_sort);
  return InsertForOutput(func_graph, node, new_sort, perm);
}

const AnfNodePtr InsertTransposeForSort::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  if (op_name != kSortOpName) {
    return nullptr;
  }

  auto axis = common::AnfAlgo::GetNodeAttr<int64_t>(node, "axis");
  auto in_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  if (axis == -1 || (axis > 0 && (LongToSize(axis) == in_shape.size() - 1))) {
    return nullptr;
  }
  // if the attr value axis of sort is not -1, or not the rank of sort, need insert transpose for correct execution
  return InsertTranspose(func_graph, cnode, in_shape, axis);
}
}  // namespace opt
}  // namespace mindspore
