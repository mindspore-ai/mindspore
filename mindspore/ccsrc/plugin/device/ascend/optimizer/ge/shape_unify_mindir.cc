/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ge/shape_unify_mindir.h"
#include <memory>
#include <vector>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"

/* This pass changes the following pattern.

  Getting shape from a dynamic tensor, and get item from the shape tuple.
    ###############
    Pattern:
    Shape -> TupleGetItem -> Scalar

    Replace:
    TensorShape -> StridedSlice -> TensorToScalar -> Scalar
    ###############

    The Shape op will be replaced by TensorShape op in order to get a Tensor, and then casted to int32 dtype for Ascend
    dynamic shape mbatch calculation. Followed by StridedSlice, which is a Tensor equivalent to tuple's TupleGetItem.
    And finally casted back to a Scalar by using TensorToScalar op, which is to make sure the old/new pattern outputs'
    types are agree.
*/

namespace mindspore {
namespace opt {

const BaseRef ShapeUnifyMindIR::DefinePattern() const {
  VarPtr x = std::make_shared<Var>();
  auto prim = std::make_shared<Primitive>(kShapeOpName);
  return VectorRef({prim, x});
}

const AnfNodePtr ShapeUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto shape_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(shape_cnode);

  // Replace Shape by TensorShape
  auto tensor_shape_node = CreateTensorShape(graph, node);
  MS_EXCEPTION_IF_NULL(tensor_shape_node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto shape_users = manager->node_users()[shape_cnode];
  for (auto &shape_user : shape_users) {
    auto user_cnode = shape_user.first->cast<CNodePtr>();
    if (common::AnfAlgo::CheckPrimitiveType(user_cnode, prim::kPrimTupleGetItem)) {
      auto strided_slice_node = CreateStridedSlice(graph, tensor_shape_node, user_cnode, manager);
      MS_EXCEPTION_IF_NULL(strided_slice_node);

      CNodePtr tensor_to_scalar = CreateTensorToScalar(graph, strided_slice_node);
      MS_EXCEPTION_IF_NULL(tensor_to_scalar);

      auto tuple_get_users = manager->node_users()[user_cnode];
      for (auto &tuple_get_user : tuple_get_users) {
        auto post_cnode = tuple_get_user.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(post_cnode);
        manager->SetEdge(post_cnode, GetInputNodeIndex(user_cnode, post_cnode) + kSizeOne, tensor_to_scalar);
      }
    } else if (common::AnfAlgo::CheckPrimitiveType(user_cnode, prim::kPrimReturn)) {
      auto tensor_to_tuple_cnode = CreateTensorToTuple(graph, tensor_shape_node);
      manager->SetEdge(user_cnode, GetInputNodeIndex(shape_cnode, user_cnode) + kSizeOne, tensor_to_tuple_cnode);
    }
  }
  return tensor_shape_node;
}

CNodePtr ShapeUnifyMindIR::CreateTensorToScalar(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node) const {
  auto prim = NewValueNode(std::make_shared<Primitive>(kTensorToScalarOpName));
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, anf_node};
  CNodePtr tensor_to_scalar = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tensor_to_scalar);

  // set abstract
  auto abstract_tensor = anf_node->abstract()->cast<abstract::AbstractTensorPtr>();
  auto type_ptr = abstract_tensor->element()->GetTypeTrack();
  TypeId type_id = type_ptr->type_id();
  auto tmp_abstract = std::make_shared<abstract::AbstractScalar>(kValueAny, TypeIdToType(type_id));
  MS_EXCEPTION_IF_NULL(tmp_abstract);
  tensor_to_scalar->set_abstract(tmp_abstract);
  return tensor_to_scalar;
}

CNodePtr ShapeUnifyMindIR::CreateTensorShape(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node) const {
  auto prim = std::make_shared<Primitive>(kTensorShapeOpName);
  MS_EXCEPTION_IF_NULL(prim);
  auto shape_cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(shape_cnode);
  AnfNodePtrList inputs = {NewValueNode(prim), shape_cnode->input(kIndex1)};

  CNodePtr tensor_shape_node = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tensor_shape_node);

  auto abs = InferAbstract(prim, {shape_cnode->input(kIndex1)});
  MS_EXCEPTION_IF_NULL(abs);
  tensor_shape_node->set_abstract(abs);
  return tensor_shape_node;
}

CNodePtr ShapeUnifyMindIR::CreateStridedSlice(const FuncGraphPtr &func_graph, const AnfNodePtr &shape_node,
                                              const AnfNodePtr &tuple_get_node,
                                              const FuncGraphManagerPtr &manager) const {
  auto begin_index = common::AnfAlgo::GetTupleGetItemOutIndex(tuple_get_node->cast<CNodePtr>());
  if (begin_index < 0) {
    MS_LOG(EXCEPTION) << "begin index is less than zero.";
  }

  // set inputs
  auto begin_node = CreateScalarValueTuple(func_graph, begin_index);
  MS_EXCEPTION_IF_NULL(begin_node);
  auto end_node = CreateScalarValueTuple(func_graph, begin_index + kSizeOne);
  MS_EXCEPTION_IF_NULL(end_node);
  auto strides_node = CreateScalarValueTuple(func_graph, kSizeOne);
  MS_EXCEPTION_IF_NULL(strides_node);

  auto prim = NewValueNode(std::make_shared<Primitive>(kStridedSliceOpName));
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, shape_node, begin_node, end_node, strides_node};
  CNodePtr strided_slice = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(strided_slice);
  strided_slice->set_fullname_with_scope(tuple_get_node->fullname_with_scope() + "_strided_slice");
  auto primitive = GetCNodePrimitive(strided_slice);
  MS_EXCEPTION_IF_NULL(primitive);

  // set abstract
  auto tmp_abstract = InferAbstract(primitive, {shape_node, begin_node, end_node, strides_node});
  MS_EXCEPTION_IF_NULL(tmp_abstract);
  strided_slice->set_abstract(tmp_abstract);

  // set attrs, all defaults to zero
  primitive->set_attr("new_axis_mask", MakeValue<int64_t>(0));
  primitive->set_attr("shrink_axis_mask", MakeValue<int64_t>(0));
  primitive->set_attr("end_mask", MakeValue<int64_t>(0));
  primitive->set_attr("begin_mask", MakeValue<int64_t>(0));
  primitive->set_attr("ellipsis_mask", MakeValue<int64_t>(0));

  return strided_slice;
}

/*
Create a ValueNode with single scalar as input.
*/
ValueNodePtr ShapeUnifyMindIR::CreateScalarValueTuple(const FuncGraphPtr &func_graph, int64_t value) const {
  std::vector<int64_t> vec({value});
  auto tuple_value = MakeValue(vec);
  auto tuple_node = NewValueNode(tuple_value);
  tuple_node->set_abstract(tuple_value->ToAbstract());
  func_graph->AddValueNode(tuple_node);
  return tuple_node;
}

CNodePtr ShapeUnifyMindIR::CreateTensorToTuple(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto prim = std::make_shared<Primitive>(kTensorToTupleOpName);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {NewValueNode(prim), node};
  CNodePtr tensor_to_tuple = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tensor_to_tuple);

  // set abstract
  auto abs = InferAbstract(prim, {node});
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for TensorToTuple op is " << abs->ToString();
  tensor_to_tuple->set_abstract(abs);

  return tensor_to_tuple;
}

}  // namespace opt
}  // namespace mindspore
