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

#include "plugin/device/ascend/optimizer/mindir/shape_unify_mindir.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/arithmetic_ops.h"
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
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kShapeOpName);
  return VectorRef({prim, Xs});
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
    auto tuple_get_node = shape_user.first->cast<CNodePtr>();
    if (common::AnfAlgo::CheckPrimitiveType(tuple_get_node, prim::kPrimTupleGetItem)) {
      auto strided_slice_node = CreateStridedSlice(graph, tensor_shape_node, tuple_get_node, manager);
      MS_EXCEPTION_IF_NULL(strided_slice_node);

      CNodePtr tensor_to_scalar = CreateTensorToScalar(graph, strided_slice_node);
      MS_EXCEPTION_IF_NULL(tensor_to_scalar);

      auto tuple_get_users = manager->node_users()[tuple_get_node];
      for (auto &tuple_get_user : tuple_get_users) {
        auto post_cnode = tuple_get_user.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(post_cnode);
        manager->SetEdge(post_cnode, GetInputNodeIndex(tuple_get_node, post_cnode) + kSizeOne, tensor_to_scalar);
      }
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
  auto prim = NewValueNode(std::make_shared<Primitive>(kTensorShapeOpName));
  MS_EXCEPTION_IF_NULL(prim);
  auto shape_cnode = anf_node->cast<CNodePtr>();
  AnfNodePtrList inputs = {prim, shape_cnode->input(kIndex1)};

  CNodePtr tensor_shape_node = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tensor_shape_node);

  abstract::AbstractBasePtr tmp_abstract;
  auto shape_input_abs = shape_cnode->input(kIndex1)->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(shape_input_abs);
  auto shape = shape_input_abs->shape()->shape();
  ShapeVector tensor_shp({static_cast<int64_t>(shape.size())});
  if (IsDynamic(shape)) {
    if (IsDynamicRank(shape)) {
      tmp_abstract = abstract::MakeAbstract(
        std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeDimAny}), kInt64);
    } else {
      auto elem = std::make_shared<abstract::AbstractScalar>(std::make_shared<ValueAny>(), std::make_shared<Int>(64));
      auto abs_tensor = std::make_shared<abstract::AbstractTensor>(elem, std::make_shared<abstract::Shape>(tensor_shp));
      auto shape_value = MakeValue(shape);
      abs_tensor->set_shape_value(shape_value);
      tmp_abstract = abs_tensor;
    }
  } else {
    auto shp_buf_size = sizeof(int64_t) * shape.size();
    auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, tensor_shp, shape.data(), shp_buf_size);
    tmp_abstract = tensor->ToAbstract();
  }

  // set abstract
  tensor_shape_node->set_fullname_with_scope(anf_node->fullname_with_scope() + "_tensorshape");
  tensor_shape_node->set_abstract(tmp_abstract);
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

  // set abstract
  ShapeVector tensor_shape = {1};
  auto tensor_shape_ptr = std::make_shared<abstract::Shape>(tensor_shape);
  MS_EXCEPTION_IF_NULL(tensor_shape_ptr);

  auto abstract_tensor = shape_node->abstract()->cast<abstract::AbstractTensorPtr>();
  auto type_ptr = abstract_tensor->element()->GetTypeTrack();
  TypeId infer_type = type_ptr->type_id();

  auto tmp_abstract = abstract::MakeAbstract(std::make_shared<abstract::Shape>(tensor_shape), TypeIdToType(infer_type));
  MS_EXCEPTION_IF_NULL(tmp_abstract);

  auto prim = NewValueNode(std::make_shared<Primitive>(kStridedSliceOpName));
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, shape_node, begin_node, end_node, strides_node};
  CNodePtr strided_slice = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(strided_slice);
  strided_slice->set_fullname_with_scope(tuple_get_node->fullname_with_scope() + "_strided_slice");
  strided_slice->set_abstract(tmp_abstract);

  // set attrs, all defaults to zero
  auto primitive = GetCNodePrimitive(strided_slice);
  MS_EXCEPTION_IF_NULL(primitive);
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

}  // namespace opt
}  // namespace mindspore
