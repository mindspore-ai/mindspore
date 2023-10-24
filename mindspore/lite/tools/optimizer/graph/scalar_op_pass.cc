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
#include <memory>
#include <vector>
#include <string>
#include "ops/array_ops.h"
#include "ops/lite_ops.h"
#include "tools/optimizer/graph/scalar_op_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "mindspore/core/ops/mul.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "mindspore/core/abstract/ops/primitive_infer_map.h"
#include "mindspore/core/utils/anf_utils.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"

/* This pass changes the following pattern(s).

  1. Getting shape from a dynamic tensor, and get item from the shape tuple.
    ###############
    Pattern:
    Shape -> TupleGetItem -> Scalar

    Replace:
    TensorShape -> Cast(int32) -> StridedSlice -> TensorToScalar -> Scalar
    ###############

    The Shape op will be replaced by TensorShape op in order to get a Tensor, and then casted to int32 dtype for Ascend
    dynamic shape mbatch calculation. Followed by StridedSlice, which is a Tensor equivalent to tuple's TupleGetItem.
    And finally casted back to a Scalar by using TensorToScalar op, which is to make sure the old/new pattern outputs'
    types are agree.

  2. Do scalar arithmetic using scalar from TupleGetItem op.
    ###############
    Pattern:
    Scalar -> ScalarMul/ScalarDiv/…. -> Scalar

    Replace:
    ScalarToTensor -> Mul/Div/… -> Tensor -> TensorToScalar
    ###############

    The ScalarXXX arithmetic ops will be replaced by Tensor equivalent arithmetic ops. ScalarToTensor and TensorToScalar
    conversion ops are inserted before and after, in order to make sure the old/new pattern inputs/outputs' types are
    agree.

  3. MakeTuple and Reshape operations using Scalars.
    ###############
    Pattern:
    Scalar -> MakeTuple -> Tuple(Scalar) -> Reshape -> Tensor

    Replace:
    ScalarToTensor -> MakeTuple -> Tuple(Tensor) -> Concat -> Reshape -> Tensor
    ###############

    MakeTuple's Scalar inputs will be converted to Tensors, followed by a Concat op to allow the reshape by a Tensor.
    ScalarToTensor conversion op is inserted before to make sure the old/new pattern inputs/outputs' types are agree.

  4. ScalarToTensor and TensorToScalar ops are temporary placeholders. The last step is to remove them.
    ###############
    Pattern:
    TensorToScalar -> ScalarToTensor
    TensorToScalar -> Cast -> Tensor

    Replace:
    remove both, the Tensor are connected.
    ###############
*/
namespace mindspore::opt {
/*
This function returns the index of the input node, which is used by the user node.
*/
size_t ScalarOpPass::GetInputNodeIndex(const AnfNodePtr &input, const CNodePtr &user_node) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(user_node);

  AnfNodePtrList input_list = user_node->inputs();
  auto pos = std::find(input_list.begin(), input_list.end(), input);
  if (pos == input_list.end()) {
    MS_LOG(EXCEPTION) << input->fullname_with_scope() << " is not the input of " << user_node->fullname_with_scope();
  }

  // The first input is Primitive and needs to be skipped.
  return std::distance(input_list.begin() + kSizeOne, pos);
}

/*
Create a Tensor with type scalar. This pass assumes that the scalar is from TensorShape, which will be integers.
*/
ValueNodePtr ScalarOpPass::GenerateScalarValueTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                                     int input_index) {
  lite::DataInfo data_info;
  auto ret = lite::FetchConstData(anf_node->cast<CNodePtr>(), input_index, converter::kFmkTypeMs, &data_info, false);
  MS_CHECK_TRUE_RET(ret == lite::RET_OK, nullptr);
  if (data_info.data_type_ != kNumberTypeInt32 && data_info.data_type_ != kNumberTypeInt64) {
    MS_LOG(ERROR) << "Unsupported scalar data type: " << data_info.data_type_ << ", need to add support.";
    return nullptr;
  }
  int32_t scalar_value = *reinterpret_cast<int32_t *>(data_info.data_.data());
  ShapeVector const_data_shape = {1};
  tensor::TensorPtr const_data_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, const_data_shape);
  auto *val = static_cast<int32_t *>(const_data_tensor->data_c());
  *val = scalar_value;
  auto const_value_node = NewValueNode(const_data_tensor);
  const_value_node->set_abstract(const_data_tensor->ToAbstract());
  func_graph->AddValueNode(const_value_node);
  return const_value_node;
}

CNodePtr ScalarOpPass::GenerateScalarToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                              int input_index) {
  auto prim = NewValueNode(std::make_shared<Primitive>(kScalarToTensorOpName));
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto scalar_cnode = anf_node->cast<CNodePtr>();
  auto scalar_input = scalar_cnode->input(input_index);
  AnfNodePtrList inputs = {prim, scalar_input};
  CNodePtr scalar_to_tensor = func_graph->NewCNode(inputs);
  MS_CHECK_TRUE_RET(scalar_to_tensor != nullptr, nullptr);

  // Data type of the tensor should be set as an attr of ScalarToTensor op.
  TypeId scalar_data_type;
  if (opt::GetDataTypeFromAnfNode(scalar_cnode->input(input_index), &scalar_data_type) != RET_OK) {
    MS_LOG(ERROR) << "Failed to get " << anf_node->fullname_with_scope() << " output tensor data type.";
    return nullptr;
  }
  auto primitive = GetCNodePrimitive(scalar_to_tensor);
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  primitive->set_attr("dtype", TypeIdToType(scalar_data_type));

  // set abstract
  ShapeVector tensor_shape = {1};
  auto tensor_shape_ptr = std::make_shared<abstract::Shape>(tensor_shape);
  MS_CHECK_TRUE_MSG(tensor_shape_ptr != nullptr, nullptr, "tensor_shape_ptr is nullptr.");

  auto tmp_abstract =
    abstract::MakeAbstract(std::make_shared<abstract::Shape>(tensor_shape), TypeIdToType(scalar_data_type));
  MS_CHECK_TRUE_MSG(tmp_abstract != nullptr, nullptr, "make AbstractTensor failed");
  scalar_to_tensor->set_abstract(tmp_abstract);
  return scalar_to_tensor;
}

CNodePtr ScalarOpPass::GenerateTensorToScalar(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                              bool is_curr_node) {
  auto prim = NewValueNode(std::make_shared<Primitive>(kTensorToScalarOpName));
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto input_cnode = anf_node->cast<CNodePtr>();
  AnfNodePtrList inputs = {prim, input_cnode->input(kIndexOne)};
  if (is_curr_node) {  // insert TensorToScalar after current anf_node
    inputs = {prim, anf_node};
  }
  CNodePtr tensor_to_scalar = func_graph->NewCNode(inputs);
  MS_CHECK_TRUE_RET(tensor_to_scalar != nullptr, nullptr);

  // set abstract
  TypeId type_id;
  (void)GetDataTypeFromAnfNode(anf_node, &type_id);

  auto tmp_abstract = std::make_shared<abstract::AbstractScalar>(kValueAny, TypeIdToType(type_id));
  MS_CHECK_TRUE_MSG(tmp_abstract != nullptr, nullptr, "make AbstractScalar failed");
  tensor_to_scalar->set_abstract(tmp_abstract);
  return tensor_to_scalar;
}

CNodePtr ScalarOpPass::GenerateTensorShape(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node) {
  auto prim = NewValueNode(std::make_shared<Primitive>(kTensorShapeOpName));
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto shape_cnode = anf_node->cast<CNodePtr>();
  AnfNodePtrList inputs = {prim, shape_cnode->input(kIndexOne)};

  CNodePtr tensor_shape_node = func_graph->NewCNode(inputs);
  MS_CHECK_TRUE_MSG(tensor_shape_node != nullptr, nullptr, "tensor_shape_node is nullptr.");

  abstract::AbstractBasePtr tmp_abstract;
  auto shape_input_abs = shape_cnode->input(kIndexOne)->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_CHECK_TRUE_MSG(shape_input_abs != nullptr, nullptr, "shape input abstract is not AbstractTensor.");
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

/*
Create a ValueNode with single scalar as input.
*/
ValueNodePtr ScalarOpPass::GenerateScalarValueTuple(const FuncGraphPtr &func_graph, int64_t value) {
  std::vector<int64_t> vec({value});
  auto tuple_value = MakeValue(vec);
  auto tuple_node = NewValueNode(tuple_value);
  tuple_node->set_abstract(tuple_value->ToAbstract());
  func_graph->AddValueNode(tuple_node);
  return tuple_node;
}

CNodePtr ScalarOpPass::GenerateStridedSlice(const FuncGraphPtr &func_graph, const AnfNodePtr &shape_node,
                                            const AnfNodePtr &tuple_get_node, const FuncGraphManagerPtr &manager) {
  auto begin_index = GetTupleGetItemOutIndex(tuple_get_node->cast<CNodePtr>());
  MS_CHECK_TRUE_MSG(begin_index >= 0, nullptr, "begin index is less than zero.");

  // set inputs
  auto begin_node = GenerateScalarValueTuple(func_graph, begin_index);
  MS_CHECK_TRUE_MSG(begin_node != nullptr, nullptr, "generate StridedSlice begin node failed.");
  auto end_node = GenerateScalarValueTuple(func_graph, begin_index + kSizeOne);
  MS_CHECK_TRUE_MSG(end_node != nullptr, nullptr, "generate StridedSlice end node failed.");
  auto strides_node = GenerateScalarValueTuple(func_graph, kSizeOne);
  MS_CHECK_TRUE_MSG(strides_node != nullptr, nullptr, "generate StridedSlice strides node failed.");

  // set abstract
  ShapeVector tensor_shape = {1};
  auto tensor_shape_ptr = std::make_shared<abstract::Shape>(tensor_shape);
  MS_CHECK_TRUE_MSG(tensor_shape_ptr != nullptr, nullptr, "tensor_shape_ptr is nullptr.");
  TypeId infer_type;
  auto ret = GetDataTypeFromAnfNode(shape_node, &infer_type);
  MS_CHECK_TRUE_MSG(ret == RET_OK, nullptr, "get data_type from node failed.");

  auto tmp_abstract = abstract::MakeAbstract(std::make_shared<abstract::Shape>(tensor_shape), TypeIdToType(infer_type));
  MS_CHECK_TRUE_MSG(tmp_abstract != nullptr, nullptr, "make AbstractTensor failed");

  auto prim = NewValueNode(std::make_shared<Primitive>(kStridedSliceOpName));
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  AnfNodePtrList inputs = {prim, shape_node, begin_node, end_node, strides_node};
  CNodePtr strided_slice = func_graph->NewCNode(inputs);
  MS_CHECK_TRUE_RET(strided_slice != nullptr, nullptr);
  strided_slice->set_fullname_with_scope(tuple_get_node->fullname_with_scope() + "_strided_slice");
  strided_slice->set_abstract(tmp_abstract);

  // set attrs, all defaults to zero
  auto primitive = GetCNodePrimitive(strided_slice);
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  primitive->set_attr("new_axis_mask", MakeValue<int64_t>(0));
  primitive->set_attr("shrink_axis_mask", MakeValue<int64_t>(0));
  primitive->set_attr("end_mask", MakeValue<int64_t>(0));
  primitive->set_attr("begin_mask", MakeValue<int64_t>(0));
  primitive->set_attr("ellipsis_mask", MakeValue<int64_t>(0));

  return strided_slice;
}

STATUS ScalarOpPass::ReplaceScalarOp(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                     const FuncGraphManagerPtr &manager, const PrimitivePtr &replace_op_prim) {
  auto replace_op_prim_node = NewValueNode(replace_op_prim);
  MS_CHECK_TRUE_RET(replace_op_prim_node != nullptr, lite::RET_ERROR);
  AnfNodePtrList replace_op_inputs = {replace_op_prim_node};

  auto scalar_cnode = anf_node->cast<CNodePtr>();
  std::vector<mindspore::AnfNodePtr> scalar_inputs = {};
  scalar_inputs.push_back(scalar_cnode->input(kIndexOne));
  scalar_inputs.push_back(scalar_cnode->input(kIndexTwo));
  for (size_t i = 0; i < scalar_inputs.size(); i++) {
    if (!scalar_inputs[i]->isa<ValueNode>()) {
      auto node = GenerateScalarToTensor(func_graph, anf_node, i + kSizeOne);
      MS_CHECK_TRUE_RET(node != nullptr, lite::RET_ERROR);
      replace_op_inputs.push_back(node);
    } else {
      auto node = GenerateScalarValueTensor(func_graph, anf_node, i + kSizeOne);
      MS_CHECK_TRUE_RET(node != nullptr, lite::RET_ERROR);
      replace_op_inputs.push_back(node);
    }
  }
  CNodePtr replace_op = func_graph->NewCNode(replace_op_inputs);
  MS_CHECK_TRUE_RET(replace_op != nullptr, lite::RET_ERROR);

  ShapeVector replace_op_shape = {1};
  auto replace_op_shape_ptr = std::make_shared<abstract::Shape>(replace_op_shape);
  MS_CHECK_TRUE_MSG(replace_op_shape_ptr != nullptr, RET_ERROR, "replace op is nullptr.");

  // Replace op has the same type as the first input
  auto abstract = replace_op->input(kIndexOne)->abstract();
  auto tmp_abstract = abstract->Clone();
  tmp_abstract->set_shape(replace_op_shape_ptr);
  replace_op->set_abstract(tmp_abstract);

  CNodePtr tensor_to_scalar = GenerateTensorToScalar(func_graph, replace_op, true);

  // Set input of the Scalar op users to tensor_to_scalar
  auto orig_scalar_op_cnode = anf_node->cast<CNodePtr>();
  auto node_users = manager->node_users()[orig_scalar_op_cnode];
  for (auto &node_user : node_users) {
    auto post_cnode = node_user.first->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(post_cnode != nullptr, lite::RET_ERROR);
    manager->SetEdge(post_cnode, GetInputNodeIndex(anf_node, post_cnode) + kSizeOne, tensor_to_scalar);
  }

  return lite::RET_OK;
}

STATUS ScalarOpPass::ReplaceMakeTuple(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                      const FuncGraphManagerPtr &manager) {
  auto make_tuple_cnode = anf_node->cast<CNodePtr>();
  if (!utils::isa<mindspore::abstract::AbstractScalarPtr>(make_tuple_cnode->input(kIndexOne)->abstract())) {
    return lite::RET_NO_CHANGE;
  }

  abstract::BaseShapePtrList tuple_shape_list;
  TypePtrList tuple_type_list;
  for (size_t i = kIndexOne; i < make_tuple_cnode->size(); i++) {
    auto make_tuple_input = make_tuple_cnode->input(i);

    // Parse abstract shape for modified MakeTuple
    ShapeVector scalar_shape = {1};
    tuple_shape_list.push_back(std::make_shared<abstract::Shape>(scalar_shape));

    // Parse abstract type for modified MakeTuple
    TypeId orig_type_id;
    auto ret = GetDataTypeFromAnfNode(make_tuple_input, &orig_type_id);
    MS_CHECK_TRUE_MSG(ret != RET_ERROR, lite::RET_ERROR, "get datatype from MakeTuple input failed.");

    // Insert ScalarToTensor before MakeTuple
    if (!make_tuple_input->isa<ValueNode>()) {
      auto node = GenerateScalarToTensor(func_graph, anf_node, i);
      MS_CHECK_TRUE_MSG(node != nullptr, lite::RET_ERROR, "generate ScalarToTensor node failed.");

      // Parse abstract type for modified MakeTuple
      ret = GetDataTypeFromAnfNode(node, &orig_type_id);
      MS_CHECK_TRUE_MSG(ret != RET_ERROR, lite::RET_ERROR, "get datatype from MakeTuple input failed.");
      tuple_type_list.push_back(TypeIdToType(orig_type_id));

      manager->SetEdge(anf_node, i, node);
    } else {  // For ValueNode the input type is int32
      auto node = GenerateScalarValueTensor(func_graph, anf_node, i);
      MS_CHECK_TRUE_MSG(node != nullptr, lite::RET_ERROR, "generate ScalarValueTensor node failed.");

      // Parse abstract type for modified MakeTuple
      ret = GetDataTypeFromAnfNode(node, &orig_type_id);
      MS_CHECK_TRUE_MSG(ret != RET_ERROR, lite::RET_ERROR, "get datatype from MakeTuple input failed.");
      tuple_type_list.push_back(TypeIdToType(orig_type_id));

      manager->SetEdge(anf_node, i, node);
    }
  }

  // Apply modified abstract to MakeTuple
  auto tmp_abstract = abstract::MakeAbstract(std::make_shared<abstract::TupleShape>(tuple_shape_list),
                                             std::make_shared<Tuple>(tuple_type_list));
  anf_node->set_abstract(tmp_abstract);

  // Insert concat after MakeTuple
  std::vector<AnfNodePtr> concat_input_vec({anf_node});
  auto concat_node = GenConcatNode(func_graph, concat_input_vec,
                                   anf_node->cast<CNodePtr>()->fullname_with_scope() + "_concat_make_tuple");
  auto primitive = GetCNodePrimitive(concat_node);
  MS_CHECK_TRUE_RET(primitive != nullptr, lite::RET_ERROR);
  int64_t num_of_inputs = anf_node->cast<CNodePtr>()->inputs().size() - kSizeOne;
  primitive->set_attr("N", MakeValue<int64_t>(num_of_inputs));
  primitive->set_attr("inputNums", MakeValue<int64_t>(num_of_inputs));

  // The first input type is used as the type for concat (need to add type check)
  TypeId make_tuple_type;
  if (opt::GetDataTypeFromAnfNode(anf_node, &make_tuple_type) != RET_OK) {
    MS_LOG(ERROR) << "Failed to get " << anf_node->fullname_with_scope() << " output tensor data type.";
    return lite::RET_ERROR;
  }
  auto concat_abstract = abstract::MakeAbstract(std::make_shared<abstract::Shape>(ShapeVector({num_of_inputs})),
                                                TypeIdToType(make_tuple_type));
  concat_node->set_abstract(concat_abstract);

  // set MakeTuple users' input to concat
  auto make_tuple_users = manager->node_users()[anf_node];
  for (auto &make_tuple_user : make_tuple_users) {
    auto post_cnode = make_tuple_user.first->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(post_cnode != nullptr, lite::RET_ERROR, "MakeTuple user is null.");
    manager->SetEdge(post_cnode, GetInputNodeIndex(anf_node, post_cnode) + kSizeOne, concat_node);
  }

  return lite::RET_OK;
}

STATUS ScalarOpPass::ReplaceShapeTupleGet(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                          const FuncGraphManagerPtr &manager) {
  auto shape_cnode = anf_node->cast<CNodePtr>();

  // Replace Shape by TensorShape
  auto tensor_shape_node = GenerateTensorShape(func_graph, anf_node);
  MS_CHECK_TRUE_MSG(tensor_shape_node != nullptr, lite::RET_ERROR, "generate TensorShape node failed.");
  ShapeVector tensor_shape_shape;
  auto ret = FetchShapeFromAbstract(tensor_shape_node->abstract(), &tensor_shape_shape);
  MS_CHECK_TRUE_MSG(ret != RET_ERROR, lite::RET_ERROR, "fetch shape from TensorShape node failed.");
  auto cast_abstract =
    abstract::MakeAbstract(std::make_shared<abstract::Shape>(tensor_shape_shape), TypeIdToType(kNumberTypeInt32));

  CNodePtr cast_node = nullptr;
  auto shape_users = manager->node_users()[shape_cnode];
  for (auto &shape_user : shape_users) {
    auto tuple_get_node = shape_user.first->cast<CNodePtr>();
    if (CheckPrimitiveType(tuple_get_node, prim::kPrimTupleGetItem)) {
      if (cast_node == nullptr) {
        cast_node =
          GenCastNode(func_graph, tensor_shape_node, tensor_shape_node->fullname_with_scope() + "_cast_tensorshape",
                      kNumberTypeInt32, cast_abstract);
      }
      auto strided_slice_node = GenerateStridedSlice(func_graph, cast_node, tuple_get_node, manager);
      MS_CHECK_TRUE_MSG(strided_slice_node != nullptr, lite::RET_ERROR, "generate StridedSlice node failed.");

      CNodePtr tensor_to_scalar = GenerateTensorToScalar(func_graph, strided_slice_node, true);
      MS_CHECK_TRUE_MSG(tensor_to_scalar != nullptr, lite::RET_ERROR, "generate TensorToScalar node failed.");

      auto tuple_get_users = manager->node_users()[tuple_get_node];
      for (auto &tuple_get_user : tuple_get_users) {
        auto post_cnode = tuple_get_user.first->cast<CNodePtr>();
        MS_CHECK_TRUE_MSG(post_cnode != nullptr, lite::RET_ERROR, "TupleGetItem user is null.");
        manager->SetEdge(post_cnode, GetInputNodeIndex(tuple_get_node, post_cnode) + kSizeOne, tensor_to_scalar);
      }
    }
  }

  return lite::RET_OK;
}

STATUS ScalarOpPass::RemoveTensorToScalar(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                          const FuncGraphManagerPtr &manager) {
  auto tensor_to_scalar_cnode = anf_node->cast<CNodePtr>();
  auto tensor_to_scalar_users = manager->node_users()[tensor_to_scalar_cnode];
  auto parent_node = tensor_to_scalar_cnode->input(kIndexOne);
  for (auto &tensor_to_scalar_user : tensor_to_scalar_users) {
    auto user_node = tensor_to_scalar_user.first->cast<CNodePtr>();
    if (CheckPrimitiveType(user_node, prim::kPrimScalarToTensor) || CheckPrimitiveType(user_node, prim::kPrimCast)) {
      auto child_node_users = manager->node_users()[user_node];
      for (auto &child_node_user : child_node_users) {
        auto child_node = child_node_user.first->cast<CNodePtr>();
        manager->SetEdge(child_node, GetInputNodeIndex(user_node, child_node) + kSizeOne, parent_node);
      }
    } else {
      std::string prim_name = "";
      (void)GetPrimitiveType(user_node, &prim_name);
      MS_LOG(ERROR) << "Cannot handle primitive " << prim_name << " after TensorToScalar, please check graph.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS ScalarOpPass::RunScalarOpPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto node_list = TopoSort(func_graph->get_return());
  STATUS status = lite::RET_NO_CHANGE;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    // First replace all Scalar ops to tensor equivalents
    if (CheckPrimitiveType(node, prim::kPrimScalarMul)) {
      status = this->ReplaceScalarOp(func_graph, node, manager, prim::kPrimMul);
    } else if (CheckPrimitiveType(node, prim::kPrimScalarDiv)) {
      status = this->ReplaceScalarOp(func_graph, node, manager, prim::kPrimRealDiv);
    } else if (CheckPrimitiveType(node, prim::kPrimScalarFloorDiv)) {
      status = this->ReplaceScalarOp(func_graph, node, manager, prim::kPrimFloorDiv);
    } else if (CheckPrimitiveType(node, prim::kPrimScalarSub)) {
      status = this->ReplaceScalarOp(func_graph, node, manager, prim::kPrimSub);
    } else if (CheckPrimitiveType(node, prim::kPrimScalarAdd)) {
      status = this->ReplaceScalarOp(func_graph, node, manager, prim::kPrimAdd);
    } else if (CheckPrimitiveType(node, prim::kPrimScalarCast)) {
      MS_LOG(ERROR) << "For models with dynamic input shapes, ScalarCast node conversion has not been supported yet, "
                       "please check cast operations such as \"int(some_var)\" in the front-end code and remove them.";
      status = lite::RET_NOT_SUPPORT;
    }

    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Failed to run scalar op pass at cnode: " << node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return status;
}

STATUS ScalarOpPass::RunMakeTuplePass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto node_list = TopoSort(func_graph->get_return());
  auto status = lite::RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    // Then change MakeTuple's input from a tuple of scalars to a tuple of tensors
    if (CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
      status = this->ReplaceMakeTuple(func_graph, node, manager);
    }

    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Failed to run make tuple pass at cnode: " << node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS ScalarOpPass::RunShapeTupleGetPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto node_list = TopoSort(func_graph->get_return());
  auto status = lite::RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    // Replace Shape+TupleGetItem to TensorShape+StridedSlice
    if (CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
      auto tuple_get_input = node->cast<CNodePtr>()->input(kIndexOne);
      if (CheckPrimitiveType(tuple_get_input, prim::kPrimShape)) {
        MS_LOG(INFO) << "Start processing Shape + TupleGetItem pass.";
        status = this->ReplaceShapeTupleGet(func_graph, tuple_get_input, manager);
      }
    }

    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Failed to run shape tuple get pass at cnode: " << node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS ScalarOpPass::RunRemoveTensorToScalarPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto node_list = TopoSort(func_graph->get_return());
  auto status = lite::RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    // Remove TensorToScalar + ScalarToTensor
    // Remove TensorToScalar + Cast
    if (CheckPrimitiveType(node, prim::kPrimTensorToScalar)) {
      MS_LOG(DEBUG) << "Found TensorToScalar, start removing...";
      status = this->RemoveTensorToScalar(func_graph, node, manager);
    }

    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Failed to run remove TensorToScalar pass at cnode: " << node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

/*
This pass checks the arithmetic ops have correct infer types when all TensorToScalar/ScalarToTensor ops are removed. If
datatypes do not agree, insert cast op.
*/
STATUS ScalarOpPass::RunArithmeticCheckPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    // Check arithmetic op infer type, insert cast op if two inputs do not agree
    if (CheckPrimitiveType(node, prim::kPrimMul) || CheckPrimitiveType(node, prim::kPrimDiv) ||
        CheckPrimitiveType(node, prim::kPrimFloorDiv) || CheckPrimitiveType(node, prim::kPrimRealDiv) ||
        CheckPrimitiveType(node, prim::kPrimSub)) {
      auto first_input = node->cast<CNodePtr>()->input(kIndexOne);
      auto second_input = node->cast<CNodePtr>()->input(kIndexTwo);

      TypeId first_data_type;
      if (opt::GetDataTypeFromAnfNode(first_input, &first_data_type) != RET_OK) {
        MS_LOG(ERROR) << "Failed to get arithmetic op first input tensor data type.";
        return lite::RET_ERROR;
      }
      TypeId second_data_type;
      if (opt::GetDataTypeFromAnfNode(second_input, &second_data_type) != RET_OK) {
        MS_LOG(ERROR) << "Failed to get arithmetic op second input tensor data type.";
        return lite::RET_ERROR;
      }
      if (first_data_type == second_data_type) {
        continue;
      }

      // Insert cast node before second input, and set infer type the same as the first input
      auto cast_data_type = first_data_type;
      ShapeVector cast_shape;
      if (FetchShapeFromAbstract(second_input->abstract(), &cast_shape) != lite::RET_OK) {
        MS_LOG(ERROR) << "Fetch shape from second input abstract failed!";
        return lite::RET_ERROR;
      }
      auto new_cast_abstract =
        abstract::MakeAbstract(std::make_shared<abstract::Shape>(cast_shape), TypeIdToType(cast_data_type));
      auto new_cast_node =
        GenCastNode(func_graph, second_input, second_input->fullname_with_scope() + "cast_after_second_in",
                    cast_data_type, new_cast_abstract);
      new_cast_node->set_abstract(new_cast_abstract);
      manager->SetEdge(node, kIndexTwo, new_cast_node);
    }
  }
  return lite::RET_OK;
}

bool ScalarOpPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto status = RunShapeTupleGetPass(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  auto scalar_replace_status = RunScalarOpPass(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  status = RunMakeTuplePass(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  status = RunRemoveTensorToScalarPass(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  if (scalar_replace_status != lite::RET_NO_CHANGE) {
    status = RunArithmeticCheckPass(func_graph, manager);
    MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  }

  return true;
}
}  // namespace mindspore::opt
