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

#include "frontend/optimizer/irpass/ge/dropout_for_ge.h"

#include <functional>
#include "pybind_api/pybind_patch.h"
#include "pybind_api/ir/tensor_py.h"
#include "pipeline/pynative/base.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "include/common/utils/python_adapter.h"
#include "mindspore/core/mindapi/ir/common.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
namespace mindspore {
namespace opt {
namespace irpass {
namespace {
constexpr int64_t kAlignSize = 128;
constexpr char kKeepProbAttrName[] = "keep_prob";
constexpr char kSeed0AttrName[] = "Seed0";
constexpr char kSeed1AttrName[] = "Seed1";
constexpr char kOpsArrayFunctionName[] = "mindspore.ops.operations.array_ops";
constexpr char kOpsNNFunctionName[] = "mindspore.ops.operations.nn_ops";
constexpr char kDstAttrName[] = "DstT";
constexpr char kSrcAttrName[] = "SrcT";
constexpr char kDstTypeAttrName[] = "dst_type";
constexpr size_t kInputIndexOne = 1;
constexpr size_t kInputIndexTwo = 2;
}  // namespace
std::vector<int64_t> CalGenMaskOutputShape(const std::vector<int64_t> &shape) {
  auto output_size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  auto output_count = output_size / kAlignSize;
  if (output_size % kAlignSize != 0) {
    output_count++;
  }
  auto ret = output_count * kAlignSize;
  MS_LOG(INFO) << "Output_size: " << ret;
  return {ret};
}

PrimitivePtr GetPrimitiveFromPyAdapter(const py::object &prim_obj) {
  const auto &adapter = py::cast<PrimitivePyAdapterPtr>(prim_obj);
  MS_EXCEPTION_IF_NULL(adapter);
  auto attached_prim = adapter->attached_primitive();
  if (attached_prim == nullptr) {
    attached_prim = std::make_shared<PrimitivePy>(prim_obj, adapter);
    adapter->set_attached_primitive(attached_prim);
  }
  return attached_prim->cast<PrimitivePtr>();
}

AnfNodePtr DropoutForGE::operator()(const OptimizerPtr &opt, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimDropout, {IsNode})(node);

  if (!is_match_ && node->func_graph() == nullptr) {
    return nullptr;
  }
  auto dropout_node = node->cast<CNodePtr>();
  auto origin_prim = GetValueNode<PrimitivePtr>(dropout_node->input(0));
  auto keep_prob = origin_prim->GetAttr(kKeepProbAttrName);
  auto seed_0 = origin_prim->GetAttr(kSeed0AttrName);
  auto seed_1 = origin_prim->GetAttr(kSeed1AttrName);

  auto shape = dropout_node->input(kInputIndexOne)->Shape();
  auto input_shape_ptr = shape->cast<abstract::ShapePtr>();
  if (input_shape_ptr->IsDynamic()) {
    MS_LOG(EXCEPTION) << "Dropout does not support dynamic shape in GE backend for now";
  }
  auto shape_vector = input_shape_ptr->shape();
  auto shape_value = MakeValue(shape_vector);
  auto gen_mask_input_shape = NewValueNode(shape_value);
  gen_mask_input_shape->set_abstract(shape_value->ToAbstract());
  auto keep_prob_node = NewValueNode(keep_prob);
  keep_prob_node->set_abstract(keep_prob->ToAbstract());

  AnfNodePtr gen_mask_input_prob = keep_prob_node;
  auto dtype_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(dropout_node, 0);
  if (dtype_id == TypeId::kNumberTypeFloat16) {
    auto cast_obj = python_adapter::GetPyFn(kOpsArrayFunctionName, "Cast")();
    auto cast_prim = GetPrimitiveFromPyAdapter(cast_obj);

    auto dtype_value = MakeValue(TypeIdToType(dtype_id));
    auto stype_value = MakeValue(TypeIdToType(TypeId::kNumberTypeFloat32));
    auto cast_input_type = NewValueNode(dtype_value);
    cast_input_type->set_abstract(dtype_value->ToAbstract());
    cast_prim->set_attr(kSrcAttrName, stype_value);
    cast_prim->set_attr(kDstAttrName, dtype_value);
    cast_prim->set_attr(kDstTypeAttrName, dtype_value);
    auto cast_node = node->func_graph()->NewCNode(cast_prim, {keep_prob_node, cast_input_type});
    auto cast_abstract = std::make_shared<abstract::AbstractScalar>(TypeIdToType(dtype_id));
    cast_node->set_abstract(cast_abstract);
    gen_mask_input_prob = cast_node;
  }

  auto dropout_gen_mask_obj = python_adapter::GetPyFn(kOpsNNFunctionName, "DropoutGenMask")();
  auto gen_mask_prim = GetPrimitiveFromPyAdapter(dropout_gen_mask_obj);
  gen_mask_prim->set_attr(kSeed0AttrName, seed_0);
  gen_mask_prim->set_attr(kSeed1AttrName, seed_1);

  auto mask_shape = CalGenMaskOutputShape(shape_vector);
  auto dropout_gen_mask_node = node->func_graph()->NewCNode(gen_mask_prim, {gen_mask_input_shape, gen_mask_input_prob});
  auto gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, mask_shape);
  dropout_gen_mask_node->set_abstract(gen_mask_abstract);

  auto dropout_obj = python_adapter::GetPyFn(kOpsNNFunctionName, "DropoutDoMask")();
  auto do_mask_prim = GetPrimitiveFromPyAdapter(dropout_obj);

  auto dropout_do_mask_node = node->func_graph()->NewCNode(
    do_mask_prim, {dropout_node->input(kInputIndexOne), dropout_gen_mask_node, gen_mask_input_prob});
  auto do_mask_abstract = dropout_node->input(kInputIndexOne)->abstract();
  dropout_do_mask_node->set_abstract(do_mask_abstract);

  std::vector<abstract::AbstractBasePtr> make_tuple_input;
  make_tuple_input.push_back(do_mask_abstract);
  make_tuple_input.push_back(gen_mask_abstract);

  auto new_make_tuple_node =
    node->func_graph()->NewCNode(prim::kPrimMakeTuple, {dropout_do_mask_node, dropout_gen_mask_node});
  auto tuple_abstract = std::make_shared<abstract::AbstractTuple>(make_tuple_input);
  new_make_tuple_node->set_abstract(tuple_abstract);
  return new_make_tuple_node;
}

AnfNodePtr DropoutGradForGE::operator()(const OptimizerPtr &opt, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimDropoutGrad, {IsNode, IsNode})(node);
  auto dropout_grad_node = node->cast<CNodePtr>();
  auto origin_prim = GetValueNode<PrimitivePtr>(dropout_grad_node->input(0));
  auto keep_prob = origin_prim->GetAttr(kKeepProbAttrName);

  auto dropout_do_mask_obj = python_adapter::GetPyFn(kOpsNNFunctionName, "DropoutDoMask")();
  auto do_mask_prim = GetPrimitiveFromPyAdapter(dropout_do_mask_obj);

  auto keep_prob_node = NewValueNode(keep_prob);
  keep_prob_node->set_abstract(keep_prob->ToAbstract());

  AnfNodePtr do_mask_input_prob = keep_prob_node;
  auto dtype_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(dropout_grad_node, 0);
  if (dtype_id == TypeId::kNumberTypeFloat16) {
    auto cast_obj = python_adapter::GetPyFn(kOpsArrayFunctionName, "Cast")();
    auto cast_prim = GetPrimitiveFromPyAdapter(cast_obj);

    auto dtype_value = MakeValue(TypeIdToType(dtype_id));
    auto stype_value = MakeValue(TypeIdToType(TypeId::kNumberTypeFloat32));
    auto cast_input_type = NewValueNode(dtype_value);
    cast_input_type->set_abstract(dtype_value->ToAbstract());
    cast_prim->set_attr(kSrcAttrName, stype_value);
    cast_prim->set_attr(kDstAttrName, dtype_value);
    cast_prim->set_attr(kDstTypeAttrName, dtype_value);
    auto cast_node = node->func_graph()->NewCNode(cast_prim, {keep_prob_node, cast_input_type});
    auto cast_abstract = std::make_shared<abstract::AbstractScalar>(TypeIdToType(dtype_id));
    cast_node->set_abstract(cast_abstract);
    do_mask_input_prob = cast_node;
  }
  auto dropout_do_mask_node = node->func_graph()->NewCNode(
    do_mask_prim,
    {dropout_grad_node->input(kInputIndexOne), dropout_grad_node->input(kInputIndexTwo), do_mask_input_prob});
  auto do_mask_abstract = node->abstract();
  dropout_do_mask_node->set_abstract(do_mask_abstract);
  return dropout_do_mask_node;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
