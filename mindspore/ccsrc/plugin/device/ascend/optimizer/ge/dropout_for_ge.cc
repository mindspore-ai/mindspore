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

#include "plugin/device/ascend/optimizer/ge/dropout_for_ge.h"
#include <vector>
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int64_t kAlignSize = 128;
constexpr char kKeepProbAttrName[] = "keep_prob";
constexpr char kSeed0AttrName[] = "Seed0";
constexpr char kSeed1AttrName[] = "Seed1";
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

abstract::ShapePtr GetDropoutInputShape(const CNodePtr &dropout_node) {
  MS_EXCEPTION_IF_NULL(dropout_node);
  auto input = dropout_node->input(kInputIndexOne);
  MS_EXCEPTION_IF_NULL(input);
  auto input_base_shape = input->Shape();
  MS_EXCEPTION_IF_NULL(input_base_shape);
  auto input_shape = input_base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  return input_shape;
}

CNodePtr CreateDynamicShapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node_input,
                                 const abstract::ShapePtr &input_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_shape);
  std::vector<AnfNodePtr> dynamic_shape_inputs{NewValueNode(std::make_shared<Primitive>("Shape")), node_input};
  CNodePtr dynamic_shape = func_graph->NewCNode(dynamic_shape_inputs);
  MS_EXCEPTION_IF_NULL(dynamic_shape);
  ShapeVector tensor_shp({static_cast<int64_t>(input_shape->shape().size())});
  auto dynamic_shape_abstract =
    std::make_shared<abstract::AbstractTensor>(kInt64, std::make_shared<abstract::Shape>(tensor_shp));
  MS_EXCEPTION_IF_NULL(dynamic_shape_abstract);
  dynamic_shape->set_abstract(dynamic_shape_abstract);
  return dynamic_shape;
}

CNodePtr CreateDropoutGenMaskCNode(const FuncGraphPtr &func_graph, const CNodePtr &dropout,
                                   const AnfNodePtr &keep_prob_value, const abstract::ShapePtr &input_shape,
                                   const ValuePtr &seed_0, const ValuePtr &seed_1) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dropout);
  MS_EXCEPTION_IF_NULL(input_shape);
  std::vector<AnfNodePtr> dropout_gen_mask_inputs =
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kDropoutGenMaskOpName))};
  if (input_shape->IsDynamic() || common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, dropout)) {
    CNodePtr dynamic_shape = CreateDynamicShapeCNode(func_graph, dropout->input(kIndex1), input_shape);
    dynamic_shape->set_scope(dropout->scope());
    dropout_gen_mask_inputs.push_back(dynamic_shape);
    dropout_gen_mask_inputs.push_back(keep_prob_value);
  } else {
    auto shape_value = CreateShapeValueNode(func_graph, input_shape->shape(), true);
    dropout_gen_mask_inputs.push_back(shape_value);
    dropout_gen_mask_inputs.push_back(keep_prob_value);
  }
  CNodePtr dropout_gen_mask = opt::NewCNode(dropout_gen_mask_inputs, func_graph, {dropout});
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  if (dropout->HasPrimalAttr(kAttrFusion)) {
    dropout_gen_mask->AddPrimalAttr(kAttrFusion, dropout->GetPrimalAttr(kAttrFusion));
  }

  std::shared_ptr<abstract::AbstractTensor> gen_mask_abstract;
  if (input_shape->IsDynamic() || common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, dropout)) {
    ShapeVector mask_shp = {abstract::Shape::kShapeDimAny};
    auto gen_mask_shape = std::make_shared<abstract::Shape>(mask_shp);
    MS_EXCEPTION_IF_NULL(gen_mask_shape);
    gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, gen_mask_shape);
  } else {
    auto gen_mask_shape = CalGenMaskOutputShape(input_shape->shape());
    gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, gen_mask_shape);
  }
  MS_EXCEPTION_IF_NULL(gen_mask_abstract);
  dropout_gen_mask->set_abstract(gen_mask_abstract);
  dropout_gen_mask->set_scope(dropout->scope());
  common::AnfAlgo::CopyNodeAttrs(dropout, dropout_gen_mask);
  common::AnfAlgo::SetNodeAttr(kSeed0AttrName, seed_0, dropout_gen_mask);
  common::AnfAlgo::SetNodeAttr(kSeed1AttrName, seed_1, dropout_gen_mask);
  return dropout_gen_mask;
}

const BaseRef DropoutForGE::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  return VectorRef({prim::kPrimDropout, x1});
}

const AnfNodePtr DropoutForGE::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_node);
  auto origin_prim = GetValueNode<PrimitivePtr>(dropout_node->input(0));
  MS_EXCEPTION_IF_NULL(origin_prim);
  auto keep_prob = origin_prim->GetAttr(kKeepProbAttrName);
  auto seed_0 = origin_prim->GetAttr(kSeed0AttrName);
  auto seed_1 = origin_prim->GetAttr(kSeed1AttrName);

  auto input_shape_ptr = GetDropoutInputShape(dropout_node);
  auto keep_prob_node = NewValueNode(keep_prob);
  MS_EXCEPTION_IF_NULL(keep_prob_node);
  keep_prob_node->set_abstract(keep_prob->ToAbstract());
  AnfNodePtr gen_mask_input_prob = keep_prob_node;
  auto dtype_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(dropout_node, 0);
  if (dtype_id == TypeId::kNumberTypeFloat16) {
    auto dtype_value = MakeValue(TypeIdToType(dtype_id));
    auto stype_value = MakeValue(TypeIdToType(TypeId::kNumberTypeFloat32));
    auto cast_input_type = NewValueNode(dtype_value);
    cast_input_type->set_abstract(dtype_value->ToAbstract());

    auto cast_node = node->func_graph()->NewCNode(
      {NewValueNode(std::make_shared<Primitive>(kCastOpName)), keep_prob_node, cast_input_type});
    MS_EXCEPTION_IF_NULL(cast_node);
    common::AnfAlgo::SetNodeAttr(kSrcAttrName, stype_value, cast_node);
    common::AnfAlgo::SetNodeAttr(kDstAttrName, dtype_value, cast_node);
    common::AnfAlgo::SetNodeAttr(kDstTypeAttrName, dtype_value, cast_node);
    auto cast_abstract = std::make_shared<abstract::AbstractScalar>(TypeIdToType(dtype_id));
    cast_node->set_abstract(cast_abstract);
    gen_mask_input_prob = cast_node;
  }
  CNodePtr dropout_gen_mask_node =
    CreateDropoutGenMaskCNode(graph, dropout_node, gen_mask_input_prob, input_shape_ptr, seed_0, seed_1);

  auto dropout_do_mask_node =
    node->func_graph()->NewCNode({NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                  dropout_node->input(kInputIndexOne), dropout_gen_mask_node, gen_mask_input_prob});
  auto do_mask_abstract = dropout_node->input(kInputIndexOne)->abstract();
  dropout_do_mask_node->set_abstract(do_mask_abstract);

  std::vector<abstract::AbstractBasePtr> make_tuple_input;
  make_tuple_input.push_back(do_mask_abstract);
  make_tuple_input.push_back(dropout_gen_mask_node->abstract());
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), dropout_do_mask_node,
                                               dropout_gen_mask_node};
  auto new_make_tuple_node = NewCNode(make_tuple_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_make_tuple_node);
  auto tuple_abstract = std::make_shared<abstract::AbstractTuple>(make_tuple_input);
  new_make_tuple_node->set_abstract(tuple_abstract);
  return new_make_tuple_node;
}

const BaseRef DropoutGradForGE::DefinePattern() const {
  VarPtr x1 = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimDropoutGrad, x1});
}

const AnfNodePtr DropoutGradForGE::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_grad_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_grad_node);
  auto origin_prim = GetValueNode<PrimitivePtr>(dropout_grad_node->input(0));
  auto keep_prob = origin_prim->GetAttr(kKeepProbAttrName);

  auto keep_prob_node = NewValueNode(keep_prob);
  keep_prob_node->set_abstract(keep_prob->ToAbstract());

  AnfNodePtr do_mask_input_prob = keep_prob_node;
  auto dtype_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(dropout_grad_node, 0);
  if (dtype_id == TypeId::kNumberTypeFloat16) {
    auto dtype_value = MakeValue(TypeIdToType(dtype_id));
    auto stype_value = MakeValue(TypeIdToType(TypeId::kNumberTypeFloat32));
    auto cast_input_type = NewValueNode(dtype_value);
    cast_input_type->set_abstract(dtype_value->ToAbstract());
    auto cast_node = node->func_graph()->NewCNode(
      {NewValueNode(std::make_shared<Primitive>(kCastOpName)), keep_prob_node, cast_input_type});
    MS_EXCEPTION_IF_NULL(cast_node);
    common::AnfAlgo::SetNodeAttr(kSrcAttrName, stype_value, cast_node);
    common::AnfAlgo::SetNodeAttr(kDstAttrName, dtype_value, cast_node);
    common::AnfAlgo::SetNodeAttr(kDstTypeAttrName, dtype_value, cast_node);
    auto cast_abstract = std::make_shared<abstract::AbstractScalar>(TypeIdToType(dtype_id));
    cast_node->set_abstract(cast_abstract);
    do_mask_input_prob = cast_node;
  }
  auto dropout_do_mask_node = node->func_graph()->NewCNode(
    {NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)), dropout_grad_node->input(kInputIndexOne),
     dropout_grad_node->input(kInputIndexTwo), do_mask_input_prob});
  MS_EXCEPTION_IF_NULL(dropout_do_mask_node);
  auto do_mask_abstract = node->abstract();
  dropout_do_mask_node->set_abstract(do_mask_abstract);
  return dropout_do_mask_node;
}
}  // namespace opt
}  // namespace mindspore
