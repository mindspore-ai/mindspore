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
#include "backend/optimizer/ascend/ir_fission/gather_v2_ds_fission.h"
#include <memory>
#include <vector>
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
// only pad operator can run in dynamic shape.
CNodePtr CreatePad(const FuncGraphPtr &graph, const CNodePtr &origin_node, const size_t &pad_dim_size) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  std::vector<AnfNodePtr> pad_inputs = {NewValueNode(std::make_shared<Primitive>(kPadOpName)), origin_node->input(1)};
  auto pad = graph->NewCNode(pad_inputs);
  MS_EXCEPTION_IF_NULL(pad);
  pad->set_scope(origin_node->scope());

  auto param_abstract_shape = origin_node->input(1)->Shape();
  MS_EXCEPTION_IF_NULL(param_abstract_shape);
  if (!param_abstract_shape->isa<abstract::Shape>()) {
    MS_LOG(EXCEPTION) << "Gatherv2 's first input has wrong shape type";
  }
  auto param_dyn_shape = param_abstract_shape->cast<abstract::ShapePtr>();
  ShapeVector shape(param_dyn_shape->shape());
  if (shape.empty()) {
    MS_LOG(EXCEPTION) << "Gatherv2 's shape is empty";
  }
  if (shape[shape.size() - 1] == -1) {
    MS_LOG(EXCEPTION) << "Dim needs pad should not be dynamic";
  }
  shape[shape.size() - 1] = pad_dim_size;
  auto type_id = AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0);
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape);
  if (param_dyn_shape->max_shape().size() == param_dyn_shape->shape().size() &&
      param_dyn_shape->min_shape().size() == param_dyn_shape->shape().size()) {
    ShapeVector max_shape(param_dyn_shape->max_shape());
    ShapeVector min_shape(param_dyn_shape->min_shape());
    ShapeVector new_shape(shape);
    max_shape[max_shape.size() - 1] = pad_dim_size;
    min_shape[min_shape.size() - 1] = pad_dim_size;
    abstract->set_shape(std::make_shared<abstract::Shape>(new_shape, min_shape, max_shape));
  }
  pad->set_abstract(abstract);

  std::vector<ValuePtr> elements;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ShapeVector padding_vector(2);
    auto padding_value = MakeValue(padding_vector);
    elements.push_back(padding_value);
  }
  ShapeVector last_padding_vector = {0, SizeToLong(pad_dim_size - 1)};
  auto last_padding_value = MakeValue(last_padding_vector);
  elements.push_back(last_padding_value);
  ValueTuplePtr paddings = std::make_shared<ValueTuple>(elements);
  AnfAlgo::SetNodeAttr(kAttrPaddings, paddings, pad);
  AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), pad);
  AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), pad);
  AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), pad);
  return pad;
}

CNodePtr CreateGatherV2Ds(const FuncGraphPtr &graph, const CNodePtr &origin_node, const CNodePtr &pad,
                          const size_t &pad_dim_size) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  MS_EXCEPTION_IF_NULL(pad);
  if (origin_node->size() != 4) {
    MS_LOG(EXCEPTION) << "In dynamic shape scene, gatherv2 should have 3 inputs";
  }
  std::vector<AnfNodePtr> gatherv2_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimGather->name())), pad,
                                             origin_node->input(2), origin_node->input(3)};
  auto gather_v2 = graph->NewCNode(gatherv2_inputs);
  MS_EXCEPTION_IF_NULL(gather_v2);
  gather_v2->set_scope(origin_node->scope());

  auto shape = AnfAlgo::GetOutputInferShape(origin_node, 0);
  shape[shape.size() - 1] = pad_dim_size;
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(origin_node, 0)}, {shape}, gather_v2.get());
  AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), gather_v2);
  AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), gather_v2);
  auto input_names = AnfAlgo::GetNodeAttr<std::vector<std::string>>(origin_node, kAttrInputNames);
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), gather_v2);
  auto output_names = AnfAlgo::GetNodeAttr<std::vector<std::string>>(origin_node, kAttrOutputNames);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), gather_v2);
  return gather_v2;
}

CNodePtr CreateSlice(const FuncGraphPtr &graph, const CNodePtr &gather_v2, const CNodePtr &gather_v2_padding_8) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(gather_v2);
  MS_EXCEPTION_IF_NULL(gather_v2_padding_8);
  std::vector<AnfNodePtr> slice_inputs = {NewValueNode(std::make_shared<Primitive>(kSliceOpName)), gather_v2_padding_8};
  auto slice = graph->NewCNode(slice_inputs);
  MS_EXCEPTION_IF_NULL(slice);
  slice->set_scope(gather_v2->scope());
  slice->set_abstract(gather_v2->abstract());
  auto gather_v2_shape = AnfAlgo::GetOutputInferShape(gather_v2, 0);
  std::vector<size_t> offsets(gather_v2_shape.size(), 0);
  AnfAlgo::SetNodeAttr(kAttrBegin, MakeValue(Convert2Long(offsets)), slice);
  AnfAlgo::SetNodeAttr(kAttrSize, MakeValue(Convert2Long(gather_v2_shape)), slice);
  return slice;
}

bool CheckInputs(const CNodePtr &origin_node) {
  MS_EXCEPTION_IF_NULL(origin_node);
  if (AnfAlgo::GetInputTensorNum(origin_node) != kGatherV2DynInputTensorNum) {
    MS_LOG(DEBUG) << "GatherV2 in dynamic shape has wrong inputs num, not equal " << kGatherV2DynInputTensorNum
                  << ". CNode= " << origin_node->DebugString();
    return false;
  }
  auto param_shape = AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  auto indice_shape = AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 1);
  // this optimizer only support embedding_table has dynamic shape
  if (param_shape.empty() || indice_shape.empty() || AnfAlgo::IsDynamicShape(origin_node->input(2))) {
    return false;
  }
  if (param_shape[param_shape.size() - 1] != 1) {
    MS_LOG(DEBUG) << "GatherV2 in dynamic shape is not need fission. The last value of input0's shape is "
                  << param_shape[param_shape.size() - 1];
    return false;
  }
  return true;
}
}  // namespace

const BaseRef GatherV2DsFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimGather, Xs});
  return pattern;
}

const AnfNodePtr GatherV2DsFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto origin_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_node);
  if (!CheckInputs(origin_node)) {
    return nullptr;
  }
  size_t pad_dim_size;
  auto input_dtype = AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0);
  if (input_dtype == kNumberTypeFloat32) {
    pad_dim_size = 8;
  } else if (input_dtype == kNumberTypeFloat16) {
    pad_dim_size = 16;
  } else {
    MS_LOG(DEBUG) << "GatherV2 data type not in (float32, float16), no need change";
    return nullptr;
  }
  CNodePtr gather_v2_8;
  auto pad = CreatePad(graph, origin_node, pad_dim_size);
  gather_v2_8 = CreateGatherV2Ds(graph, origin_node, pad, pad_dim_size);
  return CreateSlice(graph, origin_node, gather_v2_8);
}
}  // namespace opt
}  // namespace mindspore
