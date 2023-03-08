/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/gather_v2_ds_fission.h"
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kOriginPaddingSize = 2;
constexpr size_t kGatherInputNum = 4;
constexpr size_t kGatherInputIndicesIndex = 2;
constexpr size_t kGatherInputAxisIndex = 3;

bool CheckInputs(const CNodePtr &origin_node) {
  MS_EXCEPTION_IF_NULL(origin_node);
  if (common::AnfAlgo::GetInputTensorNum(origin_node) != kGatherV2DynInputTensorNum) {
    MS_LOG(DEBUG) << "GatherV2 in dynamic shape has wrong inputs num, not equal " << kGatherV2DynInputTensorNum
                  << ". CNode= " << origin_node->DebugString();
    return false;
  }
  auto param_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  auto indice_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 1);
  // this optimizer only support embedding_table has dynamic shape
  if (param_shape.empty() || indice_shape.empty() || common::AnfAlgo::IsDynamicShape(origin_node->input(kDim2))) {
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

// only pad operator can run in dynamic shape.
CNodePtr GatherV2DsFission::CreatePad(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                                      const size_t &pad_dim_size) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  std::vector<AnfNodePtr> pad_inputs = {NewValueNode(std::make_shared<Primitive>(kPadDOpName)), origin_node->input(1)};
  auto pad = NewCNode(pad_inputs, graph);
  MS_EXCEPTION_IF_NULL(pad);
  pad->set_scope(origin_node->scope());

  auto param_abstract_shape = origin_node->input(1)->Shape();
  MS_EXCEPTION_IF_NULL(param_abstract_shape);
  if (!param_abstract_shape->isa<abstract::Shape>()) {
    MS_LOG(EXCEPTION) << "The node [" << origin_node->DebugString() << "]'s first input has wrong shape type."
                      << trace::DumpSourceLines(origin_node);
  }
  auto param_dyn_shape = param_abstract_shape->cast<abstract::ShapePtr>();
  ShapeVector shape(param_dyn_shape->shape());
  if (shape.empty()) {
    MS_LOG(EXCEPTION) << "The shape of node [" << origin_node->DebugString() << "]'s first input is empty."
                      << trace::DumpSourceLines(origin_node);
  }
  if (shape[shape.size() - 1] == -1) {
    MS_LOG(EXCEPTION) << "The node [" << origin_node->DebugString()
                      << "]'s first input should not be dynamic, but got shape:" << shape
                      << trace::DumpSourceLines(origin_node);
  }
  shape[shape.size() - 1] = SizeToLong(pad_dim_size);
  auto type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0);
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape);
  MS_EXCEPTION_IF_NULL(abstract);
  pad->set_abstract(abstract);

  std::vector<ValuePtr> elements;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ShapeVector padding_vector(kOriginPaddingSize);
    auto padding_value = MakeValue(padding_vector);
    elements.push_back(padding_value);
  }
  ShapeVector last_padding_vector = {0, SizeToLong(pad_dim_size - 1)};
  auto last_padding_value = MakeValue(last_padding_vector);
  elements.push_back(last_padding_value);
  ValueTuplePtr paddings = std::make_shared<ValueTuple>(elements);
  common::AnfAlgo::SetNodeAttr(kAttrPaddings, paddings, pad);
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), pad);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), pad);
  return pad;
}

CNodePtr GatherV2DsFission::CreateGatherV2Ds(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                                             const CNodePtr &pad, const size_t &pad_dim_size) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  MS_EXCEPTION_IF_NULL(pad);
  if (origin_node->size() != kGatherInputNum) {
    MS_LOG(EXCEPTION) << "In dynamic shape scene, gatherv2 should have 3 inputs, but got " << origin_node->size()
                      << trace::DumpSourceLines(origin_node);
  }
  std::vector<AnfNodePtr> gatherv2_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimGather->name())), pad,
                                             origin_node->input(kGatherInputIndicesIndex),
                                             origin_node->input(kGatherInputAxisIndex)};
  auto gather_v2 = NewCNode(gatherv2_inputs, graph);
  MS_EXCEPTION_IF_NULL(gather_v2);
  gather_v2->set_scope(origin_node->scope());

  auto shape = common::AnfAlgo::GetOutputInferShape(origin_node, 0);
  shape[shape.size() - 1] = SizeToLong(pad_dim_size);
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(origin_node, 0)}, {shape},
                                              gather_v2.get());

  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), gather_v2);
  auto input_names = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(origin_node, kAttrInputNames);
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), gather_v2);
  auto output_names = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(origin_node, kAttrOutputNames);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), gather_v2);
  return gather_v2;
}

CNodePtr GatherV2DsFission::CreateSlice(const FuncGraphPtr &graph, const CNodePtr &gather_v2,
                                        const CNodePtr &gather_v2_padding_8) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(gather_v2);
  MS_EXCEPTION_IF_NULL(gather_v2_padding_8);
  auto gather_v2_shape = common::AnfAlgo::GetOutputInferShape(gather_v2, 0);
  std::vector<int64_t> offsets(gather_v2_shape.size(), 0);
  auto offsets_input = CreateShapeValueNode(graph, offsets, true);
  auto size_input = CreateShapeValueNode(graph, gather_v2_shape, true);
  std::vector<AnfNodePtr> slice_inputs = {NewValueNode(std::make_shared<Primitive>(kSliceOpName)), gather_v2_padding_8,
                                          offsets_input, size_input};
  auto slice = NewCNode(slice_inputs, graph);
  MS_EXCEPTION_IF_NULL(slice);
  slice->set_scope(gather_v2->scope());
  slice->set_abstract(gather_v2->abstract());
  SetInputOutputNames({"x", "offsets", "size"}, {"y"}, slice);
  return slice;
}

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
  auto input_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0);
  constexpr auto PADSIZE32 = 8;
  constexpr auto PADSIZE16 = 16;
  if (input_dtype == kNumberTypeFloat32) {
    pad_dim_size = PADSIZE32;
  } else if (input_dtype == kNumberTypeFloat16) {
    pad_dim_size = PADSIZE16;
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
