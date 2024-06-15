/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ge/convert_pad_v3_paddings.h"
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/op_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "mindspore/core/ops/array_op_name.h"
#include "mindspore/core/ops/sequence_op_name.h"
#include "mindspore/core/ops/auto_generate/gen_ops_name.h"

namespace mindspore {
namespace opt {
constexpr auto kLength8 = 8;
constexpr auto kStep2 = 2;

bool ConvertBasePaddings::HasDynPaddings(const CNodePtr &cnode) const {
  auto input_paddings = common::AnfAlgo::GetInputNode(cnode, kIndex1);
  MS_EXCEPTION_IF_NULL(input_paddings);
  auto paddings_abstract = input_paddings->abstract();
  MS_EXCEPTION_IF_NULL(paddings_abstract);
  auto paddings_value = paddings_abstract->GetValue();
  MS_EXCEPTION_IF_NULL(paddings_value);
  auto input_paddings_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, kIndex1);
  if (input_paddings_type_id == kNumberTypeInt32) {
    auto paddings_array_value = ops::GetArrayValue<int32_t>(paddings_value);
    return !paddings_array_value.has_value();
  }
  auto paddings_array_value = ops::GetArrayValue<int64_t>(paddings_value);
  return !paddings_array_value.has_value();
}

const CNodePtr ConvertBasePaddings::CreateReshapeNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
                                                      const ShapeVector &shape) const {
  auto prim = std::make_shared<Primitive>(kReshapeOpName);
  MS_EXCEPTION_IF_NULL(prim);
  auto shape_value_node = CreateValueNodeWithKernelInfo(graph, MakeValue(shape));
  MS_EXCEPTION_IF_NULL(shape_value_node);
  AnfNodePtrList reshape_inputs = {NewValueNode(prim), input_node, shape_value_node};
  auto reshape_node = NewCNode(reshape_inputs, graph);
  MS_EXCEPTION_IF_NULL(reshape_node);
  auto abs = InferAbstract(prim, {input_node, shape_value_node});
  MS_EXCEPTION_IF_NULL(abs);
  reshape_node->set_abstract(abs);
  return reshape_node;
}

const CNodePtr ConvertBasePaddings::CreateStridedSliceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                                           int64_t index) const {
  // set inputs
  auto begin_node = CreateValueNodeWithKernelInfo(func_graph, MakeValue(std::vector<int64_t>{index}));
  MS_EXCEPTION_IF_NULL(begin_node);
  auto end_node = CreateValueNodeWithKernelInfo(func_graph, MakeValue(std::vector<int64_t>{index + 1}));
  MS_EXCEPTION_IF_NULL(end_node);
  auto strides_node = CreateValueNodeWithKernelInfo(func_graph, MakeValue(std::vector<int64_t>{1}));
  MS_EXCEPTION_IF_NULL(strides_node);
  int64_t const_value = 0;
  auto begin_mask = CreateValueNodeWithKernelInfo(func_graph, MakeValue(const_value));
  MS_EXCEPTION_IF_NULL(begin_mask);
  auto end_mask = CreateValueNodeWithKernelInfo(func_graph, MakeValue(const_value));
  MS_EXCEPTION_IF_NULL(end_mask);
  auto ellipsis_mask = CreateValueNodeWithKernelInfo(func_graph, MakeValue(const_value));
  MS_EXCEPTION_IF_NULL(ellipsis_mask);
  auto new_axis_mask = CreateValueNodeWithKernelInfo(func_graph, MakeValue(const_value));
  MS_EXCEPTION_IF_NULL(new_axis_mask);
  auto shrink_axis_mask = CreateValueNodeWithKernelInfo(func_graph, MakeValue(const_value));
  MS_EXCEPTION_IF_NULL(shrink_axis_mask);

  auto prim = std::make_shared<Primitive>(kStridedSliceOpName);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {NewValueNode(prim), input_node, begin_node,    end_node,      strides_node,
                           begin_mask,         end_mask,   ellipsis_mask, new_axis_mask, shrink_axis_mask};
  auto strided_slice_node = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(strided_slice_node);
  auto abs = InferAbstract(prim, {input_node, begin_node, end_node, strides_node, begin_mask, end_mask, ellipsis_mask,
                                  new_axis_mask, shrink_axis_mask});
  MS_EXCEPTION_IF_NULL(abs);
  strided_slice_node->set_abstract(abs);
  static size_t slice_index = 0;
  strided_slice_node->set_fullname_with_scope(input_node->fullname_with_scope() + "_strided_slice_" +
                                              std::to_string(slice_index++));
  return strided_slice_node;
}

const CNodePtr ConvertBasePaddings::CreateConcatNode(const FuncGraphPtr &func_graph,
                                                     const std::vector<AnfNodePtr> &concat_input_vec,
                                                     const std::string &concat_node_name) const {
  auto concat_prim = std::make_shared<Primitive>(kConcatOpName);
  MS_EXCEPTION_IF_NULL(concat_prim);
  std::vector<int64_t> dyn_input_sizes = {SizeToLong(concat_input_vec.size()), -1};
  concat_prim->AddAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes));

  AnfNodePtrList inputs = {NewValueNode(concat_prim)};
  inputs.insert(inputs.end(), concat_input_vec.begin(), concat_input_vec.end());
  int64_t axis = 0;
  auto axis_node = CreateValueNodeWithKernelInfo(func_graph, MakeValue(axis));
  inputs.push_back(axis_node);
  auto concat_node = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(concat_node);

  std::vector<AnfNodePtr> concat_inputs = concat_input_vec;
  concat_inputs.push_back(axis_node);
  auto concat_abs = InferAbstract(concat_prim, concat_inputs);
  MS_EXCEPTION_IF_NULL(concat_abs);
  concat_node->set_abstract(concat_abs);
  concat_node->set_fullname_with_scope(concat_node_name);
  return concat_node;
}

const CNodePtr ConvertBasePaddings::ProcessSliceNConcat(const FuncGraphPtr &func_graph, const AnfNodePtr &pad_node,
                                                        const AnfNodePtr &input_node, const int64_t &padding_dst_length,
                                                        const int64_t &padding_src_length) const {
  auto prim = GetCNodePrimitive(pad_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto paddings_contiguous = GetValue<bool>(prim->GetAttr("paddings_contiguous"));
  std::vector<AnfNodePtr> concat_input_vec;

  // slice and insert to concat in reverse order
  if (paddings_contiguous) {
    for (int64_t i = 0; i < padding_src_length; i += static_cast<int64_t>(kSizeTwo)) {
      auto slice_node_2 = CreateStridedSliceNode(func_graph, input_node, i + kSizeOne);
      concat_input_vec.insert(concat_input_vec.begin(), slice_node_2);

      auto slice_node_1 = CreateStridedSliceNode(func_graph, input_node, i);
      concat_input_vec.insert(concat_input_vec.begin(), slice_node_1);
    }
  } else {
    for (int64_t i = 0; i < padding_src_length / 2; ++i) {
      auto slice_node_2 = CreateStridedSliceNode(func_graph, input_node, i + padding_src_length / 2);
      concat_input_vec.insert(concat_input_vec.begin(), slice_node_2);

      auto slice_node_1 = CreateStridedSliceNode(func_graph, input_node, i);
      concat_input_vec.insert(concat_input_vec.begin(), slice_node_1);
    }
    prim->AddAttr("paddings_contiguous", MakeValue(True));
  }

  if (padding_dst_length > padding_src_length) {
    auto input_paddings_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(pad_node, kIndex1);
    std::shared_ptr<tensor::Tensor> fill_tensor;
    if (input_paddings_type_id == kNumberTypeInt32) {
      fill_tensor =
        std::make_shared<tensor::Tensor>(std::vector<int32_t>(padding_dst_length - padding_src_length, 0), kInt32);
    } else if (input_paddings_type_id == kNumberTypeInt64) {
      fill_tensor =
        std::make_shared<tensor::Tensor>(std::vector<int64_t>(padding_dst_length - padding_src_length, 0), kInt64);
    } else {
      MS_LOG_EXCEPTION << "Unsupported data type for PadV3 padddings input.";
    }
    MS_EXCEPTION_IF_NULL(fill_tensor);
    auto fill_node = CreateValueNodeWithKernelInfo(func_graph, fill_tensor);
    MS_EXCEPTION_IF_NULL(fill_node);
    concat_input_vec.insert(concat_input_vec.begin(), fill_node);
  }
  static size_t concat_index = 0;
  auto concat_node =
    CreateConcatNode(func_graph, concat_input_vec,
                     pad_node->fullname_with_scope() + "_pad_slice_concat" + std::to_string(concat_index++));
  return concat_node;
}

const AnfNodePtr ConvertBasePaddings::CreateDynPaddingsPass(const FuncGraphPtr &graph, const CNodePtr &pad_node,
                                                            const bool &is_grad) const {
  // For dyn paddings in PadV3 and PadV3Grad on Ascend, add StridedSlice -> Concat to adjust paddings in ge::PadV3.
  auto input_x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(pad_node, kIndex0);
  size_t dst_length = input_x_shape.size() * 2;
  auto prim_name = "PadV3";
  if (is_grad) {
    prim_name = "PadV3Grad";
  }

  auto paddings = common::AnfAlgo::GetInputNode(pad_node, kIndex1);
  MS_EXCEPTION_IF_NULL(paddings);
  auto paddings_abstract = paddings->abstract();
  MS_EXCEPTION_IF_NULL(paddings_abstract);
  auto paddings_type = paddings_abstract->GetType();
  MS_EXCEPTION_IF_NULL(paddings_type);
  if (!paddings_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For " << prim_name
                            << ", the input `paddings` is required to be Tensor when it is dynamic.";
  }
  auto paddings_shape_ptr = paddings_abstract->GetShape();
  MS_EXCEPTION_IF_NULL(paddings_shape_ptr);
  auto paddings_shape = paddings_shape_ptr->GetShapeVector();
  (void)CheckAndConvertUtils::CheckInteger("paddings_shape_size", SizeToLong(paddings_shape.size()), kEqual, kDim1,
                                           prim_name);
  auto paddings_length = paddings_shape[0];
  // Not implemented: if is_grad and dst_length < 8, the filled paddings should be expanded to 8.
  auto concat_node = ProcessSliceNConcat(graph, pad_node, paddings, dst_length, paddings_length);
  MS_EXCEPTION_IF_NULL(concat_node);
  return concat_node;
}

template <typename T, TypeId type_id>
const AnfNodePtr ConvertBasePaddings::OptimizePaddingsValue(const FuncGraphPtr &graph,
                                                            const AbstractBasePtr &ori_paddings,
                                                            const bool &paddings_contiguous, const size_t &dst_length,
                                                            bool force_length8) const {
  std::vector<T> paddings_data;
  auto paddings_type = ori_paddings->GetType();
  MS_EXCEPTION_IF_NULL(paddings_type);
  if (paddings_type->template isa<TensorType>()) {
    auto paddings_value = ori_paddings->GetValue();
    MS_EXCEPTION_IF_NULL(paddings_value);
    auto paddings_array_value = ops::GetArrayValue<T>(paddings_value);
    paddings_data = paddings_array_value.value().ToVector();
  } else {
    auto paddings_value = ops::GetArrayValue<T>(ori_paddings);
    paddings_data = paddings_value->ToVector();
  }
  if (!paddings_contiguous) {
    auto tmp = paddings_data;
    for (size_t i = 0; i < paddings_data.size(); i++) {
      if (i % kStep2 == 0) {
        paddings_data[i] = tmp[i / kStep2];
      } else {
        paddings_data[i] = tmp[(i + paddings_data.size()) / kStep2];
      }
    }
  }
  // (0, 1, 2, 3, 4, 5, 6, 7) -> (6, 7, 4, 5, 2, 3, 0, 1)
  std::reverse(paddings_data.begin(), paddings_data.end());
  for (size_t i = 1; i < paddings_data.size(); i += kStep2) {
    std::swap(paddings_data[i - 1], paddings_data[i]);
  }
  // (1, 2, 3, 4) -> (0, 0, 0, 0, 1, 2, 3, 4)
  std::vector<T> opt_paddings_data(dst_length);
  auto offset = opt_paddings_data.size() - paddings_data.size();
  std::transform(paddings_data.begin(), paddings_data.end(), opt_paddings_data.begin() + offset,
                 [](const T &val) { return val; });
  // For ge::PadV3Grad, the length of paddings is required to be 8
  if (force_length8 && dst_length <= kLength8) {
    for (size_t i = 0; i < kLength8 - dst_length; i++) {
      opt_paddings_data.push_back(0);
    }
  }
  if (!paddings_contiguous) {
    auto opt_paddings_size = opt_paddings_data.size();
    std::vector<T> tmp_l;
    std::vector<T> tmp_r;
    for (size_t i = 0; i < opt_paddings_size; i++) {
      if (i % kStep2 == 0) {
        tmp_l.template emplace_back(opt_paddings_data[i]);
      } else {
        tmp_r.template emplace_back(opt_paddings_data[i]);
      }
    }
    opt_paddings_data.clear();
    std::transform(tmp_l.begin(), tmp_l.end(), std::back_inserter(opt_paddings_data), [](const T &val) { return val; });
    std::transform(tmp_r.begin(), tmp_r.end(), std::back_inserter(opt_paddings_data), [](const T &val) { return val; });
  }
  // Create ValueNode
  auto extend_paddings = CreateValueNodeWithKernelInfo(graph, MakeValue(opt_paddings_data));
  return extend_paddings;
}

const AnfNodePtr ConvertBasePaddings::CreateConstPaddingsNode(const FuncGraphPtr &graph,
                                                              const CNodePtr &pad_node) const {
  auto prim = GetCNodePrimitive(pad_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto paddings_contiguous = GetValue<bool>(prim->GetAttr("paddings_contiguous"));
  // ge::padV3 only support that the length of `paddings` is twice than the rank of `x`
  auto input_paddings = common::AnfAlgo::GetInputNode(pad_node, kIndex1);
  MS_EXCEPTION_IF_NULL(input_paddings);
  auto paddings_abstract = input_paddings->abstract();
  MS_EXCEPTION_IF_NULL(paddings_abstract);

  auto input_x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(pad_node, kIndex0);
  auto input_paddings_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(pad_node, kIndex1);
  auto paddings_value_node = CreateConstPaddingsPass(graph, paddings_abstract, paddings_contiguous,
                                                     input_x_shape.size() * 2, input_paddings_type_id);
  MS_EXCEPTION_IF_NULL(paddings_value_node);
  return paddings_value_node;
}

const AnfNodePtr ConvertBasePaddings::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto input_x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, kIndex0);
  auto padding_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, kIndex1);
  if (IsDynamicRank(input_x_shape) || IsDynamic(padding_shape)) {
    MS_LOG_EXCEPTION << "The input is dynamic rank";
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (HasDynPaddings(cnode)) {
    auto concat_node = CreateDynPaddingsNode(graph, cnode);
    MS_EXCEPTION_IF_NULL(concat_node);
    auto node_prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(node_prim);
    node_prim->AddAttr("is_dyn_paddings", MakeValue(true));
    common::AnfAlgo::SetNodeInput(cnode, concat_node, kIndex1);
  } else {
    auto paddings_value_node = CreateConstPaddingsNode(graph, cnode);
    MS_EXCEPTION_IF_NULL(paddings_value_node);
    common::AnfAlgo::SetNodeInput(cnode, paddings_value_node, kIndex1);
  }
  // Not verified: for PadV3Grad, if the input tensor rand < 4, the input should be expanded to 4.
  auto is_expand = ExpandInputXDims(graph, cnode);
  if (is_expand) {
    ReduceOutputDims(graph, cnode);
  }
  return node;
}

const AnfNodePtr ConvertPadV3GradPaddings::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (HasDynPaddings(cnode)) {
    MS_EXCEPTION(RuntimeError) << "PadV3Grad doesn't support dynamic paddings input.";
  }
  return ConvertBasePaddings::Process(graph, node, equiv);
}

bool ConvertPadV3GradPaddings::ExpandInputXDims(const FuncGraphPtr &graph, const CNodePtr &node) const {
  auto input_x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, kIndex0);
  auto input_x_rank = input_x_shape.size();
  // For ge::PadV3Grad, input x must be no less than 4-dimension
  if (input_x_rank >= 4) {
    return false;
  }
  // Expand shape to 4 dimensions
  auto new_shape = input_x_shape;
  for (size_t i = 0; i < kDim4 - input_x_rank; i++) {
    (void)new_shape.emplace_back(1);
  }
  // Replace the x with Reshape
  auto input_x_node = common::AnfAlgo::GetInputNode(node, kIndex0);
  MS_EXCEPTION_IF_NULL(input_x_node);
  auto reshape_node = CreateReshapeNode(graph, input_x_node, new_shape);
  MS_EXCEPTION_IF_NULL(reshape_node);
  common::AnfAlgo::SetNodeInput(node, reshape_node, kIndex0);
  return true;
}

void ConvertPadV3GradPaddings::ReduceOutputDims(const FuncGraphPtr &graph, const CNodePtr &node) const {
  auto output_shape = common::AnfAlgo::GetOutputInferShape(node, kIndex0);
  auto reshape_node = CreateReshapeNode(graph, node, output_shape);
  MS_EXCEPTION_IF_NULL(reshape_node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(node, reshape_node);
}

const BaseRef ConvertPadV3Paddings::DefinePattern() const {
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPadV3, inputs});
}

const BaseRef ConvertPadV3GradPaddings::DefinePattern() const {
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPadV3Grad, inputs});
}
}  // namespace opt
}  // namespace mindspore
