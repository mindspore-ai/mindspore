/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/concat_outputs_for_all_gather.h"
#include <utility>
#include <algorithm>
#include "primitive/other_op_name.h"
#include "primitive/sequence_ops.h"
#include "primitive/array_ops.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::opt {
namespace {
constexpr auto kPatternOpaque = "Opaque";
OutputInfo GetNodeOutputInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<TypeId> output_infer_dtype;
  std::vector<ShapeVector> output_infer_shape;
  auto type_ptr = node->Type();
  size_t output_num = AnfUtils::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; i++) {
    (void)output_infer_dtype.emplace_back(common::AnfAlgo::GetOutputInferDataType(type_ptr, i));
    (void)output_infer_shape.emplace_back(common::AnfAlgo::GetOutputInferShape(node, i));
  }
  return {output_infer_dtype, output_infer_shape};
}

OutputInfo GetNodesOutputInfo(const std::vector<AnfNodePtr> &nodes) {
  std::vector<TypeId> output_infer_dtype;
  std::vector<ShapeVector> output_infer_shape;
  for (const auto &node : nodes) {
    (void)output_infer_dtype.emplace_back(common::AnfAlgo::GetOutputInferDataType(node, 0));
    (void)output_infer_shape.emplace_back(common::AnfAlgo::GetOutputInferShape(node, 0));
  }

  return {output_infer_dtype, output_infer_shape};
}

void GetInputNodeInfoForOneRank(const OutputInfo &output_info, const std::vector<AnfNodePtr> &new_tuple_getitems,
                                size_t index, size_t input_size, int64_t rank_size, OutputInfo *concat_input_info,
                                std::vector<AnfNodePtr> *concat_input_nodes) {
  const auto &output_infer_dtype = std::get<kIndex0>(output_info);
  const auto &output_infer_shape = std::get<kIndex1>(output_info);
  auto &input_infer_dtype = std::get<kIndex0>(*concat_input_info);
  auto &input_infer_shape = std::get<kIndex1>(*concat_input_info);
  for (size_t j = 0; j < LongToSize(rank_size); ++j, index += input_size) {
    concat_input_nodes->push_back(new_tuple_getitems[index]);
    input_infer_dtype.push_back(output_infer_dtype[index]);
    input_infer_shape.push_back(output_infer_shape[index]);
  }
}
}  // namespace

CNodePtr ConcatOutputsForAllGather::CreateNewConcat(const FuncGraphPtr &func_graph,
                                                    const std::vector<AnfNodePtr> &concat_input_nodes,
                                                    const OutputInfo &concat_input_info, size_t begin_index,
                                                    int64_t offset) const {
  std::vector<AnfNodePtr> concat_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
  size_t end_index = begin_index + static_cast<size_t>(offset);
  for (size_t i = begin_index; i < end_index; ++i) {
    concat_inputs.emplace_back(concat_input_nodes[i]);
  }

  // Add axis value as the last input of concat node.
  auto axis_abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(kIndex0), kInt64);
  ValueNodePtr axis_node = std::make_shared<ValueNode>(MakeValue<int64_t>(kIndex0));
  axis_node->set_abstract(axis_abs);
  axis_node = func_graph->cast<KernelGraphPtr>()->NewValueNode(axis_node);
  func_graph->cast<KernelGraphPtr>()->AddValueNodeToGraph(axis_node);
  concat_inputs.emplace_back(axis_node);

  CNodePtr concat = NewCNode(concat_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(concat);
  const std::vector<TypeId> &dtypes = {std::get<kIndex0>(concat_input_info)[0]};
  const auto &input_shapes = std::get<kIndex1>(concat_input_info);
  auto shape = input_shapes[begin_index];
  for (size_t i = begin_index + 1; i < end_index; ++i) {
    shape[0] += input_shapes[i][0];
  }

  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {shape}, concat.get());
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(static_cast<int64_t>(0)), concat);
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(offset), concat);
  std::vector<int64_t> dyn_input_size{offset, -1};
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size), concat);
  return concat;
}

AnfNodePtr ConcatOutputsForAllGather::InsertConcatForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                            const OutputInfo &output_info,
                                                            const std::vector<AnfNodePtr> &new_tuple_getitems,
                                                            int64_t rank_size) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  size_t inputs_size = common::AnfAlgo::GetInputTensorNum(node);

  for (size_t i = 0; i < inputs_size; ++i) {
    OutputInfo concat_input_info;
    std::vector<AnfNodePtr> concat_input_nodes;
    GetInputNodeInfoForOneRank(output_info, new_tuple_getitems, i, inputs_size, rank_size, &concat_input_info,
                               &concat_input_nodes);
    size_t concat_input_num = concat_input_nodes.size();
    AnfNodePtr concat = concat_input_nodes[0];
    while (concat_input_num > 1) {
      std::vector<AnfNodePtr> output_nodes;
      size_t cur_input_index = 0;
      while (concat_input_num - cur_input_index >= static_cast<size_t>(inputs_divisor_)) {
        concat = CreateNewConcat(func_graph, concat_input_nodes, concat_input_info, cur_input_index, inputs_divisor_);
        output_nodes.push_back(concat);
        cur_input_index += LongToSize(inputs_divisor_);
      }
      size_t rest_num = concat_input_num - cur_input_index;
      if (rest_num == 1) {
        output_nodes.push_back(concat_input_nodes.back());
      } else if (rest_num > 1) {
        concat = CreateNewConcat(func_graph, concat_input_nodes, concat_input_info, cur_input_index,
                                 static_cast<int64_t>(rest_num));
        output_nodes.push_back(concat);
      }

      concat_input_num = output_nodes.size();
      concat_input_nodes = output_nodes;
      concat_input_info = GetNodesOutputInfo(concat_input_nodes);
    }
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(node, i)) {
      kernel_graph->ReplaceInternalOutput(node, concat, i, 0);
    }
    make_tuple_inputs.push_back(concat);
  }

  auto make_tuple = NewCNode(make_tuple_inputs, func_graph);
  return make_tuple;
}

const BaseRef ConcatOutputsForAllGather::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kAllGatherOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr ConcatOutputsForAllGather::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode) || !common::AnfAlgo::HasNodeAttr(kAttrRankSize, cnode)) {
    return nullptr;
  }
  auto fusion = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
  if (fusion <= 0) {
    return nullptr;
  }
  if (common::AnfAlgo::HasNodeAttr("fused", cnode) || common::AnfAlgo::GetInputTensorNum(node) == 1) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr("fused", MakeValue(true), node);
  auto rank_size = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrRankSize);
  std::vector<AnfNodePtr> new_outputs;
  OutputInfo output_info = GetNodeOutputInfo(node);
  size_t output_num = AnfUtils::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; ++i) {
    auto tuple_getitem = CreatTupleGetItemNode(func_graph, node, i);
    common::AnfAlgo::SetOutputInferTypeAndShape({std::get<0>(output_info)[i]},
                                                {common::AnfAlgo::GetOutputInferShape(node, i)}, tuple_getitem.get());
    (void)new_outputs.emplace_back(std::move(tuple_getitem));
  }

  return InsertConcatForOutput(func_graph, node, output_info, new_outputs, rank_size);
}
}  // namespace mindspore::opt
