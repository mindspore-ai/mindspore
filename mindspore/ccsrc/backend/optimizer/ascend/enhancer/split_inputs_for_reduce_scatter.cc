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

#include "backend/optimizer/ascend/enhancer/split_inputs_for_reduce_scatter.h"
#include <string>
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
std::vector<AnfNodePtr> SplitInputsForReduceScatter::InsertSplitForInput(const FuncGraphPtr &func_graph,
                                                                         const CNodePtr &node,
                                                                         int64_t rank_size) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  size_t inputs_size = AnfAlgo::GetInputTensorNum(node);
  std::vector<AnfNodePtr> split_outputs;
  for (size_t i = 0; i < inputs_size; i++) {
    std::vector<AnfNodePtr> split_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name()))};
    split_inputs.push_back(AnfAlgo::GetInputNode(node, i));
    auto split = func_graph->NewCNode(split_inputs);
    MS_EXCEPTION_IF_NULL(split);
    std::vector<TypeId> dtypes(rank_size, AnfAlgo::GetPrevNodeOutputInferDataType(node, i));
    std::vector<std::vector<size_t>> shapes;
    std::vector<int> size_splits;
    for (size_t j = 0; j < IntToSize(rank_size); j++) {
      std::vector<size_t> output_node_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, i);
      output_node_shape[0] /= rank_size;
      shapes.push_back(output_node_shape);
      size_splits.push_back(output_node_shape[0]);
    }
    AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());

    AnfAlgo::SetNodeAttr("split_dim", MakeValue(0L), split);
    AnfAlgo::SetNodeAttr("num_split", MakeValue(SizeToInt(rank_size)), split);
    AnfAlgo::SetNodeAttr("size_splits", MakeValue(size_splits), split);
    kernel_select_->SelectKernel(split);
    std::vector<AnfNodePtr> new_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, split, AnfAlgo::GetOutputTensorNum(split), &new_outputs);
    for (size_t j = 0; j < new_outputs.size(); j++) {
      split_outputs.push_back(new_outputs[j]);
    }
  }
  return split_outputs;
}

AnfNodePtr SplitInputsForReduceScatter::RearrangeInputsForReduceScatter(const FuncGraphPtr &func_graph,
                                                                        const AnfNodePtr &node,
                                                                        const std::vector<AnfNodePtr> &inputs,
                                                                        int64_t rank_size) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  size_t inputs_size = AnfAlgo::GetInputTensorNum(node);
  std::vector<AnfNodePtr> reduce_scatter_inputs{
    NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceScatter->name()))};
  for (size_t i = 0; i < IntToSize(rank_size); i++) {
    for (size_t j = 0, idx = i; j < inputs_size; j++, idx += IntToSize(rank_size)) {
      reduce_scatter_inputs.push_back(inputs[idx]);
    }
  }
  auto reduce_scatter = func_graph->NewCNode(reduce_scatter_inputs);
  MS_EXCEPTION_IF_NULL(reduce_scatter);
  reduce_scatter->set_abstract(node->abstract());

  AnfAlgo::CopyNodeAttrs(node, reduce_scatter);
  AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(1L), reduce_scatter);
  kernel_select_->SelectKernel(reduce_scatter);
  return reduce_scatter;
}

const BaseRef SplitInputsForReduceScatter::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kReduceScatterOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr SplitInputsForReduceScatter::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (AnfAlgo::GetInputTensorNum(node) == 1) {
    AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(0L), node);
    return nullptr;
  }
  if (!AnfAlgo::HasNodeAttr(kAttrFusion, cnode) || !AnfAlgo::HasNodeAttr(kAttrRankSize, cnode)) {
    return nullptr;
  }
  auto fusion = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
  if (fusion <= 0) {
    return nullptr;
  }
  if (AnfAlgo::HasNodeAttr("Fused", cnode)) {
    return nullptr;
  }

  AnfAlgo::SetNodeAttr("Fused", MakeValue(true), node);
  auto rank_size = AnfAlgo::GetNodeAttr<int64_t>(node, kAttrRankSize);
  std::vector<AnfNodePtr> split_outputs = InsertSplitForInput(func_graph, cnode, rank_size);
  return RearrangeInputsForReduceScatter(func_graph, node, split_outputs, rank_size);
}
}  // namespace opt
}  // namespace mindspore
