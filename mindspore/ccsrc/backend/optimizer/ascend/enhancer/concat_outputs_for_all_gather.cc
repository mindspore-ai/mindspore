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

#include "backend/optimizer/ascend/enhancer/concat_outputs_for_all_gather.h"
#include <string>
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
AnfNodePtr ConcatOutputsForAllGather::InsertConcatForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                            const std::vector<AnfNodePtr> &new_tuple_getitems,
                                                            int64_t rank_size) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  size_t inputs_size = AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < inputs_size; ++i) {
    std::vector<AnfNodePtr> concat_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
    for (size_t j = 0, idx = i; j < IntToSize(rank_size); ++j, idx += inputs_size) {
      concat_inputs.push_back(new_tuple_getitems[idx]);
    }
    auto concat = func_graph->NewCNode(concat_inputs);
    MS_EXCEPTION_IF_NULL(concat);
    MS_EXCEPTION_IF_NULL(new_tuple_getitems[i]);
    auto dtypes = {AnfAlgo::GetOutputInferDataType(new_tuple_getitems[i], 0)};
    std::vector<size_t> shape = AnfAlgo::GetOutputInferShape(new_tuple_getitems[i], 0);
    shape[0] *= rank_size;
    std::vector<std::vector<size_t>> shapes = {shape};
    AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, concat.get());
    AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(static_cast<int64_t>(0)), concat);
    AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(rank_size), concat);
    std::vector<int64_t> dyn_input_size{rank_size};
    AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size), concat);
    kernel_select_->SelectKernel(concat);
    make_tuple_inputs.push_back(concat);
  }

  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}

const BaseRef ConcatOutputsForAllGather::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kAllGatherOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr ConcatOutputsForAllGather::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::HasNodeAttr(kAttrFusion, cnode) || !AnfAlgo::HasNodeAttr(kAttrRankSize, cnode)) {
    return nullptr;
  }
  auto fusion = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
  if (fusion <= 0) {
    return nullptr;
  }
  if (AnfAlgo::HasNodeAttr("fused", cnode) || AnfAlgo::GetInputTensorNum(node) == 1) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr("fused", MakeValue(true), node);
  auto rank_size = AnfAlgo::GetNodeAttr<int64_t>(node, kAttrRankSize);
  std::vector<AnfNodePtr> new_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, node, AnfAlgo::GetOutputTensorNum(node), &new_outputs);
  return InsertConcatForOutput(func_graph, node, new_outputs, rank_size);
}
}  // namespace opt
}  // namespace mindspore
