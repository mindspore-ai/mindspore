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
namespace {
void AddOutputs(const AnfNodePtr &node, int rank_size) {
  MS_EXCEPTION_IF_NULL(node);
  auto origin_abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(origin_abstract);
  auto tuple_abstract = origin_abstract->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  auto &origin_abstracts = tuple_abstract->elements();
  AbstractBasePtrList abstract_list;
  std::vector<TypeId> outputs_device_type;
  std::vector<std::string> outputs_device_format;
  for (int i = 0; i < rank_size; ++i) {
    for (size_t j = 0; j < origin_abstracts.size(); ++j) {
      abstract_list.push_back(origin_abstracts[j]);
      outputs_device_type.push_back(AnfAlgo::GetOutputDeviceDataType(node, j));
      outputs_device_format.push_back(AnfAlgo::GetOutputFormat(node, j));
    }
  }
  // Update abstract
  auto new_abstracts = std::make_shared<abstract::AbstractTuple>(abstract_list);
  node->set_abstract(new_abstracts);
  // Update kernel build info
  auto builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
  builder->SetOutputsDeviceType(outputs_device_type);
  builder->SetOutputsFormat(outputs_device_format);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
}
}  // namespace

AnfNodePtr ConcatOutputsForAllGather::InsertConcatForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                            const std::vector<AnfNodePtr> &new_tuple_getitems,
                                                            int rank_size) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> make_tuple_inputs;
  size_t inputs_size = AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < inputs_size; ++i) {
    for (size_t j = 0, idx = i; j < IntToSize(rank_size); ++j, idx += inputs_size) {
      std::vector<AnfNodePtr> concat_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
      concat_inputs.push_back(new_tuple_getitems[idx]);
      auto concat = func_graph->NewCNode(concat_inputs);
      MS_EXCEPTION_IF_NULL(concat);
      MS_EXCEPTION_IF_NULL(new_tuple_getitems[idx]);
      concat->set_abstract(new_tuple_getitems[idx]->abstract());
      AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(0), concat);
      AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(rank_size), concat);
      std::vector<int> dyn_input_size{rank_size};
      AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size), concat);
      kernel_select_->SelectKernel(concat);
      make_tuple_inputs.push_back(concat);
    }
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
  auto fusion = AnfAlgo::GetNodeAttr<int>(cnode, kAttrFusion);
  if (fusion <= 0) {
    return nullptr;
  }
  auto rank_size = AnfAlgo::GetNodeAttr<int>(node, kAttrRankSize);
  AddOutputs(node, rank_size);
  std::vector<AnfNodePtr> new_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, node, AnfAlgo::GetOutputTensorNum(node), &new_outputs);
  return InsertConcatForOutput(func_graph, node, new_outputs, rank_size);
}
}  // namespace opt
}  // namespace mindspore
