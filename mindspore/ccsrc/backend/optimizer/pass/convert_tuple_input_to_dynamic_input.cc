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
#include "backend/optimizer/pass/convert_tuple_input_to_dynamic_input.h"

#include <algorithm>
#include <memory>

#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/kernel_graph.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/kernel_info.h"

namespace mindspore {
namespace opt {
namespace {
int64_t SplitTupleInputs(const FuncGraphPtr &graph, const AnfNodePtr &tuple_input,
                         std::vector<AnfNodePtr> *plant_inputs) {
  if (!AnfAlgo::IsTupleOutput(tuple_input)) {
    auto abs = tuple_input->abstract();
    MS_LOG(WARNING) << "The Function only split the output type is tuple type but got" << abs->ToString();
    return -1;
  }
  MS_EXCEPTION_IF_NULL(plant_inputs);
  auto input_size = AnfAlgo::GetOutputTensorNum(tuple_input);
  if (tuple_input->isa<CNode>() && AnfAlgo::CheckPrimitiveType(tuple_input, prim::kPrimMakeTuple)) {
    auto make_tuple = tuple_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    size_t tuple_input_num = AnfAlgo::GetInputTensorNum(make_tuple);
    for (size_t j = 0; j < tuple_input_num; ++j) {
      // using for graph kernel
      auto dyn_input_node = AnfAlgo::GetInputNode(make_tuple, j);
      MS_EXCEPTION_IF_NULL(dyn_input_node);
      plant_inputs->emplace_back(dyn_input_node);
    }
    return input_size;
  }
  for (size_t index = 0; index < input_size; ++index) {
    auto dyn_input_node = CreatTupleGetItemNode(graph, tuple_input, index);
    plant_inputs->emplace_back(dyn_input_node);
  }
  return input_size;
}

void ConvertMakeTupleInputToPlantInputs(const FuncGraphPtr &graph, const CNodePtr &cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  MS_EXCEPTION_IF_NULL(graph);
  if (AnfAlgo::CheckPrimitiveType(cnode_ptr, prim::kPrimCall) ||
      AnfAlgo::CheckPrimitiveType(cnode_ptr, prim::kPrimPartial)) {
    return;
  }
  std::vector<AnfNodePtr> plant_inputs;
  std::vector<int64_t> dyn_input_sizes;
  plant_inputs.push_back(AnfAlgo::GetCNodePrimitiveNode(cnode_ptr));
  size_t input_num = AnfAlgo::GetInputTensorNum(cnode_ptr);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node = AnfAlgo::GetInputNode(cnode_ptr, i);
    MS_EXCEPTION_IF_NULL(input_node);
    if (AnfAlgo::IsTupleOutput(input_node)) {
      dyn_input_sizes.emplace_back(SplitTupleInputs(graph, input_node, &plant_inputs));
    } else {
      dyn_input_sizes.push_back(-1);
      plant_inputs.push_back(input_node);
    }
  }
  // If there is dynamic input, set the dyn_input_sizes as an attribute and update the inputs.
  if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int64_t s) { return s >= 0; })) {
    AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), cnode_ptr);
    cnode_ptr->set_inputs(plant_inputs);
  }
}
}  // namespace

const BaseRef ConvertTupleInputToDynamicInput::DefinePattern() const {
  VarPtr V = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr ConvertTupleInputToDynamicInput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                          const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfAlgo::IsRealKernel(node)) {
    return nullptr;
  }
  if (AnfAlgo::IsGraphKernel(node)) {
    auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(sub_graph);
    std::vector<AnfNodePtr> todos;
    kernel::GetValidKernelNodes(sub_graph, &todos);
    for (auto &t : todos) {
      ConvertMakeTupleInputToPlantInputs(sub_graph, t->cast<CNodePtr>());
    }
  }
  ConvertMakeTupleInputToPlantInputs(func_graph, node->cast<CNodePtr>());
  return node;
}
}  // namespace opt
}  // namespace mindspore
