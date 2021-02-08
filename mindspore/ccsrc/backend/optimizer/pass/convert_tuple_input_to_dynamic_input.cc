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
    if (input_node->isa<CNode>() && AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
      auto input_size = AnfAlgo::GetOutputTensorNum(input_node);
      dyn_input_sizes.push_back(input_size);
      auto make_tuple = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(make_tuple);
      size_t tuple_input_num = AnfAlgo::GetInputTensorNum(make_tuple);
      for (size_t j = 0; j < tuple_input_num; ++j) {
        auto dyn_input_node = AnfAlgo::GetInputNode(make_tuple, j);
        MS_EXCEPTION_IF_NULL(dyn_input_node);
        if (IsValueNode<tensor::Tensor>(dyn_input_node)) {
          auto kernel_graph = graph->cast<KernelGraphPtr>();
          MS_EXCEPTION_IF_NULL(kernel_graph);
          auto success = kernel_graph->NewValueNode(dyn_input_node->cast<ValueNodePtr>());
          if (!success) {
            MS_LOG(WARNING) << "Make value node failed, " << dyn_input_node->DebugString();
          }
        }
        plant_inputs.push_back(dyn_input_node);
      }
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
