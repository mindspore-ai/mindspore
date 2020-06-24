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
#include "pre_activate/pass/convert_tuple_input_to_dynamic_input.h"

#include <algorithm>
#include <memory>

#include "session/anf_runtime_algorithm.h"
#include "pre_activate/common/helper.h"
#include "session/kernel_graph.h"
#include "kernel/common_utils.h"
#include "device/kernel_info.h"

namespace mindspore {
namespace opt {
namespace {
bool MakeValueNode(const AnfNodePtr &node) {
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return false;
  }

  // create kernel_info fo new value node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  // set the format of value_node to DEFAULT_FORMAT
  kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
  // set value node initial device data type = infer data type
  TypeId infer_data_type;
  if (AnfAlgo::GetOutputTensorNum(value_node) == 0) {
    infer_data_type = kTypeUnknown;
  } else {
    infer_data_type = AnfAlgo::GetOutputInferDataType(value_node, 0);
  }
  kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{infer_data_type});
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), value_node.get());
  return true;
}

void ConvertTupleOuputToPlantInputs(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
                                    std::vector<AnfNodePtr> *plant_inputs, std::vector<int> *dyn_input_sizes) {
  MS_EXCEPTION_IF_NULL(plant_inputs);
  MS_EXCEPTION_IF_NULL(dyn_input_sizes);
  MS_EXCEPTION_IF_NULL(graph);
  auto output_size = AnfAlgo::GetOutputTensorNum(input_node);
  dyn_input_sizes->push_back(output_size);
  std::vector<AnfNodePtr> convert_inputs;
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (input_node->isa<ValueNode>()) {
    auto value_node = input_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    convert_inputs = kernel_graph->SplitTupleValueNodeToNodeList(value_node);
  } else {
    for (size_t index = 0; index < output_size; ++index) {
      auto tuple_get_item = CreatTupleGetItemNode(graph, input_node, index);
      AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(input_node, index)},
                                          {AnfAlgo::GetOutputInferShape(input_node, index)}, tuple_get_item.get());
      convert_inputs.emplace_back(tuple_get_item);
    }
  }
  (void)std::copy(convert_inputs.begin(), convert_inputs.end(), std::back_inserter(*plant_inputs));
}

void ConvertMakeTupleInputToPlantInputs(const FuncGraphPtr &graph, const CNodePtr &cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  MS_EXCEPTION_IF_NULL(graph);
  auto &ori_args = cnode_ptr->inputs();
  if (ori_args.size() < 1) {
    return;
  }
  std::vector<AnfNodePtr> plant_inputs;
  std::vector<int> dyn_input_sizes;
  plant_inputs.push_back(ori_args[kAnfPrimitiveIndex]);
  for (size_t i = 1; i < ori_args.size(); ++i) {
    auto input_node = ori_args[i];
    if (IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
      auto input_size = AnfAlgo::GetOutputTensorNum(input_node);
      dyn_input_sizes.push_back(input_size);
      auto cnode = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto inputs = cnode->inputs();
      for (size_t j = 1; j < inputs.size(); ++j) {
        MS_EXCEPTION_IF_NULL(inputs[j]);
        if (IsValueNode<tensor::Tensor>(inputs[j])) {
          auto success = MakeValueNode(inputs[j]);
          if (!success) {
            MS_LOG(WARNING) << "Make value node failed, " << inputs[j]->DebugString();
          }
        }
        plant_inputs.push_back(inputs[j]);
      }
    } else if (input_node->Type() != nullptr && AnfAlgo::IsTupleOutput(input_node)) {
      ConvertTupleOuputToPlantInputs(graph, input_node, &plant_inputs, &dyn_input_sizes);
    } else {
      dyn_input_sizes.push_back(-1);
      plant_inputs.push_back(input_node);
    }
  }
  // If there is dynamic input, set the dyn_input_sizes as an attribute and update the inputs.
  if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int s) { return s >= 0; })) {
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
  } else {
    ConvertMakeTupleInputToPlantInputs(func_graph, node->cast<CNodePtr>());
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
