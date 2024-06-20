/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ge/adjust_print_for_ge.h"

#include <algorithm>
#include <memory>
#include <vector>
#include "ops/framework_ops.h"
#include "ops/sequence_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kIndexOne = 1;
constexpr size_t kInputSizeTwo = 2;

bool PrintUnvisited(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    auto node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitive(node, prim::kPrimPrint)) {
      return false;
    }
    return UnVisited(ref);
  }
  return false;
}

ValueNodePtr CreateValueNode(const ValuePtr &value_ptr, TypeId output_type, bool is_scalar = false) {
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto new_node = std::make_shared<ValueNode>(value_ptr);
  MS_EXCEPTION_IF_NULL(new_node);
  auto value_abstract = value_ptr->ToAbstract();
  new_node->set_abstract(value_abstract);

  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  new_node->set_kernel_info(kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsDeviceType({output_type});
  if (is_scalar) {
    builder1.SetOutputsKernelObjectType({kernel::KernelObjectType::SCALAR});
  } else {
    builder1.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  }
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), new_node.get());
  return new_node;
}
}  // namespace

const BaseRef AdjustPrintForGe::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(PrintUnvisited);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

void AdjustPrintForGe::UnfoldMakeTuple(const AnfNodePtr &input, int64_t *ptr_num_inputs,
                                       std::vector<AnfNodePtr> *ptr_new_inputs) const {
  if (!IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
    ptr_new_inputs->push_back(input);
    (*ptr_num_inputs)++;
    return;
  }

  auto input_cnode = input->cast<CNodePtr>();
  std::vector<AnfNodePtr> tuple_inputs = input_cnode->inputs();
  for (size_t tuple_inputs_index = 1; tuple_inputs_index < tuple_inputs.size(); ++tuple_inputs_index) {
    auto &tuple_input_node = tuple_inputs[tuple_inputs_index];
    MS_EXCEPTION_IF_NULL(tuple_input_node);
    UnfoldMakeTuple(tuple_input_node, ptr_num_inputs, ptr_new_inputs);
  }
}

// replace print(i1, i2, U) with print(dummy_input, i1, i2, U) and set attributes of print
const AnfNodePtr AdjustPrintForGe::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);

  // create a dummy input with string value "print" for input 'tensor_name' of ge operator 'OutfeedEnqueueOpV2'
  const auto tensor_name = "print";
  auto input_tensor_name = CreateValueNode(std::make_shared<StringImm>(tensor_name), kObjectTypeString, true);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddValueNodeToGraph(input_tensor_name);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // NOTE: input(0) of cnode is value node of Primitive, the last input of Print is a Monad node
  int64_t num_inputs = 0;
  std::vector<AnfNodePtr> new_inputs;
  new_inputs.push_back(cnode->input(0));
  new_inputs.push_back(input_tensor_name);
  // unfold tuple inputs
  std::vector<AnfNodePtr> node_inputs = cnode->inputs();
  for (size_t node_inputs_index = 1; node_inputs_index < node_inputs.size(); ++node_inputs_index) {
    auto &input = node_inputs[node_inputs_index];
    MS_EXCEPTION_IF_NULL(input);
    if (IsValueNode<UMonad>(input) || IsValueNode<IOMonad>(input) || HasAbstractMonad(input)) {
      continue;
    }
    UnfoldMakeTuple(input, &num_inputs, &new_inputs);
  }
  new_inputs.push_back(node_inputs.back());
  auto new_print_node = func_graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_print_node);
  new_print_node->set_abstract(node->abstract());
  new_print_node->set_scope(node->scope());
  new_print_node->set_fullname_with_scope(node->fullname_with_scope());
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), new_print_node);

  // set attribute channel_name and dynamic_input_sizes of print node
  auto primitive = GetCNodePrimitive(new_print_node);
  MS_EXCEPTION_IF_NULL(primitive);
  (void)primitive->AddAttr(kAttrChannelName, MakeValue(kChannelNameNpuLog));
  (void)primitive->AddAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{-1, num_inputs, -1}));

  // add depend node for print
  auto tensor = std::make_shared<tensor::Tensor>(0.0);
  ValueNodePtr value_node = kernel_graph->NewValueNode(tensor->ToAbstract(), tensor);
  kernel_graph->AddValueNodeToGraph(value_node);
  std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(kDependOpName)), value_node,
                                          new_print_node};
  auto new_depend_node = func_graph->NewCNode(depend_input);
  MS_EXCEPTION_IF_NULL(new_depend_node);
  new_depend_node->set_abstract(value_node->abstract());
  new_depend_node->set_scope(node->scope());

  return new_depend_node;
}
}  // namespace opt
}  // namespace mindspore
