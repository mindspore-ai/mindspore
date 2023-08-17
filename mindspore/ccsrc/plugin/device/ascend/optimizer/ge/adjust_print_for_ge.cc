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

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kIndexOne = 1;
constexpr size_t kInputSizeTwo = 2;
}  // namespace

const BaseRef AdjustPrintForGe::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPrint, Xs});
}

// replace print(i1, i2, U) with 1. print(<i1, i2, U>) 2. depend(0.0, print)
const AnfNodePtr AdjustPrintForGe::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const std::vector<AnfNodePtr> &inputs = cnode->inputs();
  if (inputs.size() <= kInputSizeTwo) {
    return nullptr;
  }
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(std::make_shared<Primitive>(kMakeTupleOpName))};
  make_tuple_inputs.insert(make_tuple_inputs.end(), inputs.begin() + kIndexOne, inputs.end());
  auto make_tuple_node = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  std::vector<AbstractBasePtr> abstract_list;
  for (size_t input_index = kIndexOne; input_index < inputs.size(); ++input_index) {
    auto input_node = inputs.at(input_index);
    MS_EXCEPTION_IF_NULL(input_node);
    (void)abstract_list.emplace_back(input_node->abstract());
  }
  make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  std::vector<AnfNodePtr> new_print_inputs{NewValueNode(std::make_shared<Primitive>(kPrintOpName))};
  (void)new_print_inputs.emplace_back(make_tuple_node);
  auto new_print_node = func_graph->NewCNode(new_print_inputs);
  MS_EXCEPTION_IF_NULL(new_print_node);
  new_print_node->set_abstract(node->abstract());

  auto tensor = std::make_shared<tensor::Tensor>(0.0);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  ValueNodePtr value_node = kernel_graph->NewValueNode(tensor->ToAbstract(), tensor);
  kernel_graph->AddValueNodeToGraph(value_node);
  std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(kDependOpName)), value_node,
                                          new_print_node};
  auto new_depend_node = func_graph->NewCNode(depend_input);
  MS_EXCEPTION_IF_NULL(new_depend_node);
  new_depend_node->set_abstract(value_node->abstract());
  return new_depend_node;
}
}  // namespace opt
}  // namespace mindspore
