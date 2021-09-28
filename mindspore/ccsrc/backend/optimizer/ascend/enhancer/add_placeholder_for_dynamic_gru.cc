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

#include "backend/optimizer/ascend/enhancer/add_placeholder_for_dynamic_gru.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "abstract/abstract_value.h"
#include "base/core_ops.h"

namespace mindspore {
namespace opt {
const BaseRef InsertPlaceholderForDynamicGRUV2::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr InsertPlaceholderForDynamicGRUV2::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  if (op_name != kDynamicGRUV2OpName) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  size_t input_num = AnfAlgo::GetInputTensorNum(node);
  if (input_num == 0) {
    return nullptr;
  }

  std::vector<AnfNodePtr> new_inputs = {AnfAlgo::GetCNodePrimitiveNode(cnode)};
  auto none_index = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, "placeholder_index");
  size_t real_input_index = 0;
  for (size_t in_idx = 0; in_idx < input_num + none_index.size(); in_idx++) {
    auto item = find(none_index.begin(), none_index.end(), in_idx);
    if (item != none_index.end()) {
      auto value = std::make_shared<None>();
      auto value_node = NewValueNode(value);
      MS_EXCEPTION_IF_NULL(value_node);
      value_node->set_abstract(std::make_shared<abstract::AbstractNone>());
      auto new_node = kernel_graph->NewValueNode(value_node);
      kernel_graph->AddValueNodeToGraph(new_node);
      new_inputs.push_back(new_node);
    } else {
      auto input_node = AnfAlgo::GetInputNode(cnode, real_input_index);
      new_inputs.push_back(input_node);
      real_input_index++;
    }
  }

  CNodePtr new_node = nullptr;
  if (kernel_graph == nullptr) {
    new_node = std::make_shared<CNode>(*cnode);
  } else {
    new_node = kernel_graph->NewCNode(cnode);
  }
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_inputs(new_inputs);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
