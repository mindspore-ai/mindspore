/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/enhancer/add_placeholder_for_dynamic_rnn.h"
#include <vector>
#include <memory>
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace opt {
constexpr size_t kInsertIdx = 3;
const BaseRef InsertPlaceholderForDynamicRNN::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr InsertPlaceholderForDynamicRNN::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  if (op_name != kDynamicRNNOpName) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  if (input_num == 0) {
    return nullptr;
  }

  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  std::vector<AnfNodePtr> new_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(cnode)};
  for (size_t in_idx = 0; in_idx < input_num; in_idx++) {
    auto input_node = common::AnfAlgo::GetInputNode(cnode, in_idx);
    if (in_idx == kInsertIdx) {
      auto value = std::make_shared<None>();
      auto value_node = NewValueNode(value);
      MS_EXCEPTION_IF_NULL(value_node);
      value_node->set_abstract(std::make_shared<abstract::AbstractNone>());
      ValueNodePtr new_vnode = nullptr;
      if (kernel_graph == nullptr) {
        new_vnode = value_node;
      } else {
        new_vnode = kernel_graph->NewValueNode(value_node);
      }
      new_inputs.push_back(new_vnode);
    }
    new_inputs.push_back(input_node);
  }

  CNodePtr new_node = nullptr;
  if (kernel_graph == nullptr) {
    new_node = std::make_shared<CNode>(*cnode);
  } else {
    new_node = NewCNode(cnode, kernel_graph);
  }
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_inputs(new_inputs);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
