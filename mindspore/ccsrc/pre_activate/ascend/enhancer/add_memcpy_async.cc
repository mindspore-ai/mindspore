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
#include "pre_activate/ascend/enhancer/add_memcpy_async.h"
#include <vector>
#include "utils/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "optimizer/opt.h"
#include "pre_activate/ascend/ascend_helper.h"

namespace mindspore {
namespace opt {
namespace {
bool InputIsParameterOrValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
  return kernel_with_index.first->isa<Parameter>() || kernel_with_index.first->isa<ValueNode>();
}

const AnfNodePtr AddMemcpyAsyncIfInputIsUsedByOthers(const FuncGraphPtr &graph, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const std::vector<AnfNodePtr> &inputs = node->inputs();
  bool replace = false;
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "node[" + AnfAlgo::GetCNodeName(node) + "]'s inputs is empty";
  }
  std::vector<AnfNodePtr> new_inputs = {inputs[0]};
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto input = node->input(i);
    if (manager->node_users().find(input) == manager->node_users().end()) {
      MS_LOG(EXCEPTION) << "node has no output in manager";
    }
    // when input is used by others or is a parameter or is a value node, insert a memcpy_async
    if (manager->node_users()[input].size() > 1 || InputIsParameterOrValueNode(input)) {
      replace = true;
      new_inputs.push_back(CreateMemcpyAsyncOp(graph, input));
    } else {
      new_inputs.push_back(input);
    }
  }

  CNodePtr new_node = std::make_shared<CNode>(*node);
  new_node->set_inputs(new_inputs);
  return replace ? new_node : nullptr;
}
}  // namespace

const AnfNodePtr AddMemcpyAsync::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                         const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!AnfAlgo::IsCommunicationOp(node)) {
    return nullptr;
  }
  return AddMemcpyAsyncIfInputIsUsedByOthers(func_graph, cnode);
}
}  // namespace opt
}  // namespace mindspore
