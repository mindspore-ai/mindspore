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

#include "pre_activate/ascend/enhancer/getnext_memcpy_elimination.h"
#include <memory>
#include "session/anf_runtime_algorithm.h"
#include "optimizer/opt.h"

namespace mindspore::opt {

const BaseRef GetnextMemcpyElimination::DefinePattern() const {
  auto prim_memcpy = std::make_shared<Primitive>(kMemCpyAsyncOpName);
  VarPtr x = std::make_shared<SeqVar>();
  VectorRef memcpy_async({prim_memcpy, x});
  return memcpy_async;
}

const AnfNodePtr GetnextMemcpyElimination::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  if (graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  auto memcpy_cnode = node->cast<CNodePtr>();
  if (memcpy_cnode == nullptr) {
    return nullptr;
  }

  // 1. memcpy has attr kAttrLabelForInsertStreamActive
  if (!AnfAlgo::HasNodeAttr(kAttrLabelForInsertStreamActive, memcpy_cnode)) {
    MS_LOG(DEBUG) << "node has no label_for_insert_stream_active attr";
    return nullptr;
  }

  // 2. memcpy's output has only one user next_node
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(memcpy_cnode) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "memcpy has no output in manager";
  }
  auto next_nodes = manager->node_users()[memcpy_cnode];
  if (next_nodes.size() > 1) {
    MS_LOG(DEBUG) << "node's output has more than one users";
    return nullptr;
  }

  // 3. next_node is not nop node and it has only one input which is memcpy's output
  for (auto &item : next_nodes) {
    auto next_node = item.first->cast<CNodePtr>();
    if (opt::IsNopNode(next_node)) {
      return nullptr;
    }
    if (next_node->inputs().size() != 2) {
      MS_LOG(DEBUG) << "next node has more than one input";
      return nullptr;
    }
    // add attr label_for_insert_stream_active for next_node
    AnfAlgo::SetNodeAttr(kAttrLabelForInsertStreamActive, MakeValue(true), next_node);
  }

  return memcpy_cnode->input(1);
}
}  // namespace mindspore::opt
