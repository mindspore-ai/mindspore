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

#include "backend/common/expander/fallback/expander_fallback.h"
#include <algorithm>
#include <queue>
#include <map>
#include <memory>
#include "base/base.h"
#include "ops/op_name.h"
#include "utils/ms_utils.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "backend/common/expander/fallback/fallback_irbuilder.h"

namespace mindspore {
namespace expander {
bool Check(const AnfNodePtr &node) {
  if (common::GetEnv("MS_DEV_EXPANDER_FALLBACK") == "off") {
    return false;
  }
  if (!node->isa<CNode>()) {
    return false;
  }
  // Operators with 'batch_rank' attribute, which only appears in the vmap scenario, are not supported currently.
  if (common::AnfAlgo::HasNodeAttr(ops::kBatchRank, node->cast<CNodePtr>())) {
    return false;
  }
  return true;
}

void DumpGraph(const CNodePtr &ori_node, const CNodePtr &new_output) {
  auto expand_fg = std::make_shared<FuncGraph>();
  std::map<AnfNodePtr, AnfNodePtr> node_map;
  CNodePtrList newcnodes;
  for (size_t i = 1; i < ori_node->size(); i++) {
    auto p = expand_fg->add_parameter();
    p->set_abstract(ori_node->input(i)->abstract());
    node_map[ori_node->input(i)] = p;
  }
  std::queue<CNodePtr> que;
  que.push(new_output);
  while (!que.empty()) {
    auto node = que.front();
    que.pop();
    if (node_map.count(node) > 0) {
      continue;
    }
    auto new_node = expand_fg->NewCNode(node->inputs());
    new_node->CloneCNodeInfo(node);
    new_node->set_fullname_with_scope(node->fullname_with_scope());
    newcnodes.push_back(new_node);
    node_map[node] = new_node;
    for (size_t i = 1; i < node->size(); ++i) {
      const auto &inp = node->input(i);
      if (inp->isa<CNode>() && node_map.count(inp) == 0) {
        que.push(inp->cast<CNodePtr>());
      }
    }
  }
  for (const auto &cnode : newcnodes) {
    for (size_t i = 1; i < cnode->size(); i++) {
      if (node_map.count(cnode->input(i)) != 0) {
        cnode->set_input(i, node_map[cnode->input(i)]);
      }
    }
  }
  expand_fg->set_output(node_map[new_output]);
  DumpIR("verbose_ir_files/expand_" + AnfUtils::GetCNodeName(ori_node) + ".ir", expand_fg, true);
}

bool IbTryExpandCNode(const IRBuilderHandle &handle, const CNodePtr &cnode, const SelectKernelFunc &func) {
  auto mng = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(mng);
  FallbackIRBuilder ib(AnfUtils::GetCNodeName(cnode), cnode->func_graph(), func);
  auto output = ib.Run(cnode, handle);
  if (output == nullptr) {
    MS_LOG(INFO) << "Undo expanding cnode " << cnode->fullname_with_scope();
    return false;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kAdvanced)) {
    if (output->isa<CNode>()) {
      DumpGraph(cnode, output->cast<CNodePtr>());
    } else {
      MS_LOG(INFO) << "The output is not a CNode, cannot dump graph. original node: " << cnode->fullname_with_scope()
                   << ", output->DebugString: " << output->DebugString();
    }
  }
#endif
  if (!(*cnode->abstract()->Broaden() == *output->abstract())) {
    MS_LOG(WARNING) << "After expanding cnode " << cnode->fullname_with_scope() << ", the new abstract of "
                    << output->fullname_with_scope() << " does not match original cnode's abstract. "
                    << "new: " << output->abstract()->ToString() << ", old: " << cnode->abstract()->ToString();
  }
  (void)mng->Replace(cnode, output);
  return true;
}

bool TryExpandCNode(const AnfNodePtr &node, const std::function<bool(const CNodePtr &)> &func) {
  if (!Check(node)) {
    return false;
  }
  MS_LOG(DEBUG) << "Try to expand node " << node->fullname_with_scope() << ". DebugString: " << node->DebugString();
  auto graph = node->func_graph();
  auto mng = graph->manager();
  if (mng == nullptr) {
    mng = Manage(graph, true);
    MS_EXCEPTION_IF_NULL(mng);
    graph->set_manager(mng);
  }
  const auto *handle = IRBuilderFactory::Instance().GetBuilder(AnfUtils::GetCNodeName(node));
  if (handle == nullptr) {
    return false;
  }
  return IbTryExpandCNode(*handle, node->cast<CNodePtr>(), func);
}
}  // namespace expander
}  // namespace mindspore
