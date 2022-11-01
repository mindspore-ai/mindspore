/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/add_training_attr.h"

#include <vector>
#include <memory>
#include <utility>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/graph_utils.h"
#include "backend/common/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
mindspore::HashMap<std::string, mindspore::HashSet<std::string>> MarkOp{
  {"LSTM", {"LSTMGradWeight", "LSTMGrad", "LSTMGradData"}}};

bool CheckOP(const FuncGraphManagerPtr &manager, const AnfNodePtr &cnode, const mindspore::HashSet<std::string> &set) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(cnode);
  for (const auto &node_index : manager->node_users()[cnode]) {
    auto output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (common::AnfAlgo::CheckPrimitiveType(output, prim::kPrimTupleGetItem)) {
      if (CheckOP(manager, output, set)) {
        return true;
      }
    } else if (output->isa<CNode>()) {
      auto name = common::AnfAlgo::GetCNodeName(output);
      if (set.find(name) != set.end()) {
        return true;
      }
    }
  }
  return false;
}
void AddAttrTraining(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(cnode) == manager->node_users().end()) {
    return;
  }
  auto set = MarkOp[common::AnfAlgo::GetCNodeName(cnode)];
  if (CheckOP(manager, cnode, set)) {
    cnode->AddAttr(kAttrIsTraining, MakeValue(true));
  } else {
    cnode->AddAttr(kAttrIsTraining, MakeValue(false));
  }
}
}  // namespace

const AnfNodePtr AddTrainingAttr::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  if (node == nullptr || func_graph == nullptr || common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem) ||
      common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    return nullptr;
  }
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  auto name = common::AnfAlgo::GetCNodeName(node);
  auto iter = MarkOp.find(name);
  if (iter == MarkOp.end()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  AddAttrTraining(func_graph, cnode);
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
