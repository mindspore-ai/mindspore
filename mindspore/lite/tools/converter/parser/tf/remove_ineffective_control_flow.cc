/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/tf/remove_ineffective_control_flow.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/core_ops.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool RemoveIneffectiveControlFlow::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, false, "manager is a nullptr.");
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!opt::CheckPrimitiveType(node, prim::kPrimMerge)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!CheckIfIneffective(cnode)) {
      continue;
    }
    (void)manager->Replace(node, shared_input_);
  }
  return true;
}

bool RemoveIneffectiveControlFlow::CheckIfIneffective(const CNodePtr &merge) {
  MS_ASSERT(merge != nullptr);
  if (merge->size() != opt::kInputSizeThree) {
    return false;
  }
  shared_input_ = nullptr;
  auto IsSwitch = [this](const AnfNodePtr &anf_node) {
    if (!utils::isa<CNode>(anf_node)) {
      return false;
    }
    if (!opt::CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
      return false;
    }
    auto cnode = anf_node->cast<CNodePtr>();
    auto switch_node = cnode->input(1);
    if (!utils::isa<CNode>(switch_node)) {
      return false;
    }
    if (!opt::CheckPrimitiveType(switch_node, prim::kPrimSwitch)) {
      return false;
    }
    auto switch_cnode = switch_node->cast<CNodePtr>();
    if (switch_cnode->size() != opt::kInputSizeThree) {
      return false;
    }
    if (this->shared_input_ == nullptr) {
      this->shared_input_ = switch_cnode->input(1);
    } else if (this->shared_input_ != switch_cnode->input(1)) {
      return false;
    }
    return true;
  };
  return IsSwitch(merge->input(1)) && IsSwitch(merge->input(opt::kInputSizeTwo));
}
}  // namespace lite
}  // namespace mindspore
