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

#include "src/pass_registry_tutorial.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "include/registry/pass_registry.h"
#include "ops/custom.h"

namespace mindspore {
namespace opt {
namespace {
// check a certain node is designated node's type.
bool CheckPrimitiveTypeTutorial(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  if (node == nullptr) {
    return false;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    return IsPrimitive(cnode->input(0), primitive_type);
  } else if (node->isa<ValueNode>()) {
    return IsPrimitive(node, primitive_type);
  }
  return false;
}
}  // namespace

// convert addn to custom op
AnfNodePtr PassTutorial::CreateCustomOp(const api::FuncGraphPtr func_graph, const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return nullptr;
  }
  auto primc = std::make_shared<ops::Custom>();
  if (primc == nullptr) {
    return nullptr;
  }
  primc->set_type("Custom_Add");
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  std::string input_num = std::to_string(cnode->size() - 1);
  std::vector<uint8_t> input_num_attr(input_num.begin(), input_num.end());
  custom_attrs["input_num"] = input_num_attr;
  std::string op_kind = "custom op";
  std::vector<uint8_t> op_kind_attr(op_kind.begin(), op_kind.end());
  custom_attrs["op_kind"] = op_kind_attr;
  primc->set_attr(custom_attrs);
  auto inputs = cnode->inputs();
  inputs.erase(inputs.begin());
  auto custom_cnode = func_graph->NewCNode(primc, inputs);
  custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
  custom_cnode->set_abstract(cnode->abstract()->Clone());
  return custom_cnode;
}

bool PassTutorial::Execute(const api::FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return false;
  }

  // generate a func_graph manager.
  auto manager = api::FuncGraphManager::Manage(func_graph, true);
  if (manager == nullptr) {
    return false;
  }
  auto node_list = api::FuncGraph::TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (!CheckPrimitiveTypeTutorial(node, prim::kPrimAddFusion)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto custome_cnode = CreateCustomOp(func_graph, cnode);
    if (custome_cnode == nullptr) {
      return false;
    }
    // use new node to replace old node by func_graph manager.
    manager->Replace(node, custome_cnode);
  }
  return true;
}
}  // namespace opt

namespace lite {
// register customed Pass
using mindspore::registry::POSITION_BEGIN;
REG_PASS(PassTutorial, opt::PassTutorial)
REG_SCHEDULED_PASS(POSITION_BEGIN, {"PassTutorial"})
}  // namespace lite
}  // namespace mindspore
