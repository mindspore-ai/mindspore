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

#include "tools/converter/import/remove_public_primitive.h"
#include <memory>
#include <set>
#include <string>
#include "tools/converter/parser/parser_utils.h"
#include "nnacl/op_base.h"
#include "ops/core_ops.h"

namespace mindspore {
namespace lite {
namespace {
constexpr char kDoSignaturePrimitivePrefix[] = "S-Prim-";
constexpr char kHyperMapPrefix[] = "hyper_map";
constexpr auto offset = 2;
}  // namespace

bool RemovePublicPrimitiveInterference::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(func_graph, &all_func_graphs);
  std::set<AnfNodePtr> has_visited;
  for (const auto &graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!utils::isa<CNodePtr>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      MS_ASSERT(cnode->size() > 0);
      auto first_input = cnode->input(0);
      MS_ASSERT(first_input != nullptr);
      if (GetValueNode<PrimitivePtr>(first_input) == nullptr) {
        continue;
      }
      if (has_visited.find(first_input) != has_visited.end()) {
        auto succ = CreateIndividualPrim(cnode);
        if (!succ) {
          MS_LOG(ERROR) << "create individual primitive failed. node name is " << cnode->fullname_with_scope();
          return succ;
        }
      } else {
        (void)has_visited.insert(first_input);
      }
    }
  }
  return true;
}

bool RemovePublicPrimitiveInterference::CreateIndividualPrim(const CNodePtr &cnode) {
  auto public_prim = GetCNodePrimitive(cnode);
  MS_ASSERT(public_prim != nullptr);
  // Operator is  primitive.
  PrimitivePtr prim;
  std::string node_type = public_prim->name();
  auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  if (op_primc_fns.find(node_type) != op_primc_fns.end()) {
    prim = op_primc_fns[node_type]();
    MS_CHECK_TRUE_MSG(prim != nullptr, false, "create primitive failed.");
  } else {
    if (node_type.compare(0, strlen(kDoSignaturePrimitivePrefix), kDoSignaturePrimitivePrefix) == 0) {
      auto op_name = node_type.substr(strlen(kDoSignaturePrimitivePrefix));
      if (op_name.compare(0, strlen(kHyperMapPrefix), kHyperMapPrefix) == 0) {
        op_name = op_name.substr(strlen(kHyperMapPrefix) + 1, (op_name.length() - strlen(kHyperMapPrefix)) - offset);
      }
      prim = std::make_shared<prim::DoSignaturePrimitive>(op_name, std::make_shared<Primitive>(op_name));
      MS_CHECK_TRUE_MSG(prim != nullptr, false, "create primitive failed.");
      prim->set_instance_name(op_name);
    } else {
      MS_LOG(DEBUG) << "Special node_type: " << node_type;
      prim = std::make_shared<Primitive>(node_type);
      MS_CHECK_TRUE_MSG(prim != nullptr, false, "create primitive failed.");
      prim->set_instance_name(node_type);
    }
  }
  (void)prim->SetAttrs(public_prim->attrs());
  auto value_node = std::make_shared<ValueNode>(prim);
  MS_CHECK_TRUE_MSG(value_node != nullptr, false, "create valueNode failed.");
  cnode->set_input(0, value_node);
  return true;
}
}  // namespace lite
}  // namespace mindspore
