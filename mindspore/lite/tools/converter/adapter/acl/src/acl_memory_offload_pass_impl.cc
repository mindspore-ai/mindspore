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

#define USE_DEPRECATED_API
#include <set>
#include <vector>
#include <string>
#include "tools/converter/adapter/acl/src/acl_memory_offload_pass_impl.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/common/custom_ascend_utils.h"

namespace {
constexpr auto kCustomPrimTypeACL = "ACL";
constexpr auto kFuncType = "func_type";
constexpr auto kUniqueName = "uniq_name";
}  // namespace
namespace mindspore {
namespace opt {
std::shared_ptr<mindspore::ops::Custom> AclMemoryOffloadPassImpl::CreateCustomPrim() {
  auto custom_prim = std::make_shared<mindspore::ops::Custom>();
  MS_CHECK_TRUE_MSG(custom_prim != nullptr, nullptr, "New custom op failed.");
  custom_prim->set_type(kCustomPrimTypeACL);
  custom_prim->AddAttr(kFuncType, api::MakeValue<std::string>("acl_build"));
  custom_prim->AddAttr(kUniqueName, api::MakeValue<std::string>("CustomAscend"));
  return custom_prim;
}

FuncGraphPtr AclMemoryOffloadPassImpl::CreateSingleOpFuncGraph(const CNodePtr &cnode) {
  FuncGraphPtr dstGraph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(dstGraph != nullptr, nullptr, "malloc FuncGraph failed.");

  auto prim = GetCNodePrimitive(cnode);
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr.");

  std::vector<AnfNodePtr> new_inputs;
  const auto &inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    ParameterPtr om_parameter = dstGraph->add_parameter();
    MS_CHECK_TRUE_MSG(om_parameter != nullptr, nullptr, "om_parameter is nullptr.");
    om_parameter->set_abstract(inputs[i]->abstract());
    new_inputs.push_back(om_parameter);
  }
  auto newnode = dstGraph->NewCNode(prim, new_inputs);

  newnode->set_abstract(cnode->abstract());
  newnode->set_forward(cnode->forward().first, cnode->forward().second);
  newnode->set_attrs(cnode->attrs());
  newnode->set_load_flag(cnode->get_load_flag());
  newnode->CloneUserData(cnode);
  newnode->set_kernel_info(cnode->kernel_info_ptr());
  newnode->set_primal_debug_infos(cnode->primal_debug_infos());
  newnode->set_fused_debug_infos(cnode->fused_debug_infos());

  MS_CHECK_TRUE_MSG(newnode != nullptr, nullptr, "create new cnode failed.");
  return dstGraph;
}

STATUS AclMemoryOffloadPassImpl::BuildGraph(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Deparser graph start.";
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_ERROR, "func_graph is nullptr.");
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(func_graph, &all_func_graphs);
  for (auto graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!utils::isa<CNodePtr>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "cnode is nullptr.");

      auto dstGraph = CreateSingleOpFuncGraph(cnode);
      MS_CHECK_TRUE_MSG(dstGraph != nullptr, lite::RET_ERROR, "create func graph failed.");

      Buffer om_data;
      if (ConvertGraphToOm(dstGraph, &om_data) != lite::RET_OK) {
        MS_LOG(ERROR) << "Convert graph  to om failed.";
        return lite::RET_ERROR;
      }

      auto om_parameter = CustomAscendUtils::CreateOmParameter(func_graph, om_data, "ACL_om_data");
      MS_CHECK_TRUE_MSG(om_parameter != nullptr, lite::RET_ERROR, "Convert graph to om failed.");

      auto custom_prim = CreateCustomPrim();
      MS_CHECK_TRUE_MSG(custom_prim != nullptr, lite::RET_ERROR, "create custom op failed.");
      // modify node's primitive to custom
      cnode->set_input(0, std::make_shared<ValueNode>(custom_prim->GetPrim()));
      cnode->add_input(om_parameter);
    }
  }
  return lite::RET_OK;
}

bool AclMemoryOffloadPassImpl::Run(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Acl pass run start.";
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
  auto manager = Manage(func_graph, true);
  MS_CHECK_TRUE_MSG(manager != nullptr, false, "manager is nullptr.");

  if (PreProcGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Pre proc graph failed.";
    return false;
  }

  if (DeparseGraph(func_graph, manager) != lite::RET_OK) {
    MS_LOG(ERROR) << "Deparse graph failed.";
    return false;
  }

  if (BuildGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Build graph failed.";
    return false;
  }

  MS_LOG(INFO) << "Acl memory offload pass run end.";
  return true;
}
}  //  namespace opt
}  // namespace mindspore
