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

#include "common/graph_kernel/lite_adapter/build_kernel.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/graph_kernel/lite_adapter/akg_build.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ops/custom.h"
#include "utils/log_adapter.h"

namespace mindspore::graphkernel {
namespace {
void BuildAKGKernel(const std::vector<AnfNodePtr> &node_list) {
  AnfNodePtrList anf_list;
  for (auto &node : node_list) {
    if (AnfUtils::IsGraphKernel(node)) {
      anf_list.push_back(node);
    }
  }
  graphkernel::AkgKernelBuilder gk;
  if (!gk.CompileJsonsInAnfnodes(anf_list)) {
    MS_LOG(EXCEPTION) << "Graph kernel compile fail";
  }
}
}  // namespace

AnfNodePtr KernelBuilder::CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  if (func_graph == nullptr || cnode == nullptr) {
    return nullptr;
  }
  auto primc = std::make_shared<ops::Custom>();
  if (primc == nullptr) {
    return nullptr;
  }
  primc->set_type("GraphKernel");
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  auto fg = GetCNodeFuncGraph(cnode);
  MS_EXCEPTION_IF_NULL(fg);
  auto kernel_name = GetValue<std::string>(fg->get_attr("kernel_name"));
  std::vector<uint8_t> kernel_name_str(kernel_name.begin(), kernel_name.end());
  custom_attrs["kernel_name"] = kernel_name_str;
  primc->set_attr(custom_attrs);
  auto inputs = cnode->inputs();
  inputs.erase(inputs.begin());
  auto custom_cnode = func_graph->NewCNode(primc, inputs);
  custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
  custom_cnode->set_abstract(cnode->abstract()->Clone());
  return custom_cnode;
}

bool KernelBuilder::Run(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  BuildAKGKernel(node_list);
  bool changed = false;
  auto manager = Manage(func_graph, true);
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : node_list) {
    if (!AnfUtils::IsGraphKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto custom_cnode = CreateCustomOp(func_graph, cnode);
    if (custom_cnode == nullptr) {
      MS_LOG(EXCEPTION) << "Create custom op fail for " << cnode->fullname_with_scope();
    }
    manager->Replace(node, custom_cnode);
    changed = true;
  }
  return changed;
}
}  // namespace mindspore::graphkernel
