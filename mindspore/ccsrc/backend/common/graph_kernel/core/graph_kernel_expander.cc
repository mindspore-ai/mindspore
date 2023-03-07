/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/core/graph_kernel_expander.h"

#include <string>
#include <vector>
#include <algorithm>

#include "utils/anf_utils.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/core/convert_op_input_attr.h"

namespace mindspore::graphkernel {
AnfNodePtr GraphKernelExpander::CreateExpandedNode(const CNodePtr &node, const std::string &name) const {
  auto new_fg = GetCNodeFuncGraph(node);
  new_fg->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(name));
  auto main_graph = node->func_graph();
  std::vector<AnfNodePtr> inputs(node->inputs().begin() + 1, node->inputs().end());
  (void)ConvertNonscalarTensorToParameter(new_fg, &inputs);
  auto graph_kernel_node = CreateNewFuseCNode(main_graph, new_fg, inputs);
  MS_LOG(DEBUG) << "Expand node: " << node->fullname_with_scope()
                << " with: " << graph_kernel_node->fullname_with_scope();
  return graph_kernel_node;
}

bool GraphKernelExpander::DoExpand(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto todos = TopoSort(func_graph->output());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (const auto &n : todos) {
    auto node = n->cast<CNodePtr>();
    if (node != nullptr) {
      PreProcessAllNode(node);
    }
    if (node == nullptr || AnfUtils::IsGraphKernel(node) || GkUtils::IsKeepBasicNode(node) ||
        !AnfUtils::IsRealKernel(node) || !CanExpand(node)) {
      continue;
    }
    MS_LOG(DEBUG) << "Expanding node: " << node->fullname_with_scope();
    auto newnode = InitExpander(node)->Run(node);
    if (newnode == nullptr) {
      MS_LOG(DEBUG) << "Skipped node: " << node->fullname_with_scope();
      continue;
    }
    if (newnode->isa<CNode>()) {
      newnode = CreateExpandedNode(newnode->cast<CNodePtr>(), AnfUtils::GetCNodeName(node));
    }
    if (newnode == nullptr) {
      MS_LOG(DEBUG) << "Skipped node: " << node->fullname_with_scope();
      continue;
    }
    (void)mng->Replace(node, newnode);
    changed = true;
  }
  return changed;
}

bool GraphKernelExpander::Run(const FuncGraphPtr &func_graph) {
  expand_ops_ = InitOpList();
  return DoExpand(func_graph);
}
}  // namespace mindspore::graphkernel
