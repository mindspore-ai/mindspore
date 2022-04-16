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

#include "common/graph_kernel/core/graph_kernel_expander.h"

#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "common/graph_kernel/core/graph_builder.h"
#include "common/graph_kernel/core/graph_kernel_callback.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"
#include "common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel {
ExpanderPtr WrapExpander(const ExpanderPtr &base, const ExpanderCreatorFuncList &deco_creators) {
  ExpanderPtr result = base;
  for (auto &creator : deco_creators) {
    result = creator(result);
  }
  return result;
}

AnfNodePtr ExpanderDecorator::Run(const AnfNodePtr &node) {
  if (node == nullptr) return nullptr;
  auto newnode = PreProcess(node);
  if (newnode == nullptr) return nullptr;
  newnode = decorated_->Run(newnode);
  if (newnode == nullptr) return nullptr;
  return PostProcess(newnode);
}

CNodePtr ExpanderDecorator::QuickCloneCNode(const AnfNodePtr &node) const {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr new_node = func_graph->NewCNode(cnode->inputs());
  new_node->set_abstract(node->abstract());
  new_node->set_kernel_info(node->kernel_info_ptr());
  return new_node;
}

FuncGraphPtr DefaultExpander::ExpandToGraph(const CNodePtr &node) {
  auto op_desc = expanders::OpDescFactory::Instance().GetOp(AnfUtils::GetCNodeName(node));
  if (op_desc == nullptr) {
    MS_LOG(INFO) << "expander not found " << node->fullname_with_scope();
    return nullptr;
  }
  expanders::BaseInfoList inputs(node->size() - 1);
  expanders::BaseInfoList outputs(AnfUtils::GetOutputTensorNum(node));
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs[i].shape = cb->GetInputShape(node, i);
    inputs[i].type = cb->GetInputType(node, i);
    inputs[i].format = cb->GetInputFormat(node, i);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i].shape = cb->GetOutputShape(node, i);
    outputs[i].type = cb->GetOutputType(node, i);
    outputs[i].format = cb->GetOutputFormat(node, i);
  }
  auto &attrs = GetCNodePrimitive(node)->attrs();
  auto litegraph = op_desc->Run(inputs, outputs, attrs, cb->GetProcessor(node));
  if (litegraph == nullptr) {
    MS_LOG(INFO) << "undo expanding " << node->fullname_with_scope();
    return nullptr;
  }
  return GkUtils::LiteGraph2AnfGraph(litegraph);
}

AnfNodePtr GraphKernelExpander::CreateExpandedNode(const CNodePtr &node) {
  auto new_fg = GetCNodeFuncGraph(node);
  auto func_graph = node->func_graph();
  std::vector<AnfNodePtr> inputs(node->inputs().begin() + 1, node->inputs().end());
  auto graph_kernel_node = CreateNewFuseCNode(func_graph, new_fg, inputs);
  MS_LOG(DEBUG) << "Expand node: " << node->fullname_with_scope()
                << " with: " << graph_kernel_node->fullname_with_scope();
  return graph_kernel_node;
}

AnfNodePtr DefaultExpander::Run(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_fg = ExpandToGraph(cnode);
  if (new_fg == nullptr) return nullptr;
  new_fg->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(AnfUtils::GetCNodeName(cnode)));
  AnfNodePtrList inputs = {NewValueNode(new_fg)};
  inputs.insert(inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
  return node->func_graph()->NewCNode(inputs);
}

bool GraphKernelExpander::DoExpand(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  std::reverse(todos.begin(), todos.end());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (const auto &n : todos) {
    auto node = n->cast<CNodePtr>();
    if (node == nullptr || AnfUtils::IsGraphKernel(node) || GkUtils::IsKeepBasicNode(node) ||
        !AnfUtils::IsRealKernel(node) || !CanExpand(node)) {
      continue;
    }

    MS_LOG(DEBUG) << "Expanding node: " << node->fullname_with_scope();
    auto newnode = GetExpander(node)->Run(node);
    if (newnode == nullptr) {
      MS_LOG(DEBUG) << "Skipped node: " << node->fullname_with_scope();
      continue;
    }
    newnode = CreateExpandedNode(newnode->cast<CNodePtr>());
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
