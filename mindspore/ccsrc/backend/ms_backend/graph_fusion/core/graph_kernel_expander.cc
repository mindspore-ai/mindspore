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

#include "backend/ms_backend/graph_fusion/core/graph_kernel_expander.h"

#include <string>
#include "ir/core_ops_primitive.h"
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "backend/ms_backend/graph_fusion/expander/base/ir_builder.h"
#include "backend/ms_backend/graph_fusion/core/graph_builder.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "ir/func_graph_flag.h"

namespace mindspore::graphkernel {
HashMap<std::string, ValuePtr> GetAttributesToInherit(const CNodePtr &cnode) {
  HashMap<std::string, ValuePtr> attr_map;
  static const std::vector<std::string> attr_list{
    kAttrDuplicated,
  };
  for (const auto &attr : attr_list) {
    if (cnode->HasAttr(attr)) {
      attr_map[attr] = cnode->GetAttr(attr);
    }
  }
  return attr_map;
}

AnfNodePtr GraphKernelExpander::CreateExpandedNode(const CNodePtr &node, const CNodePtr &ori_node) const {
  const std::string &name = AnfUtils::GetCNodeName(ori_node);
  auto new_fg = GetCNodeFuncGraph(node);
  new_fg->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(name));
  auto main_graph = node->func_graph();
  std::vector<AnfNodePtr> inputs(node->inputs().begin() + 1, node->inputs().end());
  (void)ConvertTensorToParameter(new_fg, &inputs);
  auto graph_kernel_node = CreateNewFuseCNode(main_graph, new_fg, inputs);
  // update sub graph nodes attr
  auto expand_from = MakeValue(name);
  const auto &attr_map = GetAttributesToInherit(ori_node);
  auto nodes = TopoSort(new_fg->get_return());
  for (const auto &n : nodes) {
    if (n == nullptr || !n->isa<CNode>()) {
      continue;
    }
    auto cnode = n->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (const auto &iter : attr_map) {
      cnode->AddAttr(iter.first, iter.second);
    }
    cnode->AddAttr(kAttrExpandFrom, expand_from);
  }
  MS_LOG(DEBUG) << "Expand node: " << node->fullname_with_scope()
                << " with: " << graph_kernel_node->fullname_with_scope();
  return graph_kernel_node;
}

void ReplaceNodeWithTupleGetItem(const AnfNodePtr &node, const AnfNodePtr &newnode, const FuncGraphPtr &func_graph,
                                 const FuncGraphManagerPtr &mng) {
  auto &ib_registry = expander::IrBuilderRegistry::Instance();
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  const std::string &name = prim->name();
  const auto &output_indices = ib_registry.GetOutputNumInconsistentOps().at(name);
  if (output_indices.size() == 1) {
    auto idx = MakeValue(SizeToLong(output_indices[0]));
    AnfNodePtrList inputs{NewValueNode(prim::kPrimTupleGetItem), newnode, NewValueNode(idx)};
    inputs.back()->set_abstract(idx->ToAbstract());
    auto new_out = func_graph->NewCNode(inputs);
    auto abs = newnode->abstract();
    if (!abs->isa<abstract::AbstractSequence>()) {
      MS_LOG(EXCEPTION) << "The output abstract has to be an abstract sequence";
    }
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    auto elements = abs_seq->elements();
    new_out->set_abstract(elements[output_indices[0]]);
    mng->Replace(node, new_out);
  } else {
    mng->Replace(node, newnode);
  }
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
    MS_LOG(DEBUG) << "Expanding node run start: " << node->fullname_with_scope();
    auto newnode = InitExpander(node)->Run(node);
    MS_LOG(DEBUG) << "Expanding node run end: " << node->fullname_with_scope();
    if (newnode == nullptr) {
      MS_LOG(DEBUG) << "Skipped node: " << node->fullname_with_scope();
      continue;
    }
    if (newnode->isa<CNode>()) {
      newnode = CreateExpandedNode(newnode->cast<CNodePtr>(), node);
    }
    if (newnode == nullptr) {
      MS_LOG(DEBUG) << "Skipped node: " << node->fullname_with_scope();
      continue;
    }
    // For some ops, the output number of expander is different from the original cnode. In this case, a TupleGetItem is
    // needed to insure that later cnodes have correct input
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    const std::string &name = prim->name();
    if (expander::IrBuilderRegistry::Instance().IsOutputNumInconsistent(name)) {
      ReplaceNodeWithTupleGetItem(node, newnode, func_graph, mng);
    } else {
      mng->Replace(node, newnode);
    }
    changed = true;
  }
  return changed;
}

bool GraphKernelExpander::Run(const FuncGraphPtr &func_graph) {
  expand_ops_ = InitOpList();
  return DoExpand(func_graph);
}
}  // namespace mindspore::graphkernel
