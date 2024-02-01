/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/ps/load_mindir.h"

#include <string>
#include <set>
#include <memory>
#include <algorithm>

#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "utils/check_convert_utils.h"
#include "load_mindir/infer_mindir.h"

namespace mindspore {
namespace pipeline {
bool InferMindIR(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  const auto &root = resource->func_graph();
  InferFuncGraphLoaded(root);
  return true;
}

std::vector<AnfNodePtr> ArgsNeededToConvert(const PrimitivePtr &prim) {
  auto op_def = mindspore::ops::GetOpDef(prim->name());
  std::vector<AnfNodePtr> prim_init_arg_nodes;
  MS_EXCEPTION_IF_NULL(op_def);
  // Get init args.
  for (const auto &op_arg : op_def->args_) {
    if (op_arg.as_init_arg_) {
      auto arg_name = op_arg.arg_name_;
      ValuePtr attr;
      // "data_format" is renamed as "format" for some operator.
      if (CheckAndConvertUtils::CheckPrimAttrConverted(prim->name()) && arg_name == "data_format" &&
          prim->HasAttr("format")) {
        attr = prim->GetAttr("format");
      } else if (!prim->HasAttr(arg_name)) {
        attr = parse::GetArgDefaultValue(prim->name(), arg_name);
        if (attr == nullptr) {
          MS_LOG(EXCEPTION) << "Cannot find attribute: " << arg_name << " from primitive :" << prim->name();
        }
      } else {
        attr = prim->GetAttr(arg_name);
      }
      (void)prim_init_arg_nodes.emplace_back(NewValueNode(attr));
    }
  }
  return prim_init_arg_nodes;
}

void ModifyOneCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();
  if (IsValueNode<Primitive>(inputs[0])) {
    auto prim = GetValueNode<PrimitivePtr>(inputs[0]);
    if (mindspore::ops::IsPrimitiveFunction(prim->name())) {
      // Append Primitive arguments to the inputs.
      std::vector<AnfNodePtr> prim_init_arg_nodes = ArgsNeededToConvert(prim);
      // Get call args.
      AnfNodePtrList prim_call_arg_nodes(inputs.begin() + 1, inputs.end());
      // Create new node.
      auto new_prim = std::make_shared<Primitive>(*prim);
      AnfNodePtrList input_nodes{NewValueNode(new_prim)};
      (void)std::copy(prim_call_arg_nodes.cbegin(), prim_call_arg_nodes.cend(), std::back_inserter(input_nodes));
      (void)std::copy(prim_init_arg_nodes.cbegin(), prim_init_arg_nodes.cend(), std::back_inserter(input_nodes));
      auto new_cnode = func_graph->NewCNodeInOrder(input_nodes);
      MS_LOG(DEBUG) << "Convert primitive args: " << prim->name() << ". node: " << cnode->DebugString()
                    << ", new_node: " << new_cnode->DebugString();
      auto manager = func_graph->manager();
      if (manager == nullptr) {
        manager = MakeManager();
        manager->AddFuncGraph(func_graph, true);
      }
      (void)manager->Replace(cnode, new_cnode);
    }
  }
}

void ModifyOneFuncGraph(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *func_graph_set,
                        std::set<FuncGraphPtr> *func_graph_modified) {
  MS_LOG(DEBUG) << "Start modifying: " << func_graph->ToString();
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    ModifyOneCNode(func_graph, cnode);
    auto &inputs = cnode->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (IsValueNode<FuncGraph>(inputs[i])) {
        FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(inputs[i]);
        if ((*func_graph_set).find(fg) == (*func_graph_set).end() &&
            (*func_graph_modified).find(fg) == (*func_graph_modified).end()) {
          (void)(*func_graph_set).insert(fg);
        }
      }
    }
  }
}

bool ModifyGraphGeneratedByMindIR(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  const auto &func_graph = resource->func_graph();
  std::set<FuncGraphPtr> func_graph_set{};
  std::set<FuncGraphPtr> func_graph_modified{};
  func_graph_set.insert(func_graph);
  // Check every node in every graph to find nodes needed to convert.
  while (!func_graph_set.empty()) {
    FuncGraphPtr fg = *func_graph_set.cbegin();
    if (!func_graph->has_flag("generated_from_mindir_with_prim_func")) {
      ModifyOneFuncGraph(fg, &func_graph_set, &func_graph_modified);
    }
    (void)func_graph_set.erase(fg);
    (void)func_graph_modified.insert(fg);
  }
  return true;
}
}  // namespace pipeline
}  // namespace mindspore
