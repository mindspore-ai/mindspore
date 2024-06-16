/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/convert_call_to_prim.h"

#include <memory>
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore::graphkernel {
bool ConvertCallToPrim::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  auto todos = TopoSort(func_graph->output());
  bool is_dvm = (GraphKernelFlags::GetInstance().kernel_generator == "DVM");
  for (auto node : todos) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || (!cnode->HasAttr(kAttrToPrim) && !is_dvm)) {
      continue;
    }
    auto sub_fg = GetCNodeFuncGraph(node);
    if (sub_fg != nullptr) {
      bool has_attr_to_prim = cnode->HasAttr(kAttrToPrim);
      std::string prim_name =
        has_attr_to_prim ? GetValue<std::string>(cnode->GetAttr(kAttrToPrim)) : common::AnfAlgo::GetCNodeName(node);
      auto new_prim = std::make_shared<Primitive>(prim_name, sub_fg->attrs());
      new_prim->AddAttr(kAttrFuncGraph, sub_fg);
      auto prim_input = NewValueNode(new_prim);
      if (!has_attr_to_prim) {
        // do not create a new node, otherwise the ref pair saved in kernel graph is invalid
        cnode->set_input(0, prim_input);
        changed = true;
        continue;
      }
      AnfNodePtrList new_inputs = node->cast<CNodePtr>()->inputs();
      new_inputs[0] = prim_input;
      auto newnode = func_graph->NewCNode(new_inputs);
      newnode->CloneCNodeInfo(cnode);
      newnode->EraseAttr(kAttrToPrim);
      auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
      if (kernel_mod != nullptr) {
        kernel_mod->Init(new_prim, {}, {});
      }
      mng->Replace(node, newnode);
      changed = true;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
