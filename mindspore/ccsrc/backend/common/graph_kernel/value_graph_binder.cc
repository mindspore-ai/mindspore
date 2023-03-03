/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/value_graph_binder.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "backend/common/session/kernel_graph.h"
namespace mindspore::graphkernel {
bool BindValueToGraph::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto todos = TopoSort(func_graph->get_return());
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &value_nodes = kernel_graph->graph_value_nodes();
  bool changed = false;
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  for (auto node : todos) {
    if (!GetValueNode<tensor::TensorPtr>(node)) {
      continue;
    }
    if (auto vptr = node->cast<ValueNodePtr>(); value_nodes.count(vptr) == 0) {
      auto new_node = kernel_graph->NewValueNode(vptr);
      auto ori_kernel_info = dynamic_cast<device::KernelInfo *>(vptr->kernel_info());
      MS_EXCEPTION_IF_NULL(ori_kernel_info);
      if (ori_kernel_info->has_build_info()) {
        const auto &ori_kernel_build_info = ori_kernel_info->GetMutableSelectKernelBuildInfo();
        AnfAlgo::SetSelectKernelBuildInfo(ori_kernel_build_info, new_node.get());
      }
      (void)mng->Replace(vptr, new_node);
      kernel_graph->AddValueNodeToGraph(new_node);
      changed = true;
    }
  }

  return changed;
}
}  // namespace mindspore::graphkernel
