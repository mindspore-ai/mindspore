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

#include "backend/common/graph_kernel/uss_atomic_add.h"
#include <memory>
#include "mindspore/core/ops/core_ops.h"
#include "ir/tensor.h"
#include "include/common/utils/utils.h"
#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/session/kernel_graph.h"

namespace mindspore::graphkernel {
bool UssAtomicAdd::Run(const FuncGraphPtr &func_graph) {
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }

  bool has_change = false;
  std::shared_ptr<AtomicAddChecker> checker =
    std::make_shared<TargetAtomicAddChecker>(std::make_shared<Primitive>("UnsortedSegmentSum"));
  if (checker == nullptr) {
    return has_change;
  }

  auto topo_nodes = TopoSort(kernel_graph->get_return());
  for (const auto &node : topo_nodes) {
    if (!checker->Check(node)) {
      continue;
    }
    auto atomic_add_infos = checker->GetAtomicAddInfo();
    InsertAtomicClean(kernel_graph, node, atomic_add_infos, mng);
    has_change = true;
  }

  if (has_change) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }

  return has_change;
}
}  // namespace mindspore::graphkernel
