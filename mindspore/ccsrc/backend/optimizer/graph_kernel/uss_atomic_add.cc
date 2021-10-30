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

#include "backend/optimizer/graph_kernel/uss_atomic_add.h"
#include <vector>
#include <tuple>
#include <string>
#include <stack>
#include <set>
#include <utility>
#include <memory>
#include <map>
#include <list>
#include <functional>
#include <algorithm>
#include "base/core_ops.h"
#include "ir/tensor.h"
#include "utils/utils.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace opt {
class UssChecker : public AtomicAddChecker {
 public:
  explicit UssChecker(const PrimitivePtr &target) { target_type_ = target; }
  virtual ~UssChecker() = default;

 protected:
  bool CanActivateAtomicAdd(const AnfNodePtr &anf_node) override { return FindCandidate(anf_node); }
};

bool UssAtomicAdd::Run(const FuncGraphPtr &func_graph) {
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }

  bool changed = false;
  std::shared_ptr<AtomicAddChecker> atomic_add_checker =
    std::make_shared<UssChecker>(std::make_shared<Primitive>("UnsortedSegmentSum"));
  if (atomic_add_checker == nullptr) {
    return changed;
  }

  auto topo_nodes = TopoSort(kernel_graph->get_return());
  for (const auto &node : topo_nodes) {
    if (!atomic_add_checker->Check(node)) {
      continue;
    }
    auto info = atomic_add_checker->GetAtomicAddInfo();
    UpdateAtomicAddInfo(info);
    InsertAtomicClean(kernel_graph, node, mng);
    changed = true;
  }

  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }

  return changed;
}
}  // namespace opt
}  // namespace mindspore
