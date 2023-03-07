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

#include "backend/common/graph_kernel/csr_atomic_add.h"
#include <memory>
#include <vector>
#include "mindspore/core/ops/core_ops.h"
#include "ir/tensor.h"
#include "include/common/utils/utils.h"
#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/session/kernel_graph.h"

namespace mindspore::graphkernel {
class ReduceSumCsrChecker : public AtomicAddChecker {
 public:
  ReduceSumCsrChecker() = default;

 protected:
  bool CanActivateAtomicAdd(const AnfNodePtr &node) override {
    bool has_csr = false;
    bool has_reduce_sum = false;
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(func_graph);
    for (auto n : func_graph->nodes()) {
      if (n->isa<CNode>() && IsAKGSparseOP(n)) {
        has_csr = true;
        break;
      } else if (IsPrimitiveCNode(n, prim::kPrimReduceSum)) {
        has_reduce_sum = true;
      }
    }
    if (has_csr && has_reduce_sum) {
      return FindCandidate(node);
    }
    return false;
  }

  bool FindCandidate(const AnfNodePtr &anf_node) override {
    atomic_add_infos_.clear();
    auto node = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
    auto sub_mng = sub_graph->manager();
    if (sub_mng == nullptr) {
      sub_mng = Manage(sub_graph, false);
      sub_graph->set_manager(sub_mng);
    }

    auto CheckSuitableTarget = [&sub_mng](const InplaceAssignerInfo &atomic_add_infos) {
      // Target type should not fuse any other ops in out direction, which means it should be in output list.
      return sub_mng->node_users()[atomic_add_infos.op_node].size() <= 1;
    };

    auto real_return_node = sub_graph->get_return()->input(kFirstDataInputIndex);
    InplaceAssignerInfo atomic_add_infos;
    if (IsPrimitiveCNode(real_return_node, prim::kPrimMakeTuple)) {
      const auto &inputs = real_return_node->cast<CNodePtr>()->inputs();
      for (size_t i = 1; i < inputs.size(); ++i) {
        atomic_add_infos.op_node = inputs[i]->cast<CNodePtr>();
        atomic_add_infos.real_output_index = i - 1;
        atomic_add_infos.real_output_num = inputs.size() - 1;
        // Target type should not fuse any other ops in out direction, which means it should be in output list.
        if (CheckSuitableTarget(atomic_add_infos)) {
          atomic_add_infos_.push_back(atomic_add_infos);
        }
      }
    } else if (real_return_node->isa<CNode>()) {
      atomic_add_infos.op_node = real_return_node->cast<CNodePtr>();
      atomic_add_infos.real_output_num = 1;
      if (CheckSuitableTarget(atomic_add_infos)) {
        atomic_add_infos_.push_back(atomic_add_infos);
      }
    } else {
      return false;
    }
    return !atomic_add_infos_.empty();
  }
};

bool CsrAtomicAdd::Run(const FuncGraphPtr &func_graph) {
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }

  bool changed = false;
  std::shared_ptr<AtomicAddChecker> csr_reduce_sum_checker =
    std::make_shared<TargetAtomicAddChecker>(prim::kPrimCSRReduceSum);
  MS_EXCEPTION_IF_NULL(csr_reduce_sum_checker);
  std::shared_ptr<AtomicAddChecker> reduce_sum_csr_checker = std::make_shared<ReduceSumCsrChecker>();
  MS_EXCEPTION_IF_NULL(reduce_sum_csr_checker);

  auto topo_nodes = TopoSort(kernel_graph->get_return());
  for (const auto &node : topo_nodes) {
    std::vector<InplaceAssignerInfo> atomic_add_infos;
    if (csr_reduce_sum_checker->Check(node)) {
      atomic_add_infos = csr_reduce_sum_checker->GetAtomicAddInfo();
    } else if (reduce_sum_csr_checker->Check(node)) {
      atomic_add_infos = reduce_sum_csr_checker->GetAtomicAddInfo();
    }
    if (!atomic_add_infos.empty()) {
      InsertAtomicClean(kernel_graph, node, atomic_add_infos, mng);
      changed = true;
    }
  }

  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }

  return changed;
}
}  // namespace mindspore::graphkernel
