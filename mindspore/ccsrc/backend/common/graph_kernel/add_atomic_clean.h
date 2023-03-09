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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADD_ATOMIC_CLEAN_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADD_ATOMIC_CLEAN_H_

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/kernel_graph.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/inplace_assign_builder.h"

namespace mindspore::graphkernel {
class AtomicAddChecker {
 public:
  AtomicAddChecker() = default;
  virtual ~AtomicAddChecker() = default;
  static std::shared_ptr<AtomicAddChecker> Init();

  bool Check(const AnfNodePtr &node);
  std::vector<InplaceAssignerInfo> GetAtomicAddInfo() { return atomic_add_infos_; }

 protected:
  virtual bool SuitableForAtomicAdd(const AnfNodePtr &) { return false; }
  virtual bool FindCandidate(const AnfNodePtr &anf_node);
  virtual bool CanActivateAtomicAdd(const AnfNodePtr &anf_node);
  std::vector<InplaceAssignerInfo> atomic_add_infos_;
  PrimitivePtr target_type_{prim::kPrimReduceSum};
};

class TargetAtomicAddChecker : public AtomicAddChecker {
 public:
  explicit TargetAtomicAddChecker(const PrimitivePtr &target = prim::kPrimReduceSum) { target_type_ = target; }

 protected:
  bool CanActivateAtomicAdd(const AnfNodePtr &anf_node) override { return FindCandidate(anf_node); }
};

class AtomicAddCheckerGPU : public AtomicAddChecker {
 public:
  AtomicAddCheckerGPU() = default;
  ~AtomicAddCheckerGPU() = default;

 protected:
  bool SuitableForAtomicAdd(const AnfNodePtr &node) override;
};

class AtomicAddCheckerAscend : public AtomicAddChecker {
 public:
  AtomicAddCheckerAscend() = default;
  ~AtomicAddCheckerAscend() = default;

 protected:
  bool SuitableForAtomicAdd(const AnfNodePtr &node) override;
};

class AtomicCleanInserter : public InplaceAssignBuilder {
 public:
  explicit AtomicCleanInserter(const std::string &name = "atomic_clean") : InplaceAssignBuilder(name) {}
  ~AtomicCleanInserter() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  void InsertAtomicClean(const FuncGraphPtr &main_graph, const AnfNodePtr &anf_node,
                         const std::vector<InplaceAssignerInfo> &atomic_add_infos, const FuncGraphManagerPtr &mng);
  void SetTargetAttrs(const CNodePtr &cnode) override {
    SetNodeAttrSafely("enable_atomic_add", MakeValue(true), cnode);
  }
};
using AtomicCleanInserterPtr = std::shared_ptr<AtomicCleanInserter>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADD_ATOMIC_CLEAN_H_
