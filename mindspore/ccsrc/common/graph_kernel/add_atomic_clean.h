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
#include <utility>
#include <vector>
#include <string>
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/session/kernel_graph.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "common/graph_kernel/clean_inserter.h"

namespace mindspore::graphkernel {
struct AtomicAddUserInfo {
  AnfNodePtr clean_node{nullptr};
  AnfNodePtr update_state_node{nullptr};
  AnfNodePtr user_node{nullptr};
  size_t user_input_idx{0};
};

class AtomicAddChecker {
 public:
  AtomicAddChecker() = default;
  virtual ~AtomicAddChecker() = default;
  static std::shared_ptr<AtomicAddChecker> Init();

  bool Check(const AnfNodePtr &node);
  std::vector<CleanZeroUserInfo> GetAtomicAddInfo() { return atomic_add_infos_; }

 protected:
  virtual bool SuitableForAtomicAdd(const AnfNodePtr &) { return false; }
  virtual bool FindCandidate(const AnfNodePtr &anf_node);
  virtual bool CanActivateAtomicAdd(const AnfNodePtr &anf_node);
  std::vector<CleanZeroUserInfo> atomic_add_infos_;
  PrimitivePtr target_type_{prim::kPrimReduceSum};
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

class AtomicCleanInserter : public CleanInserter {
 public:
  explicit AtomicCleanInserter(const std::string &name = "atomic_clean") : CleanInserter(name) {}
  ~AtomicCleanInserter() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  void InsertAtomicClean(const FuncGraphPtr &main_graph, const AnfNodePtr &anf_node,
                         const std::vector<CleanZeroUserInfo> &atomic_add_infos, const FuncGraphManagerPtr &mng);

  void ProcessOriginCNodeUser(const FuncGraphPtr &main_graph, const AnfNodePtr &composite_node,
                              const std::vector<std::pair<CleanZeroUserInfo, AnfNodePtr>> &info_and_broadcast_to_nodes,
                              const FuncGraphManagerPtr &mng);
  void SetTargetAttrs(const CNodePtr &cnode) override {
    SetNodeAttrSafely("enable_atomic_add", MakeValue(true), cnode);
  }

 private:
  std::vector<AtomicAddUserInfo> FindOriginCNodeUsers(
    const FuncGraphPtr &main_graph, const AnfNodePtr &composite_node,
    const std::vector<std::pair<CleanZeroUserInfo, AnfNodePtr>> &info_and_broadcast_to_nodes,
    const FuncGraphManagerPtr &mng) const;
};
using AtomicCleanInserterPtr = std::shared_ptr<AtomicCleanInserter>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADD_ATOMIC_CLEAN_H_
