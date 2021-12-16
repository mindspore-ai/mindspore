/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/common/optimizer.h"
#include "backend/session/kernel_graph.h"

namespace mindspore::graphkernel {
struct AtomicAddInfo {
  CNodePtr atomic_add_node{nullptr};
  size_t reduce_real_output_index{0};
  size_t real_output_num{0};
};

class AtomicAddChecker {
 public:
  AtomicAddChecker() = default;
  virtual ~AtomicAddChecker() = default;
  static std::shared_ptr<AtomicAddChecker> Init();

  bool Check(const AnfNodePtr &node);
  std::vector<AtomicAddInfo> GetAtomicAddInfo() { return atomic_add_infos_; }

 protected:
  virtual bool SuitableForAtomicAdd(const AnfNodePtr &node) { return false; }
  virtual bool FindCandidate(const AnfNodePtr &anf_node);
  virtual bool CanActivateAtomicAdd(const AnfNodePtr &anf_node);
  std::vector<AtomicAddInfo> atomic_add_infos_;
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

class AtomicCleanInsertter : public opt::Pass {
 public:
  explicit AtomicCleanInsertter(const std::string &name = "atomic_clean") : Pass(name) {}
  ~AtomicCleanInsertter() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  virtual void CorrectKernelBuildInfo(const AnfNodePtr &composite_node,
                                      const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &clean_infos);
  virtual void ProcessOriginCNode(const AnfNodePtr &composite_node,
                                  const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &info_and_broadcast_to_nodes);
  virtual CNodePtr CreateAtomicCleanCompositeNode(const AtomicAddInfo &atomic_add_info,
                                                  const KernelGraphPtr &main_graph, TypeId dst_type);
  void InsertAtomicClean(const KernelGraphPtr &main_graph, const AnfNodePtr &anf_node,
                         const std::vector<AtomicAddInfo> &atomic_add_infos, const FuncGraphManagerPtr &mng);
  CNodePtr InsertUpdateState(const KernelGraphPtr &main_graph, const CNodePtr &composite_node) const;
  void CorrectAbstract(const AnfNodePtr &composite_node,
                       const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &process_infos) const;
  void CreateInplaceAssignNodeAndCorrectReturn(
    const FuncGraphPtr &sub_graph, const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &parameters_infos);
  void ProcessOriginCNodeUser(const KernelGraphPtr &main_graph, const AnfNodePtr &composite_node,
                              const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &info_and_broadcast_to_nodes,
                              const AnfNodePtr &update_state_node, const FuncGraphManagerPtr &mng);

 private:
  std::vector<std::tuple<AnfNodePtr, int, AnfNodePtr>> FindOriginCNodeUsers(
    const KernelGraphPtr &main_graph, const AnfNodePtr &composite_node,
    const std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> &info_and_broadcast_to_nodes,
    const FuncGraphManagerPtr &mng, bool correct_index) const;
};
using AtomicCleanInsertterPtr = std::shared_ptr<AtomicCleanInsertter>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADD_ATOMIC_CLEAN_H_
