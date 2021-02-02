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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADD_STITCH_ATOMIC_CLEAN_GPU_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADD_STITCH_ATOMIC_CLEAN_GPU_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/graph_kernel/add_atomic_clean_gpu.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace opt {
class StitchAtomicCleanInsertter : public AtomicCleanInsertter {
 public:
  StitchAtomicCleanInsertter() : AtomicCleanInsertter("stitch_atomic_clean") {}
  ~StitchAtomicCleanInsertter() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void CorrectKernelBuildInfo(const AnfNodePtr &composite_node, const AnfNodePtr &new_input);
  CNodePtr CreateInplaceAssignNodeAndCorrectReturn(const FuncGraphPtr &sub_graph, const AnfNodePtr &new_parameter);
  std::vector<std::pair<AnfNodePtr, int>> FindInnerCNodeUsers(const AnfNodePtr &inner_node, const CNodePtr &target);
  void ProcessOriginCNode(const KernelGraphPtr &main_graph, const AnfNodePtr &composite_node,
                          const AnfNodePtr &new_input, const FuncGraphManagerPtr &mng);
  bool IsStitchWithAtomic(const AnfNodePtr &anf_node);

  AnfNodePtr stitch_node_{nullptr};
};
using StitchAtomicCleanInsertterPtr = std::shared_ptr<StitchAtomicCleanInsertter>;
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADD_STITCH_ATOMIC_CLEAN_GPU_H_
