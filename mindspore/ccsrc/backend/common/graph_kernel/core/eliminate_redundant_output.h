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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_ELIMINATE_REDUNDANT_OUTPUT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_ELIMINATE_REDUNDANT_OUTPUT_H_

#include "include/backend/optimizer/pass.h"

namespace mindspore::graphkernel {
/* Eliminate the output without external user
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0)   // the getitem(1) does not exist.
 *   %3 = op(%2)
 *   graph_kernel:
 *      %1 = TensorAdd(p1, p2)
 *      %2 = Sub(p1, p2)
 *      return make_tuple(%1, %2)
 *   --->
 *   %1 = call @graph_kernel(p1, p2)
 *   %3 = op(%1)                 // if only one output remains, the getitem is not used
 *   graph_kernel:
 *      %1 = TensorAdd(p1, p2)
 *      return %1                // the Sub was eliminated
 */
class EliminateHangingOutput : public opt::Pass {
 public:
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  // update the GetItem(node, i) to GetItem(node, i - offset)
  void UpdateGetitemIndex(const AnfNodePtr &getitem, size_t offset) const;
  AnfNodePtr ReplaceMakeTuple(const AnfNodePtr &node, const AnfNodePtrList &getitems) const;
};

// Remove the output without user or with virtual user (like UpdateState)
class EliminateRedundantOutput : public opt::Pass {
 public:
  EliminateRedundantOutput() : Pass("eliminate_redundant_output") {}
  ~EliminateRedundantOutput() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};

bool IsSideEffectNode(const AnfNodePtr &node);
AnfNodePtrList FindGraphKernelsWithMultiOutput(const FuncGraphPtr &func_graph);

/**
 * @brief Get the GraphKernel's user getitems
 *
 * @param mng FuncGraphManagerPtr for the main func_graph
 * @param node The cnode that indicates the GraphKernel
 * @param getitem_list The user getitem list.
 * @param merge_repeated_getitem If true, getitems with same index will be merged,
 *                               otherwise, only one getitem will be outputted.
 * @return If the graph was changed, returns true, otherwise returns false.
 */
bool GetGraphKernelGetitemList(const FuncGraphManagerPtr &mng, const AnfNodePtr &node, AnfNodePtrList *getitem_list,
                               bool merge_repeated_getitem = false);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_ELIMINATE_REDUNDANT_OUTPUT_H_
