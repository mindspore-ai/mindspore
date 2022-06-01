
/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_UPDATE_STATE_FORMATTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_UPDATE_STATE_FORMATTER_H_

#include <vector>
#include "backend/common/optimizer/pass.h"
#include "ir/func_graph.h"

namespace mindspore::graphkernel {
/**
 * @brief Spread the input tuple of UpdateState
 * @example
 *   %1 = op1
 *   %2 = op2
 *   %3 = make_tuple(%1, %2)
 *   UpdateState(U, %3)
 *   -->
 *   %1 = op1
 *   %2 = op2
 *   UpdateState(U, %1, %2)
 */
class SpreadUpdateState : public opt::Pass {
 public:
  SpreadUpdateState() : Pass("spread_update_state") {}
  ~SpreadUpdateState() override = default;
  AnfNodePtrList ExtendInputsOfUpdateState(const AnfNodePtrList &nodes, const FuncGraphPtr &func_graph) const;
  bool Run(const FuncGraphPtr &func_graph) override;
};

/**
 * @brief Shrink the inputs of UpdateState to a tuple
 * @example
 *   %1 = op1
 *   %2 = op2
 *   UpdateState(U, %1, %2)
 *   -->
 *   %1 = op1
 *   %2 = op2
 *   %3 = make_tuple(%1, %2)
 *   UpdateState(U, %3)
 */
class ShrinkUpdateState : public opt::Pass {
 public:
  ShrinkUpdateState() : Pass("shrink_update_state") {}
  ~ShrinkUpdateState() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};

/**
 * @brief Extend the getitem for UpdateState
 * @example
 *   In this example, the Cast is an output of GraphKernel and only links to an UpdateState,
 *   it has two users in GraphKernel, Add and Sub, which are all outputs.
 *   after processing, the Cast was eliminate from output list and the Add and Sub was linked to UpdateState.
 *
 *   graph_kernel:
 *      %1 = Cast(p1)
 *      %2 = Add(%1, p2)   // depends on Cast
 *      %3 = Sub(%2, p3)   // depends on Cast
 *      %4 = Mul(p1, p2)   // not depends on Cast
 *      return make_tuple(%1, %2, %3, %4)
 *   main graph:
 *      %1 = call @graph_kernel(p1, p2)
 *      %2 = tuple_getitem(%1, 0)  // The Cast
 *      %3 = UpdateState(U, %2)
 *  -->
 *   graph_kernel:
 *      %1 = Cast(p1)
 *      %2 = Add(%1, p2)   // depends on Cast
 *      %3 = Sub(%2, p3)   // depends on Cast
 *      %4 = Mul(p1, p2)   // not depends on Cast
 *      return make_tuple(%2, %3, %4)  // the Cast was eliminated from output list
 *   main graph:
 *      %1 = call @graph_kernel(p1, p2)
 *      %2 = tuple_getitem(%1, 0)   // the Add
 *      %3 = tuple_getitem(%1, 1)   // the Sub
 *      %4 = UpdateState(U, %2, %3)
 */
class ExtendOutputForUpdateState : public opt::Pass {
 public:
  ExtendOutputForUpdateState() : Pass("extend_output_for_update_state") {}
  ~ExtendOutputForUpdateState() = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  // Get the nodes that have external UpdateState user.
  void FindIndexesToUpdateState(const FuncGraphManagerPtr &mng);
  void FilterIndexes(const FuncGraphPtr &func_graph);
  // Find all the func_graph's outputs that depends (directly or indirectly) on the indicated(index) node.
  std::vector<size_t> FindAllOutputs(const FuncGraphPtr &func_graph, size_t index);
  bool ProcessIndex(const FuncGraphPtr &func_graph, const FuncGraphPtr &sub_func_graph, size_t index);

  enum ExternalUserType {
    kNormalOp,     // only has normal operators
    kUpdateState,  // only has UpdateState(s)
    kMix,          // UpdateState mix with normal operator
  };
  AnfNodePtrList getitems_;                           // Users of the GraphKernel nodes.
  std::vector<size_t> indexes_;                       // Indexes of GetItem to be processed.
  std::vector<ExternalUserType> external_user_type_;  // The type of getitem's users.
};

/**
 * @brief Merge UpdateState's inputs which link to the same node
 * @example
 *   graph_kernel:
 *      %1 = Cast(p1)
 *      %2 = Add(%1, p2)
 *      %3 = Sub(%2, p3)
 *      %4 = Mul(p1, p2)
 *      return make_tuple(%1, %2, %3, %4)
 *   main graph:
 *      %1 = call @graph_kernel(p1, p2)
 *      %2 = tuple_getitem(%1, 0)
 *      %3 = tuple_getitem(%1, 1)
 *      %4 = tuple_getitem(%1, 2)
 *      %5 = UpdateState(U, %2, %3, %4)  // the %2 %3 %4 are all link to %1
 * -->
 *   main graph:
 *      %1 = call @graph_kernel(p1, p2)
 *      %2 = tuple_getitem(%1, 0)
 *      %3 = tuple_getitem(%1, 1)
 *      %4 = tuple_getitem(%1, 2)
 *      %5 = UpdateState(U, %2)  // only keep %2
 */
class MergeOutputForUpdateState : public opt::Pass {
 public:
  MergeOutputForUpdateState() : Pass("merge_output_for_update_state") {}
  ~MergeOutputForUpdateState() = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_UPDATE_STATE_FORMATTER_H_
