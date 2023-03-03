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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_TSA_ATOMIC_ADD_TO_FIRST_TENSOR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_TSA_ATOMIC_ADD_TO_FIRST_TENSOR_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/graph_kernel/add_atomic_clean.h"
#include "backend/common/session/kernel_graph.h"

namespace mindspore::graphkernel {
/*
 * output = SubGraph(input_x, indices, update) {
 *   %0 = TensorScatterAdd(%para1, %para2, %para3)
 *   return %0
 * }
 * ---------------------------------------------------------------->
 * // Initialize output with input_x.
 * output = Reshape(input_x)
 * fake_out = SubGraph'(output, indices, update) {
 *   %0 = TensorScatterAdd(%para1, %para2, %para3)
 *   %1 = Assign(%para1, %0, umond) //
 *   return %1
 * }
 */
class TsaAtomicAddToFirstTensor : public AtomicCleanInserter {
 public:
  TsaAtomicAddToFirstTensor() : AtomicCleanInserter("tensor_scatter_add_atomic_add_to_first_tensor") {}
  ~TsaAtomicAddToFirstTensor() override = default;

  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void ProcessOriginalCNode(const AnfNodePtr &composite_node,
                            const std::vector<std::tuple<InplaceAssignerInfo, AnfNodePtr, size_t>> &outer_nodes) const;
  void ChangeKernelBuildInfo(const AnfNodePtr &composite_node,
                             const std::vector<std::tuple<InplaceAssignerInfo, AnfNodePtr, size_t>> &outer_infos) const;
  void ProcessTsa(const KernelGraphPtr &main_graph, const AnfNodePtr &anf_node,
                  const std::vector<InplaceAssignerInfo> &atomic_add_infos, const FuncGraphManagerPtr &mng) const;
  std::pair<AnfNodePtr, size_t> GetOrCreateNewTsaFirstNode(const KernelGraphPtr &main_graph,
                                                           const InplaceAssignerInfo &atomic_add_info,
                                                           const AnfNodePtr &node) const;
  std::pair<AnfNodePtr, size_t> FindTsaFirstRealInputInGraph(const KernelGraphPtr &, const CNodePtr &tsa_node,
                                                             const AnfNodePtr &node) const;

  size_t tsa_first_input_index_{0};  // sub-graph parameter index.
};
using TsaAtomicAddToFirstTensorPtr = std::shared_ptr<TsaAtomicAddToFirstTensor>;
}  // namespace mindspore::graphkernel

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_TSA_ATOMIC_ADD_TO_FIRST_TENSOR_H_
