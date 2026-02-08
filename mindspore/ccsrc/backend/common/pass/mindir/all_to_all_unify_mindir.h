/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ALL_TO_ALL_UNIFY_MINDIR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ALL_TO_ALL_UNIFY_MINDIR_H_

#include <memory>
#include <string>
#include <vector>
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT NeighborExchangeUnifyMindIR : public PatternProcessPass {
 public:
  explicit NeighborExchangeUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("neighbor_exchange_unify_mindir", multigraph) {}
  ~NeighborExchangeUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateAlltoAllVNode(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange) const;
  std::vector<std::string> MustExistPrimitiveName() const override;
};

/* AllToAllUnifyMindIR
 * let rank size is 4, for ge:
 *                        Input
 *                          |
 *     Input         [Split(split_dim)]
 *       |               / | | \
 *   [AlltoAll]  ->     [AlltoAllV]
 *       |               \  |  |  /
 *     Output        [Concat(concat_dim)]
 *                           |
 *                         Output
 * for kbk:
 *                        Input
 *                          |
 *                   [Split(split_dim)]
 *                       / | | \
 *     Input          [Concat(dim 0)]
 *       |                  |
 *   [AlltoAll]  ->     [AllToAll]
 *       |                  |
 *     Output          [Split(dim 0)]
 *                       \  |  |  /
 *                   [Concat(concat_dim)]
 *                           |
 *                         Output
 */
class BACKEND_COMMON_EXPORT AllToAllUnifyMindIR : public PatternProcessPass {
 public:
  explicit AllToAllUnifyMindIR(bool multigraph = true) : PatternProcessPass("all_to_all_unify_mindir", multigraph) {}
  ~AllToAllUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  const CNodePtr CreateTransposeNode(const KernelGraphPtr &graph, const AnfNodePtr &input_node, ShapeVector shape,
                                     ShapeVector dims) const;
  const CNodePtr CreateReshapeNode(const KernelGraphPtr &, const AnfNodePtr &, const ShapeVector &) const;
  CNodePtr CreateSplitNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all, const AnfNodePtr &input_node,
                           int64_t split_count, int64_t split_dim) const;
  CNodePtr CreateSplitNodeWithSplitDim(const KernelGraphPtr &graph, const CNodePtr &all_to_all) const;
  CNodePtr CreateSplitNodeWithDim0(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                   const CNodePtr &input_node) const;
  CNodePtr CreateAlltoAllVNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all, const CNodePtr &split) const;
  CNodePtr CreateAllToAllNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                              const AnfNodePtr &all_to_all_input) const;
  CNodePtr CreateConcatNode(const KernelGraphPtr &graph, const CNodePtr &all_to_all, const CNodePtr &input_node,
                            int64_t split_count, int64_t concat_dim) const;
  CNodePtr CreateConcatNodeWithConcatDim(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                         const CNodePtr &input_node) const;
  CNodePtr CreateConcatNodeWithDim0(const KernelGraphPtr &graph, const CNodePtr &all_to_all,
                                    const CNodePtr &input_node) const;
  CNodePtr CreateAntecedentTransposeNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &all_to_all_input,
                                         const ShapeVector &shape, int64_t split_count, size_t split_idx) const;
  CNodePtr CreateSuccessorTransposeNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &new_ata,
                                        const ShapeVector &out_shape, int64_t split_count, size_t split_idx,
                                        size_t concat_idx) const;
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ALL_TO_ALL_UNIFY_MINDIR_H_
