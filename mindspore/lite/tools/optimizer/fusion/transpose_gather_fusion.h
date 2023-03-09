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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TRANSPOSE_GATHER_FUSION_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TRANSPOSE_GATHER_FUSION_H

#include <set>
#include <vector>
#include "include/backend/optimizer/pass.h"

namespace mindspore {
namespace opt {
/*
 * The subgraph such as the following, in some times, the transpose-op can be fused.
 *                            Transpose                  perm(1, 0, 2)
 *                         /      |      \
 *                    Gather    Gather   Gather          axis(0)
 *                      /         |         \
 *               Transpose    Transpose    Transpose     perm(1, 0, 2)
 */
class TransposeGatherFusion : public Pass {
 public:
  TransposeGatherFusion() : Pass("TransposeGatherFusion") {}
  ~TransposeGatherFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  int Process(const FuncGraphPtr &func_graph, const CNodePtr &transpose, std::set<AnfNodePtr> *has_visited);
  void FindNodes(const FuncGraphPtr &func_graph, const CNodePtr &transpose);
  bool CheckCanFused(const CNodePtr &transpose);
  bool CheckCommonAttr(const CNodePtr &transpose);
  bool CheckIsMatch(const std::vector<int> &pre_perm, const std::vector<int> &post_perm, int axis);
  std::vector<CNodePtr> gather_nodes_;
  std::vector<std::vector<CNodePtr>> transpose_nodes_;
  std::vector<int> gather_updated_axes_;
  std::vector<int *> gather_axes_data_ptr_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TRANSPOSE_GATHER_FUSION_H
