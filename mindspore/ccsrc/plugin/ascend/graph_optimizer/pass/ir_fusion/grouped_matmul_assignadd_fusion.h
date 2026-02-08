/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_ASCEND_OPTIMIZER_IR_FUSION_GROUPED_MATMUL_ASSIGNADD_FUSION_H_
#define MINDSPORE_CCSRC_PLUGIN_ASCEND_OPTIMIZER_IR_FUSION_GROUPED_MATMUL_ASSIGNADD_FUSION_H_

#include <vector>
#include <string>
#include <memory>
#include "include/backend/common/pass_manager/optimizer.h"
#include "primitive/math_ops.h"

namespace mindspore {
namespace opt {
/* GroupedMatmulAssignaddFusion
 *          input
 *            |
 *    [TransposeExt] weight
 *            |         |
 *     [MakeTuple] [MakeTuple] ... group_list...     input weight group_list Weight
 *               \      |        /                     \      |         |      /
 *                [GroupedMatmul]              ->      [GroupedMatmulInplaceAdd]
 *            (split_item=3, group_type=2)                         |
 *                 |                                             output
 *       out  [TupleGetItem]
 *         \        /
 *        [AssginAdd]
 *             |
 *           output
 */
class GroupedMatmulAssignaddFusion : public PatternProcessPass {
 public:
  explicit GroupedMatmulAssignaddFusion(bool multigraph = true);
  ~GroupedMatmulAssignaddFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
  bool CheckFusion(const CNodePtr &grouped_matmul, const EquivPtr &equiv) const;
  bool CheckDataType(const AnfNodePtr &input_x, const AnfNodePtr &weight, const AnfNodePtr &group_list,
                     const AnfNodePtr &out) const;
  void ReplaceGMMForDepend(const FuncGraphPtr &graph, const CNodePtr &gmm, const CNodePtr &gmm_add) const;

  VarPtr x_;
  VarPtr weight_;
  VarPtr group_list_;
  VarPtr split_item_;
  VarPtr group_type_;
  VarPtr out_;
  VarPtr transpose_a_;
  VarPtr transpose_b_;
  VarPtr grouped_matmul_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_ASCEND_OPTIMIZER_IR_FUSION_GROUPED_MATMUL_ASSIGNADD_FUSION_H_
