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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_MATMUL_SPLIT_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_MATMUL_SPLIT_FUSION_H_

#include <string>
#include <vector>
#include "include/backend/common/pass_manager/optimizer.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_split_base.h"

namespace mindspore {
namespace opt {
class MatmulSplitFusion : public MatmulSplitBase {
 public:
  explicit MatmulSplitFusion(const std::string &pass_name = "matmul_split_fusion", bool multigraph = true)
      : MatmulSplitBase(pass_name, multigraph) {}

  ~MatmulSplitFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
  std::string GetFfnSplitPriName() const override;
  std::string GetQkvSplitPriName() const override;
  void SetMatmulSplitPrimitiveAttr(const PrimitivePtr &matmul_split_prim,
                                   const ValueNodePtr &split_size_node) const override;

  static constexpr auto kMatmulFfnSplitPrimName = "MatmulSplitOut2";
  static constexpr auto kMatmulQkvSplitPrimName = "MatmulSplitOut3";
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_MATMUL_SPLIT_FUSION_H_
