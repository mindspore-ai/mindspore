/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MATMUL_ALLREDUCE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MATMUL_ALLREDUCE_FUSION_H_

#include <vector>
#include <string>
#include <memory>
#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
class MatMulAllReduceFusion : public PatternProcessPass {
 public:
  explicit MatMulAllReduceFusion(bool multigraph = true, const string &pass_name = "MatMulAllReduce")
      : PatternProcessPass(pass_name, multigraph) {}
  ~MatMulAllReduceFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 private:
  virtual AnfNodePtr CreateMatMulAllReduceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  PrimitivePtr CreateMatMulAllReducePrim(const PrimitivePtr &allreduce_prim, const CNodePtr &matmul_node) const;
  std::vector<std::string> MustExistPrimitiveName() const override;

 protected:
  const std::string kAttrNameGroup = "group";
  const std::string kAttrNameFusion = "fusion";
  const std::string kAttrNameOp = "op";
  const std::string kAttrNameTransposeA = "transpose_a";
  const std::string kAttrNameTransposeB = "transpose_b";
  const std::string kPhaseNamePrefill = "prefill";
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_MATMUL_ALLREDUCE_FUSION_H_
