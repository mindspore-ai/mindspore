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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_MATMUL_SIGMOID_CAST_ADD_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_MATMUL_SIGMOID_CAST_ADD_FUSION_H_

#include <memory>
#include <vector>
#include <string>
#include <map>
#include "include/backend/common/pass_manager/optimizer.h"
#include "primitive/math_ops.h"

namespace mindspore {
namespace opt {
class MatMulSigmoidCastAddFusion : public PatternProcessPass {
 public:
  explicit MatMulSigmoidCastAddFusion(bool multigraph = true,
                                      const string &pass_name = "matmul_sigmoid_cast_add_fusion")
      : PatternProcessPass(pass_name, multigraph) {}
  ~MatMulSigmoidCastAddFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
  const std::string kPhaseNamePrefill = "prefill";
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_MATMUL_SIGMOID_CAST_ADD_FUSION_H_
