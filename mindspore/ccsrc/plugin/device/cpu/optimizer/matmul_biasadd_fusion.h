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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_CPU_MATMUL_BIASADD_FUSION_H
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_CPU_MATMUL_BIASADD_FUSION_H

#include <string>
#include <memory>
#include "include/backend/optimizer/optimizer.h"
#include "mindspore/core/ops/math_ops.h"
#include "ir/anf.h"
#include "dnnl.hpp"

namespace mindspore {
namespace opt {
class MatMulBiasAddFusionCPU : public PatternProcessPass {
 public:
  explicit MatMulBiasAddFusionCPU(bool multigraph = true)
      : PatternProcessPass("matmul_biasadd_fusion_cpu", multigraph) {}
  ~MatMulBiasAddFusionCPU() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  AnfNodePtr CreateMatmulWithBias(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const;
  VarPtr x0_ = std::make_shared<Var>();
  VarPtr x1_ = std::make_shared<Var>();
  VarPtr x2_ = std::make_shared<Var>();
  VarPtr matmul_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimMatMul->name()));
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_CPU_MATMUL_BIASADD_FUSION_H
