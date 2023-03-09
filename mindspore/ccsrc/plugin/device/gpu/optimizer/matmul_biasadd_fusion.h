/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_MATMUL_BIASADD_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_MATMUL_BIASADD_FUSION_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class MatMulBiasAddFusion : public PatternProcessPass {
 public:
  explicit MatMulBiasAddFusion(bool multigraph = true) : PatternProcessPass("matmul_biasadd_fusion", multigraph) {
    u_ = std::make_shared<Var>();
    x_ = std::make_shared<Var>();
    w_ = std::make_shared<Var>();
    bias_ = std::make_shared<Var>();
  }
  ~MatMulBiasAddFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  VarPtr u_;
  VarPtr x_;
  VarPtr w_;
  VarPtr bias_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_MATMUL_BIASADD_FUSION_H_
