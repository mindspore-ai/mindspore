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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_RMSNORM_QUANT_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_RMSNORM_QUANT_FUSION_H_

#include <vector>
#include <string>
#include <memory>
#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
class RmsNormQuantFusion : public PatternProcessPass {
 public:
  explicit RmsNormQuantFusion(bool multigraph = true) : PatternProcessPass("rms_norm_quant_fusion", multigraph) {
    x1_ = std::make_shared<Var>();
    gamma_ = std::make_shared<Var>();
    eps_ = std::make_shared<Var>();
    scale0_ = std::make_shared<Var>();
    offset0_ = std::make_shared<Var>();
  }
  ~RmsNormQuantFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;

  VarPtr x1_;
  VarPtr gamma_;
  VarPtr eps_;
  VarPtr scale0_;
  VarPtr offset0_;
};

class RmsNormAddQuantFusion : public PatternProcessPass {
 public:
  explicit RmsNormAddQuantFusion(bool multigraph = true) : PatternProcessPass("rms_norm_add_quant_fusion", multigraph) {
    x1_ = std::make_shared<Var>();
    gamma_ = std::make_shared<Var>();
    eps_ = std::make_shared<Var>();
    beta0_ = std::make_shared<Var>();
    scale0_ = std::make_shared<Var>();
    offset0_ = std::make_shared<Var>();
  }
  ~RmsNormAddQuantFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
  const AnfNodePtr RmsNormQuantFuseWithOnePath(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &,
                                               const AnfNodePtr &shape_node) const;
  const AnfNodePtr RmsNormQuantFuseWithTwoPath(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &,
                                               const AnfNodePtr &, const AnfNodePtr &) const;

  VarPtr x1_;
  VarPtr gamma_;
  VarPtr eps_;
  VarPtr beta0_;
  VarPtr scale0_;
  VarPtr offset0_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_RMSNORM_QUANT_FUSION_H_
