/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ADD_LAYERNORM_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ADD_LAYERNORM_FUSION_H_

#include <memory>
#include <vector>
#include <string>
#include "include/backend/common/pass_manager/optimizer.h"
#include "primitive/nn_optimizer_ops.h"
#include "primitive/math_ops.h"

namespace mindspore {
namespace opt {
class AddLayernormFusionBase : public PatternProcessPass {
 public:
  AddLayernormFusionBase(std::string name, size_t gamma_idx) : PatternProcessPass(name, true) {
    x1_ = std::make_shared<Var>();
    x2_ = std::make_shared<Var>();
    gamma_ = std::make_shared<Var>();
    beta_ = std::make_shared<Var>();
    begin_norm_axis_ = std::make_shared<Var>();
    begin_params_axis_ = std::make_shared<Var>();
    eps_ = std::make_shared<Var>();
    normalize_shape_ = std::make_shared<Var>();
    gamma_idx_ = gamma_idx;
  }
  ~AddLayernormFusionBase() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  VarPtr x1_;
  VarPtr x2_;
  VarPtr gamma_;
  VarPtr beta_;
  VarPtr begin_norm_axis_;
  VarPtr begin_params_axis_;
  VarPtr eps_;
  VarPtr normalize_shape_;

 private:
  size_t gamma_idx_;
};

class AddLayernormFusion : public AddLayernormFusionBase {
 public:
  AddLayernormFusion() : AddLayernormFusionBase("add_layer_norm_fusion", 1) {}
  ~AddLayernormFusion() override = default;
  const BaseRef DefinePattern() const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class AddLayernormV3Fusion : public AddLayernormFusionBase {
 public:
  AddLayernormV3Fusion() : AddLayernormFusionBase("add_layer_norm_v3_fusion", 1) {}
  ~AddLayernormV3Fusion() override = default;
  const BaseRef DefinePattern() const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

constexpr const size_t kGammaIndex2 = 2;
class AddLayernormExtFusion : public AddLayernormFusionBase {
 public:
  AddLayernormExtFusion() : AddLayernormFusionBase("add_layer_norm_ext_fusion", kGammaIndex2) {}
  ~AddLayernormExtFusion() override = default;
  const BaseRef DefinePattern() const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ADD_LAYERNORM_FUSION_H_
