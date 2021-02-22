/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_LAYER_NORM_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_LAYER_NORM_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include "schema/inner/model_generated.h"
#include "backend/optimizer/common/optimizer.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {

/// fuse layer_norm, instance_norm into one operator
class TfNormFusion : public PatternProcessPass {
 public:
  explicit TfNormFusion(const std::string &name = "tf_norm_fusion", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {
    input_ = std::make_shared<Var>();
    mean1_ = std::make_shared<Var>();
    mean1_axes_ = std::make_shared<Var>();
    mean2_ = std::make_shared<Var>();
    mean2_axes_ = std::make_shared<Var>();
    gamma_ = std::make_shared<Var>();
    beta_ = std::make_shared<Var>();
    epsilon_ = std::make_shared<Var>();
  }

  ~TfNormFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  bool GetNormTypeAndAxis(const CNodePtr &input_cnode, const std::vector<int> &mean_axes,
                          const std::vector<int> &params_shape, schema::PrimitiveType *type, int *begin_norm_axis,
                          int *begin_params_axis) const;
  bool CheckPattern(const EquivPtr &equiv, schema::PrimitiveType *type, float *epsilon, int *begin_norm_axis,
                    int *begin_params_axis) const;
  CNodePtr CreateNormNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv, const schema::PrimitiveType type,
                          float epsilon, int begin_norm_axis, int begin_params_axis) const;

 protected:
  VarPtr input_ = nullptr;
  VarPtr mean1_ = nullptr;
  VarPtr mean1_axes_ = nullptr;
  VarPtr mean2_ = nullptr;
  VarPtr mean2_axes_ = nullptr;
  VarPtr gamma_ = nullptr;
  VarPtr beta_ = nullptr;
  VarPtr epsilon_ = nullptr;
};

inline bool IsAddNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimAddFusion);
  }
  return false;
}

inline bool IsSquaredDifferenceNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimSquaredDifference);
  }
  return false;
}

inline bool IsRsqrtNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimRsqrt);
  }
  return false;
}

inline bool IsMulNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimMulFusion);
  }
  return false;
}

inline bool IsSubNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimSubFusion);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_LAYER_NORM_FUSION_H_
