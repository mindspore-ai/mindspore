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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SWIGLU_DYNAMIC_QUANT_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SWIGLU_DYNAMIC_QUANT_FUSION_H_
#include <vector>
#include <string>
#include <memory>
#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
class SwiGLUDynamicQuantFusion : public PatternProcessPass {
 public:
  explicit SwiGLUDynamicQuantFusion(bool multigraph = true)
      : PatternProcessPass("swiglu_dynamic_quant_fusion", multigraph) {
    x_ = std::make_shared<Var>();
    axis_ = std::make_shared<Var>();
    smooth_scale_ = std::make_shared<Var>();
  }
  ~SwiGLUDynamicQuantFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateSwiGLUDynamicQuantNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &equiv) const;
  std::vector<std::string> MustExistPrimitiveName() const override;
  VarPtr x_;
  VarPtr axis_;
  VarPtr smooth_scale_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SWIGLU_DYNAMIC_QUANT_FUSION_H_
