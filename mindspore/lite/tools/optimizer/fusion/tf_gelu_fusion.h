/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_GELU_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_GELU_FUSION_H_

#include <memory>
#include <string>
#include <unordered_map>
#include "tools/optimizer/fusion/gelu_fusion.h"

namespace mindspore {
namespace opt {
class TfGeLUFusion : public GeLUFusion {
 public:
  explicit TfGeLUFusion(const std::string &name = "TfGeLUFusion", bool multigraph = true)
      : GeLUFusion(name, multigraph) {}

  ~TfGeLUFusion() override = default;

  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 private:
  bool Init() const;
  bool CheckPattern(const std::string &pattern_name, const EquivPtr &equiv) const override;
  VectorRef DefineFirstStructurePattern() const;

 private:
  mutable VarPtr power_ = nullptr;
  mutable VarPtr power_y_ = nullptr;
  mutable VarPtr mul1_x_ = nullptr;
  mutable VarPtr mul2_x_ = nullptr;
  mutable VarPtr tanh_ = nullptr;
  mutable VarPtr add2_x_ = nullptr;
  mutable VarPtr mul3_x_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TF_GELU_FUSION_H_
