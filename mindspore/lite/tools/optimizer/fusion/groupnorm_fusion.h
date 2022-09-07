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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GROUPNORM_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GROUPNORM_FUSION_H_

#include <memory>
#include <string>
#include <map>
#include "schema/inner/model_generated.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
/// fuse layer_norm or instance_norm into one operator
class GroupNormFusion : public LitePatternProcessPass {
 public:
  explicit GroupNormFusion(const std::string &name = "GroupNormFusion", bool multigraph = true)
      : LitePatternProcessPass(name, multigraph) {}

  ~GroupNormFusion() override = default;

 protected:
  bool Init() const;

 private:
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  bool CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, int *num_groups, float *epsilon,
                    bool *affine) const;
  CNodePtr CreateGroupNormNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv, int num_groups, float epsilon,
                               bool affine) const;
  const BaseRef DefinePattern() const override;

 protected:
  mutable VarPtr input_ = nullptr;
  mutable VarPtr mean1_ = nullptr;
  mutable VarPtr mean1_axis_ = nullptr;
  mutable VarPtr sum1_ = nullptr;
  mutable VarPtr sum1_axis_ = nullptr;
  mutable VarPtr gamma_ = nullptr;
  mutable VarPtr beta_ = nullptr;
  mutable VarPtr epsilon_ = nullptr;
  mutable VarPtr reshape1_axis_ = nullptr;
  mutable VarPtr reshape2_axis_ = nullptr;
  mutable VarPtr real_div_divider_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GROUPNORM_FUSION_H_
