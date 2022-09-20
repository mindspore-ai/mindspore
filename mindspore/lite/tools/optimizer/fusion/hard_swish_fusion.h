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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_HARD_SWISH_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_HARD_SWISH_FUSION_H_

#include <memory>
#include <string>
#include "schema/inner/model_generated.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
/// fuse hard swish into one operator
class HardSwishFusion : public LitePatternProcessPass {
 public:
  explicit HardSwishFusion(const std::string &name = "HardSwishFusion", bool multigraph = true)
      : LitePatternProcessPass(name, multigraph) {}

  ~HardSwishFusion() override = default;

 private:
  bool Init() const;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  bool CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv) const;
  const BaseRef DefinePattern() const override;

 protected:
  mutable VarPtr input_ = nullptr;
  mutable VarPtr add_const_ = nullptr;
  mutable VarPtr div_const_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_HARD_SWISH_FUSION_H_
