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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GELU_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GELU_FUSION_H_

#include <memory>
#include <string>
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore {
namespace opt {
class GeLUFusion : public MultiplePatternProcessPass {
 public:
  explicit GeLUFusion(const std::string &name = "GeLUFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph), input_(std::make_shared<Var>()) {}

  ~GeLUFusion() override = default;
  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;

 protected:
  virtual bool CheckPattern(const std::string &pattern_name, const EquivPtr &equiv) const = 0;
  const float GetParameterValue(const EquivPtr &equiv, const VarPtr &input) const;

 private:
  CNodePtr CreateGeLUNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;

 protected:
  mutable VarPtr input_{nullptr};
  mutable bool approximate_{false};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GELU_FUSION_H_
