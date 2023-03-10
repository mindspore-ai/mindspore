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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_CONV_PAD_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_CONV_PAD_FUSION_H_

#include <string>
#include <unordered_map>
#include "include/backend/optimizer/optimizer.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore {
namespace opt {
class ConvPadFusion : public MultiplePatternProcessPass {
 public:
  explicit ConvPadFusion(const std::string &name = "ConvPadFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}
  ~ConvPadFusion() override = default;
  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &func_graph, const AnfNodePtr &,
                     const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 private:
  VectorRef DefinePadConvPattern() const;
  VectorRef DefinePadTransposeConvPattern() const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_CONV_PAD_FUSION_H_
