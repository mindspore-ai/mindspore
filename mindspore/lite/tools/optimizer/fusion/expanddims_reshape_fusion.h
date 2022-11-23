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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_EXPANDDIMS_RESHAPE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_EXPANDDIMS_RESHAPE_FUSION_H_

#include <string>
#include <memory>
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
class ExpandDimsReshapeFusion : public LitePatternProcessPass {
 public:
  explicit ExpandDimsReshapeFusion(bool multigraph = true, const std::string &name = "ExpandDimsReshapeFusion")
      : LitePatternProcessPass(name, multigraph) {}
  ~ExpandDimsReshapeFusion() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  const BaseRef DefinePattern() const override;

 private:
  bool CheckCanFuse(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_EXPANDDIMS_RESHAPE_FUSION_H_
