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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_ACTIVATION_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_ACTIVATION_FUSION_H_

#include <string>
#include <set>
#include "schema/inner/model_generated.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
class AddActivationFusion : public LitePatternProcessPass {
 public:
  explicit AddActivationFusion(const std::string &name = "AddActivationFusion", bool multigraph = true)
      : LitePatternProcessPass(name, multigraph) {}

  ~AddActivationFusion() = default;

  const BaseRef DefinePattern() const override;

  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  bool CheckPattern(const FuncGraphPtr &func_graph, const CNodePtr &act_cnode,
                    const std::set<int64_t> support_act_types) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_ACTIVATION_FUSION_H_
