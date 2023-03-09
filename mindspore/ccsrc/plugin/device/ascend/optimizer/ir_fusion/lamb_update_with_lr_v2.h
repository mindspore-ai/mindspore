/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_UPDATE_WITH_LR_V2_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_UPDATE_WITH_LR_V2_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "utils/hash_map.h"
#include "ir/anf.h"
#include "include/backend/optimizer/pattern_engine.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class LambUpdateWithLrV2 : public PatternProcessPass {
 public:
  explicit LambUpdateWithLrV2(bool multigraph = true) : PatternProcessPass("lamb_update_with_lr_v2", multigraph) {
    for (size_t i = 0; i < kLambUpdateWithLrV2InputNum - 1; ++i) {
      input_varptr_.push_back(std::make_shared<Var>());
    }
  }
  ~LambUpdateWithLrV2() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const EquivPtr &equiv) const override;

 private:
  std::vector<VarPtr> input_varptr_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_UPDATE_WITH_LR_V2_H_
