/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_ALL_TO_ALL_V_FOR_GE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_ALL_TO_ALL_V_FOR_GE_H_

#include <string>
#include <vector>
#include "include/backend/optimizer/optimizer.h"

/* This pass adapts AllToAllv node to GE prototype
 * let rank size is 4:
 *                          I1 I2 I3 I4
 *                          |  |  |  |
 *                         [Reshape * 4]
 *  I1 I2 I3 I4              \  |  |  /
 *  |  |  |  |                [Concat]
 *  [AllToAllv]      ->          |
 *  |  |  |  |             [AllToAllv(for GE)]
 *  O1 O2 O3 O4                  |
 *                            [Split]
 *                            / | | \
 *                         [Reshape * 4]
 *                          |  |  |  |
 *                          O1 O2 O3 O4
 */

namespace mindspore {
namespace opt {
class AllToAllvForGE : public PatternProcessPass {
 public:
  explicit AllToAllvForGE(bool multigraph = true) : PatternProcessPass("all_to_all_v_for_ge", multigraph) {}
  ~AllToAllvForGE() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_ALL_TO_ALL_V_FOR_GE_H_
