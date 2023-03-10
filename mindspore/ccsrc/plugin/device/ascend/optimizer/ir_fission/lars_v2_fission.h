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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_LARS_V2_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_LARS_V2_FISSION_H_

#include <vector>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class LarsV2Fission : public PatternProcessPass {
 public:
  explicit LarsV2Fission(bool multigraph = true) : PatternProcessPass("lars_v2_fission", multigraph) {}
  ~LarsV2Fission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  void CreateOutputsOfSquareSumAll(const FuncGraphPtr &graph, const CNodePtr &lars_v2,
                                   std::vector<AnfNodePtr> *square_sum_all_outputs) const;
  CNodePtr CreateLarsV2Update(const FuncGraphPtr &graph, const CNodePtr &lars_v2,
                              const std::vector<AnfNodePtr> &square_sum_all_outputs) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_LARS_V2_FISSION_H_
