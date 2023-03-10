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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_SINGLE_BATCH_NORM_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_SINGLE_BATCH_NORM_FISSION_H_

#include <vector>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class SingleBatchNormFission : public PatternProcessPass {
 public:
  explicit SingleBatchNormFission(bool multigraph = true)
      : PatternProcessPass("single_batch_norm_fission", multigraph) {}
  ~SingleBatchNormFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  AnfNodePtr CreateBNTrainingReduce(const FuncGraphPtr &func_graph, const AnfNodePtr &bn) const;
  AnfNodePtr CreateBNTrainingUpdateV3(const FuncGraphPtr &func_graph, const AnfNodePtr &bn,
                                      const std::vector<AnfNodePtr> &bn_training_reduce_outputs) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_SINGLE_BATCH_NORM_FISSION_H_
