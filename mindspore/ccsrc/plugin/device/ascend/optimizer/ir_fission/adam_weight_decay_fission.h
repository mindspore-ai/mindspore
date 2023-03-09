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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FISSION_ADAM_WEIGHT_DECAY_FISSION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FISSION_ADAM_WEIGHT_DECAY_FISSION_H_

#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class AdamWeightDecayFission : public PatternProcessPass {
 public:
  explicit AdamWeightDecayFission(bool multi_graph = true)
      : PatternProcessPass("adam_weight_decay_fission", multi_graph) {}
  ~AdamWeightDecayFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FISSION_ADAM_WEIGHT_DECAY_FISSION_H_
