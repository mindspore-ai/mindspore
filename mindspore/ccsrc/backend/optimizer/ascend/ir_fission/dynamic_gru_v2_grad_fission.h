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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DYNAMIC_GRU_V2_GRAD_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DYNAMIC_GRU_V2_GRAD_FISSION_H_

#include <vector>
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class DynamicGRUV2GradFission : public PatternProcessPass {
 public:
  explicit DynamicGRUV2GradFission(bool multigraph = true)
      : PatternProcessPass("dynamic_gru_v2_grad_fission", multigraph) {}
  ~DynamicGRUV2GradFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DYNAMIC_GRU_V2_GRAD_FISSION_H_
