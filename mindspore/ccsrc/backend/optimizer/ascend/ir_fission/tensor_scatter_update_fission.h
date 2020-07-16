/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FISSION_TENSOR_SCATTER_UPDATE_FISSION_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FISSION_TENSOR_SCATTER_UPDATE_FISSION_H_

#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class TensorScatterUpdateFission : public PatternProcessPass {
 public:
  explicit TensorScatterUpdateFission(bool multigraph = true)
      : PatternProcessPass("tensor_scatter_update_fission", multigraph) {}
  ~TensorScatterUpdateFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FISSION_TENSOR_SCATTER_UPDATE_FISSION_H_
