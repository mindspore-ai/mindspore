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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_DROPOUT_FOR_GE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_DROPOUT_FOR_GE_H_

#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class DropoutForGE : public PatternProcessPass {
 public:
  explicit DropoutForGE(bool multigraph = true) : PatternProcessPass("dropout_for_ge", multigraph) {}
  ~DropoutForGE() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class DropoutGradForGE : public PatternProcessPass {
 public:
  explicit DropoutGradForGE(bool multigraph = true) : PatternProcessPass("dropout_grad_for_ge", multigraph) {}
  ~DropoutGradForGE() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_OPTIMIZER_IRPASS_DROPOUT_FOR_GE_H_
