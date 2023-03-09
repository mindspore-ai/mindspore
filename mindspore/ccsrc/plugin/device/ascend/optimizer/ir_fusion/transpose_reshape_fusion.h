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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_TRANSPOSE_RESHAPE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_TRANSPOSE_RESHAPE_FUSION_H_

#include <string>
#include <memory>
#include "ir/anf.h"
#include "include/backend/optimizer/pattern_engine.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/ascend_pass_control.h"

namespace mindspore {
namespace opt {
class TransposeReshapeFusion : public PatternProcessPassWithSwitch {
 public:
  explicit TransposeReshapeFusion(bool multigraph = true)
      : PatternProcessPassWithSwitch("transpose_reshape_fusion", multigraph) {
    input_varptr_ = std::make_shared<Var>();
    PassSwitchManager::GetInstance().RegistLicPass(name(), OptPassEnum::TransposeReshapeFusion);
  }
  ~TransposeReshapeFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  VarPtr input_varptr_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_TRANSPOSE_RESHAPE_FUSION_H_
