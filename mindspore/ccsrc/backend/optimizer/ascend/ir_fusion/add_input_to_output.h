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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_ADD_INPUT_TO_OUTPUT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_ADD_INPUT_TO_OUTPUT_H_

#include <string>
#include <memory>
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/ascend/ascend_helper.h"

namespace mindspore {
namespace opt {
class AddInputToOutput : public PatternProcessPass {
 public:
  explicit AddInputToOutput(bool multigraph = true)
      : PatternProcessPass("add_input_to_output", multigraph), op_finder_(std::make_shared<OpFinder>()) {}
  ~AddInputToOutput() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  OpFinderPtr op_finder_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_ADD_INPUT_TO_OUTPUT_H_
