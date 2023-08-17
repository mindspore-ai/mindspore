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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUT_AND_OUTPUT_VARIABLE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUT_AND_OUTPUT_VARIABLE_PASS_H_

#include <vector>
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class InputAndOutputVariablePass : public Pass {
 public:
  InputAndOutputVariablePass(std::vector<int64_t> inputs_variable, std::vector<int64_t> outputs_variable)
      : Pass("InputAndOutputVariablePass") {
    inputs_variable_index_ = inputs_variable;
    outputs_variable_index_ = outputs_variable;
  }
  ~InputAndOutputVariablePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  CNodePtr CreateAssign(const AnfNodePtr &anf_node, const ParameterPtr &parameter, const FuncGraphPtr &graph);

  std::vector<int64_t> inputs_variable_index_;
  std::vector<int64_t> outputs_variable_index_;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUT_AND_OUTPUT_VARIABLE_PASS_H_
