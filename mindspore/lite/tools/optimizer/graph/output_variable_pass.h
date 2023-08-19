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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_OUTPUT_VARIABLE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_OUTPUT_VARIABLE_PASS_H_

#include <vector>
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class OutputVariablePass : public Pass {
 public:
  explicit OutputVariablePass(const std::vector<int64_t> &output_variable) : Pass("OutputVariablePass") {
    outputs_variable_index_ = output_variable;
  }
  ~OutputVariablePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  bool CreateDependNode(const FuncGraphPtr &graph);
  std::vector<CNodePtr> assign_nodes_;
  std::vector<int64_t> outputs_variable_index_;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_OUTPUT_VARIABLE_PASS_H_
