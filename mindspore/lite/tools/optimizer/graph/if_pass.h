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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_IF_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_IF_PASS_H_
#include <string>
#include <vector>
#include "schema/inner/model_generated.h"
#include "tools/converter/converter_flags.h"
#include "backend/optimizer/common/pass.h"
#include "src/param_value_lite.h"

using mindspore::lite::converter::FmkType;
namespace mindspore::opt {
class IfPass : public Pass {
 public:
  IfPass() : Pass("if_pass") {}
  ~IfPass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  void ReplaceInput(const std::vector<AnfNodePtr> &node_list, AnfNodePtr new_input_cnode, std::string para_name);
  ValueNodePtr GetSwitchAnfPrim();

  const size_t kIfMinInputSize = 4;
  const size_t kIfThenIndex = 1;
  const size_t kIfElseIndex = 2;
  const size_t kIfCondIndex = 3;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_IF_PASS_H_
