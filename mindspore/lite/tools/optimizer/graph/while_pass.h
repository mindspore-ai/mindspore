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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_WHILE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_WHILE_PASS_H_
#include <string>
#include <vector>
#include "schema/inner/model_generated.h"
#include "tools/converter/converter_flags.h"
#include "backend/optimizer/common/pass.h"
#include "src/param_value_lite.h"

using mindspore::lite::converter::FmkType;
namespace mindspore::opt {
class WhilePass : public Pass {
 public:
  WhilePass() : Pass("while_pass") {}
  ~WhilePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  ValueNodePtr GetSwitchAnfPrim();

  const size_t kWhileMinInputSize = 3;
  const size_t kWhileCondIndex = 1;
  const size_t kWhileBodyIndex = 2;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_SRC_PASS_REMOVE_IDENTITY_PASS_H_
