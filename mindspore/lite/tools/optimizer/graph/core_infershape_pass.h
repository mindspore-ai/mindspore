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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_CORE_INFERSHAPE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_CORE_INFERSHAPE_PASS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "include/backend/optimizer/pass.h"
#include "include/errorcode.h"
#include "include/registry/converter_context.h"

namespace mindspore {
namespace opt {
using mindspore::lite::STATUS;
class CoreInferShapePass : public Pass {
 public:
  explicit CoreInferShapePass(converter::FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false,
                              const std::string &name = "CoreInferShapePass")
      : Pass(name), fmk_type_(fmk_type), train_flag_(train_flag) {}
  ~CoreInferShapePass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  STATUS InferProcess(const FuncGraphPtr &func_graph);
  STATUS InferProcessSubGraph(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  STATUS SetSubGraphInput(const CNodePtr &cnode, const FuncGraphPtr &sub_graph);
  STATUS SetSubGraphOutput(const FuncGraphPtr &sub_graph);
  STATUS SetSubGraphAbstract(const CNodePtr &cnode, const FuncGraphPtr &sub_graph);
  int ResetSubGraphInput();

 protected:
  converter::FmkType fmk_type_{converter::kFmkTypeMs};
  bool train_flag_{false};
  std::map<FuncGraphPtr, std::vector<AnfNodePtr>> sub_inputs_map_{};
  FuncGraphManagerPtr manager_{nullptr};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_CORE_INFERSHAPE_PASS_H_
