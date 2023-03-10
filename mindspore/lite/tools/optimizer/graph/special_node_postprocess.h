/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIAL_NODE_POSTPROCESS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIAL_NODE_POSTPROCESS_H_

#include "include/backend/optimizer/pass.h"

namespace mindspore {
namespace opt {
class SpecialNodePostProcess : public Pass {
 public:
  SpecialNodePostProcess() : Pass("SpecialNodePostProcess") {}
  ~SpecialNodePostProcess() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool CheckInstanceNorm(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  int HandleInstanceNorm(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIAL_NODE_POSTPROCESS_H_
