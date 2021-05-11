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

#include "ir/anf.h"
#include "mindspore/ccsrc/backend/optimizer/common/node_pass.h"

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FISSON_ITER_NODE_OUTPUTS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FISSON_ITER_NODE_OUTPUTS_H_

namespace mindspore {
namespace opt {
class IterNodeOutputs : public opt::NodePass {
 public:
  IterNodeOutputs() : NodePass("iter_node_outputs") {}
  ~IterNodeOutputs() override = default;
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FISSON_ITER_NODE_OUTPUTS_H_
