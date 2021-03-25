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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ONNX_PAD_ADJUST_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ONNX_PAD_ADJUST_PASS_H_
#include <string>
#include <vector>
#include "backend/optimizer/common/pass.h"
#include "backend/optimizer/common/optimizer.h"

namespace mindspore::opt {
class OnnxPadAdjustPass : public Pass {
 public:
  OnnxPadAdjustPass() : Pass("onnx_pad_adjust") {}
  ~OnnxPadAdjustPass() override = default;

  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  ParameterPtr CreateNewParameter(const FuncGraphPtr &func_graph, const std::vector<int> &data);
  CNodePtr NewReshapeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node, const std::vector<int> &shape);
  CNodePtr NewTransposeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node, std::vector<int> perm);
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ONNX_PAD_ADJUST_PASS_H_
