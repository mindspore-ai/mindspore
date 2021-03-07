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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ONNX_INPUTS_ADJUST_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ONNX_INPUTS_ADJUST_PASS_H_
#include <string>
#include <vector>
#include "backend/optimizer/common/pass.h"
#include "tools/converter/converter_flags.h"
#include "tools/optimizer/common/gllo_utils.h"

using mindspore::lite::converter::FmkType;
namespace mindspore::opt {
class OnnxInputAdjustOpPass : public Pass {
 public:
  OnnxInputAdjustOpPass() : Pass("onnx_input_adjust") {}
  ~OnnxInputAdjustOpPass() override = default;
  static STATUS ReplaceInt64ParameterNode(const FuncGraphPtr &func_graph, const ParameterPtr &param_node);
  static STATUS ReplaceConstant(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  static STATUS ReplaceTransposeWithGraphInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  static STATUS AddAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int input_num,
                               const std::string &attr_name);
  static STATUS AdjustStridedSlice(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  STATUS AdjustResize(const CNodePtr &cnode);
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_ONNX_INPUTS_ADJUST_PASS_H_
