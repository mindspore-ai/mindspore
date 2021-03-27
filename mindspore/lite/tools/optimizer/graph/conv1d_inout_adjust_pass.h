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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_CONV1D_INOUT_ADJUST_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_CONV1D_INOUT_ADJUST_PASS_H_
#include <string>
#include <vector>
#include "backend/optimizer/common/pass.h"
#include "backend/optimizer/common/optimizer.h"
#include "tools/converter/converter_flags.h"

namespace mindspore::opt {
class Conv1DInOutAdjustPass : public Pass {
 public:
  Conv1DInOutAdjustPass() : Pass("conv1d_inout_adjust_pass") {}
  ~Conv1DInOutAdjustPass() override = default;

  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  CNodePtr NewUnsqueezeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node,
                              const std::vector<int64_t> &axis);
  CNodePtr NewSqueezeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node,
                            const std::vector<int64_t> &axis);
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_CONV1D_INOUT_ADJUST_PASS_H_
