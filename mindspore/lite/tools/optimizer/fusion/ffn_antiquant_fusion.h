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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_ANTIQUANT_FUSION_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_ANTIQUANT_FUSION_H

#include <vector>
#include "include/backend/optimizer/pass.h"

namespace mindspore {
namespace opt {
class FFNAntiquantFusion : public Pass {
 public:
  FFNAntiquantFusion() : Pass("FFNAntiquantFusion") {}
  ~FFNAntiquantFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  int Process(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  CNodePtr NewFFNCNodeWithAntiquantFusion(const FuncGraphPtr &func_graph, const CNodePtr &ffn_cnode,
                                          const ParameterPtr weight1_scale_node, const ParameterPtr weight2_scale_node,
                                          const ParameterPtr weight1_zp_node, const ParameterPtr weight2_zp_node);
  ParameterPtr RepeatParameter(const FuncGraphPtr &func_graph, const ParameterPtr weight1_scale_node, int repeat_times);
  int GetExpertNumFromAntiQuantModeNodes(const AnfNodePtr &node);
  int SetInStragety(const CNodePtr &ffn_cnode, const CNodePtr &ffn_fusion_cnode);
  std::vector<int64_t> GetScaleZpInStragety(std::vector<int64_t> weight_in_strategy);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_ANTIQUANT_FUSION_H
