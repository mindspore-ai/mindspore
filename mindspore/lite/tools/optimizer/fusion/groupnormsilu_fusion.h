/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GROUPNORMSILU_BASE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GROUPNORMSILU_BASE_FUSION_H_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
namespace mindspore {
namespace opt {
class GroupNormSiluFusion : public MultiplePatternProcessPass {
 public:
  explicit GroupNormSiluFusion(const std::string &name = "GroupNormSiluFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~GroupNormSiluFusion() override = default;

  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr Process(const std::string &, const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr ReshapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;

  CNodePtr CreateGroupNormSiluNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &conv,
                                   const AnfNodePtr &gamma_3D, const AnfNodePtr &beta_3D, int64_t num_groups) const;

  CNodePtr CreateGroupNormSiluNodeForSD15(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                          const AnfNodePtr &node, const EquivPtr &equiv) const;

  CNodePtr CreateGroupNormSiluNodeForSDWithCast(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                                const AnfNodePtr &node, const EquivPtr &equiv) const;

  /*
   * --------------------------------------------------
   *  Pattern GroupNormSiluPatternForSD15:            |
   *                            reshape               |
   *                            instanceNormalization |
   *                            reshape               |
   *                            mul                   |
   *                        --- add                   |
   *                        |   sigmoid               |
   *                        --> mul                   |
   * --------------------------------------------------
   */
  const VectorRef DefineGroupNormSiluPatternForSD15() const;

  /*
   * --------------------------------------------------
   *  Pattern WithCast:                               |
   *                            cast                  |
   *                            reshape               |
   *                            instanceNormalization |
   *                            reshape               |
   *                            mul                   |
   *                            add                   |
   *                        --- cast                  |
   *                        |   sigmoid               |
   *                        --> mul                   |
   * --------------------------------------------------
   */
  const VectorRef DefineGroupNormSiluPatternForSDWithCast() const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GROUPNORMSILU_BASE_FUSION_H_
