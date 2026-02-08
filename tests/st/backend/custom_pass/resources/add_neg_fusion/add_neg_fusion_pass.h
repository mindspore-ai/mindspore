/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CUSTOM_PASS_ADD_NEG_FUSION_PASS_H_
#define MINDSPORE_CUSTOM_PASS_ADD_NEG_FUSION_PASS_H_

#include "mindspore/include/custom_pass_api.h"

namespace mindspore {
namespace opt {

/**
 * @brief Pass to fuse Add and Neg operations into Sub
 *
 * Transforms Add(x, Neg(y)) into Sub(x, y)
 * This is a standard algebraic optimization that eliminates unnecessary Neg operations
 * Works on CPU/GPU/Ascend since all platforms support Add, Neg, and Sub operations
 * Inherits from PatternToPatternPass to comply with MindSpore plugin system requirements
 */
class AddNegFusionPass : public PatternToPatternPass {
 public:
  AddNegFusionPass() : PatternToPatternPass("AddNegFusionPass") {}

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &pattern_map, const FuncGraphPtr &func_graph,
                       const AnfNodePtr &node) const override;

 private:
  static bool IsAddNode(const AnfNodePtr &node);
  static bool IsNegNode(const AnfNodePtr &node);

  static AnfNodePtr BuildSub(const PatternMap &m, const AnfNodePtr &default_node);
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CUSTOM_PASS_ADD_NEG_FUSION_PASS_H_
