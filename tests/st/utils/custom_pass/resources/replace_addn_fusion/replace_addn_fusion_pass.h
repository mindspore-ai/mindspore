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
#ifndef MINDSPORE_CUSTOM_PASS_REPLACE_ADDN_FUSION_PASS_H_
#define MINDSPORE_CUSTOM_PASS_REPLACE_ADDN_FUSION_PASS_H_

#include "mindspore/include/custom_pass_api.h"

namespace mindspore {
namespace opt {

/**
 * @brief Pass to replace AddN with Add (MindSpore version)
 *
 * When AddN has only two inputs, replace it with Add operation
 * Inherits from PatternToPatternPass, conforms to MindSpore plugin system requirements
 */
class ReplaceAddNFusionPass : public PatternToPatternPass {
 public:
  ReplaceAddNFusionPass() : PatternToPatternPass("ReplaceAddNFusionPass") {}

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &pattern_map, const FuncGraphPtr &func_graph,
                       const AnfNodePtr &node) const override;

 private:
  // Helper functions
  static bool IsAddNNode(const AnfNodePtr &node);
  static int GetAddNInputCount(const AnfNodePtr &node);
  static bool IsUInt32Output(const AnfNodePtr &node);

  // Build function
  static AnfNodePtr BuildAdd(const PatternMap &m, const AnfNodePtr &default_node);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CUSTOM_PASS_REPLACE_ADDN_FUSION_PASS_H_
