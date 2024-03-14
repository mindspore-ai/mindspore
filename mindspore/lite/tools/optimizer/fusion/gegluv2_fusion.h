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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GEGLUV2_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GEGLUV2_FUSION_H_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
namespace mindspore {
namespace opt {
class GeGluV2Fusion : public MultiplePatternProcessPass {
 public:
  explicit GeGluV2Fusion(const std::string &name = "GeGluV2Fusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~GeGluV2Fusion() override = default;

  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr Process(const std::string &, const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateGeGluV2Node(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                             const AnfNodePtr &add_output) const;

  //           +-+
  //    +------+x+-----+
  //    |      +-+     |
  //    |              |
  // +--v--+        +--v--+
  // |slice|        |slice|
  // +--+--+        +--+--+
  //    |              |
  //    |            +-v--+
  //    |            |gelu|
  //    |     +---+  +-+--+
  //    +----->mul<--+-+
  //          +-+-+
  //            |
  //         +--v---+
  //         |output|
  //         +------+
  // gelu(x) = x (1/2 [1 + erf(x / sqrt(2))])
  const VectorRef DefineGeGluV2Pattern() const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_GEGLUV2_FUSION_H_
