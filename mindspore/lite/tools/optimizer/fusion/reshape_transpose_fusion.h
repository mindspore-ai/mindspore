/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_RESHAPE_TRANSPOSE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_RESHAPE_TRANSPOSE_FUSION_H_

#include <string>
#include <memory>
#include <unordered_map>
#include "include/backend/optimizer/optimizer.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
class ReshapeTransposeFusion : public MultiplePatternProcessPass {
 public:
  explicit ReshapeTransposeFusion(const std::string &name = "ReshapeTransposeFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}
  ~ReshapeTransposeFusion() override = default;

 private:
  VectorRef DefineReshapeTransposePattern() const;
  VectorRef DefineTransposeReshapePattern() const;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr ReshapeTransFusion(const FuncGraphPtr &func_graph, const CNodePtr &transpose) const;
  AnfNodePtr TransReshapeFusion(const FuncGraphPtr &func_graph, const CNodePtr &transpose) const;
  AnfNodePtr TransReshapeTransFusion(const FuncGraphPtr &func_graph, const CNodePtr &transpose) const;
  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_RESHAPE_TRANSPOSE_FUSION_H_
