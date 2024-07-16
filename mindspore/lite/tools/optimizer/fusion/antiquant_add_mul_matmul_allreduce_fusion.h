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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ANTIQUANT_ADD_MUL_MATMUL_ALLREDUCE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ANTIQUANT_ADD_MUL_MATMUL_ALLREDUCE_FUSION_H_

#include <string>
#include <unordered_map>
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore::opt {

class AntiquantAddMulMatMulAllReduceFusion : public MultiplePatternProcessPass {
 public:
  explicit AntiquantAddMulMatMulAllReduceFusion(bool multigraph = true,
                                                const std::string &name = "AntiquantAddMulMatMulAllReduceFusion")
      : MultiplePatternProcessPass(name, multigraph) {}
  ~AntiquantAddMulMatMulAllReduceFusion() override = default;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;
  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;

 private:
  VectorRef DefineAntiquantAddMulMatMulAllReducePattern() const;
  CNodePtr CreateAntiquantAddMulMatMulAllReduceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;

 protected:
  const std::string kPatternNameAntiquantAddMulMatMulAllReduce = "AntiquantAddMulMatMulAllReduce";
  const std::string kAttrNameGroup = "group";
  const std::string kAttrNameFusion = "fusion";
  const std::string kAttrNameOp = "op";
  const std::string kAttrNameTransposeA = "transpose_a";
  const std::string kAttrNameTransposeB = "transpose_b";
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ANTIQUANT_ADD_MUL_MATMUL_ALLREDUCE_FUSION_H_
