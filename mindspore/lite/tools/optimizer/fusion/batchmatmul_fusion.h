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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_BATCHMATMUL_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_BATCHMATMUL_FUSION_H_

#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "tools/converter/converter_context.h"

namespace mindspore {
namespace opt {
class BatchMatMulFusion : public LitePatternProcessPass {
 public:
  explicit BatchMatMulFusion(bool multigraph = true) : LitePatternProcessPass("BatchMatMulFusion", multigraph) {}
  ~BatchMatMulFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  bool CheckCnodeProper(const CNodePtr &stack_cnode, const CNodePtr &fullconnect_cnode,
                        const CNodePtr &left_slice_cnode) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_BATCHMATMUL_FUSION_H_
