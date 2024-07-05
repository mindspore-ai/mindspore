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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_SPLIT_CONCAT_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_SPLIT_CONCAT_FUSION_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BACKEND_EXPORT SplitConcatFusion : public PatternProcessPass {
 public:
  explicit SplitConcatFusion(bool multigraph = true) : PatternProcessPass("split_concat_fusion", multigraph) {
    x1_ = std::make_shared<Var>();
  }
  ~SplitConcatFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const EquivPtr &equiv) const override;

 private:
  VarPtr x1_;
  mutable VarPtr split_axis_;
  mutable VarPtr output_num_;
  mutable VarPtr concat_axis_;
  mutable uint32_t global_rank_size_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_SPLIT_CONCAT_FUSION_H_
