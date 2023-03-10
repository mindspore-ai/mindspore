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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_QUANT_DTYPE_CAST_ADJUST_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_QUANT_DTYPE_CAST_ADJUST_H_

#include <vector>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
class QuantDTypeCastAdjust : public PatternProcessPass {
 public:
  explicit QuantDTypeCastAdjust(bool multigraph = true) : PatternProcessPass("quant_dtype_cst_adjust", multigraph) {}
  ~QuantDTypeCastAdjust() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_FAKE_LEARNED_SCALE_QUANT_GRAD_UNIFY_MINDIR_H_
