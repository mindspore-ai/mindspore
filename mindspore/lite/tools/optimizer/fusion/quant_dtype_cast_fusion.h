/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef LITE_QUANT_DTYPE_CAST_FUSION_H
#define LITE_QUANT_DTYPE_CAST_FUSION_H

#include <string>
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class QuantDtypeCastFusion : public PatternProcessPass {
 public:
  explicit QuantDtypeCastFusion(bool multigraph = true, const std::string &name = "quant_dtype_cast_fusion")
      : PatternProcessPass(name, multigraph) {}
  ~QuantDtypeCastFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // LITE_QUANT_DTYPE_CAST_FUSION_H
