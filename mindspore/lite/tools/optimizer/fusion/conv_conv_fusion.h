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

#ifndef MINDSPORE_LITE_SRC_PASS_FUSION_CONV_CONV_FUSION_H_
#define MINDSPORE_LITE_SRC_PASS_FUSION_CONV_CONV_FUSION_H_

#include <string>
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class ConvConvFusion : public PatternProcessPass {
 public:
  explicit ConvConvFusion(bool multigraph = true) : PatternProcessPass("conv_conv_fusion", multigraph) {}
  ~ConvConvFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PASS_FUSION_CONV_CONV_FUSION_H_
