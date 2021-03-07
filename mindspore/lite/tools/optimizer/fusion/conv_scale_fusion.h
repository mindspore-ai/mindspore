/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *conv_activation_fusion.h
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_SRC_PASS_FUSION_CONV_SCALE_FUSION_H_
#define MINDSPORE_LITE_SRC_PASS_FUSION_CONV_SCALE_FUSION_H_

#include "tools/optimizer/fusion/conv_transform_fusion.h"

namespace mindspore::opt {
class ConvScaleFusion : public ConvTransformFusion {
 public:
  explicit ConvScaleFusion(bool multigraph = true) : ConvTransformFusion(multigraph, "conv_scale_fusion") {}
  ~ConvScaleFusion() override = default;
  const BaseRef DefinePattern() const override;
  void InitTransParam(const CNodePtr &, int, float *, float *) const override;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_SRC_PASS_FUSION_CONV_SCALE_FUSION_H_
