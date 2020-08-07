/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PREDICT_CONV_RELU6_FUSION_PASS_H
#define MINDSPORE_PREDICT_CONV_RELU6_FUSION_PASS_H

#include "tools/converter/legacy_optimizer/fusion/conv_activation_fusion_pass.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace mindspore {
namespace lite {
class ConvRelu6FusionPass : public ConvActivationFusionPass {
 public:
  ConvRelu6FusionPass() = default;

  ~ConvRelu6FusionPass() override = default;

  STATUS DefinePattern() override;

  STATUS SetActivationType() override;

  STATUS DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                  std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) override;

  STATUS Run(schema::MetaGraphT *graph) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_CONV_RELU6_FUSION_PASS_H

