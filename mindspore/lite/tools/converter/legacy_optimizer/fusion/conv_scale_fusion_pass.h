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

#ifndef MINDSPORE_PREDICT_CONV_SCALE_FUSION_PASS_H
#define MINDSPORE_PREDICT_CONV_SCALE_FUSION_PASS_H

#include "tools/converter/legacy_optimizer/fusion/conv_scale_bias_fusion_pass.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace mindspore {
namespace lite {
class ConvScaleFusionPass : public ConvScaleBiasFusionPass {
 public:
  ConvScaleFusionPass() = default;

  ~ConvScaleFusionPass() override = default;

  STATUS DefinePattern() override;

  STATUS DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                  std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) override;

  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  STATUS GetTransParam(schema::MetaGraphT *graph, std::shared_ptr<Path> scalePath, int32_t kernelNum) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_CONV_SCALE_FUSION_PASS_H
