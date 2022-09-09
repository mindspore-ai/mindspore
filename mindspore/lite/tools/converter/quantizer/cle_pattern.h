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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CLE_PATTERN_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CLE_PATTERN_H_

#include <vector>
#include <unordered_map>
#include <string>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore::lite::quant {
struct CombinationLayer {
  CNodePtr layer1 = nullptr;
  CNodePtr layer2 = nullptr;
  CNodePtr layer3 = nullptr;
  size_t layer_num = 0;
};
constexpr size_t kInputsNum2 = 2;
constexpr size_t kInputsNum3 = 3;
enum ConvNode { COMMON_CONV, DEPTHWISE_CONV };
class CLEPattern : public opt::MultiplePatternProcessPass {
 public:
  explicit CLEPattern(const std::string &name = "CLEPattern", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}
  ~CLEPattern() override = default;
  AnfNodePtr Process(const std::string &, const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  std::vector<CombinationLayer> GetCombinationLayer() const { return combination_layer_; }

 private:
  VectorRef DefineConvWithConvPattern() const;
  VectorRef DefineConvWithDepthWithConvPattern() const;
  VectorRef DefineDepthWithConvPattern() const;

 private:
  const std::string kConvWithConvPatternName = "ConvWithConvPatternName";
  const std::string kConvWithDepthWithConvPatternName = "ConvWithDepthWithConvPatternName";
  const std::string kDepthWithConvPatternName = "DepthWithConvPatternName";

  mutable std::vector<CombinationLayer> combination_layer_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CLE_PATTERN_H_
