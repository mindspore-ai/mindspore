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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CLE_STRATEGY_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CLE_STRATEGY_H_

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "ir/func_graph.h"
#include "tools/converter/quantizer/cle_pattern.h"
#include "tools/converter/quantizer/calibrator.h"

namespace mindspore::lite::quant {
class CLEStrategy {
 public:
  explicit CLEStrategy(const FuncGraphPtr &func_graph) : func_graph_(func_graph) {}

  ~CLEStrategy();

  int Run();

 private:
  int FindPattern();
  int WeightEqualization();

  int EqualizationWithTwoLayer(const CombinationLayer &layer_group, const std::vector<double> &scales);
  int EqualizationWithThreeLayer(const CombinationLayer &layer_group, const std::vector<double> &scales12,
                                 const std::vector<double> &scales23);
  int EqualizationAdjust(const CNodePtr &cnode, const std::vector<double> &scales, size_t input_index,
                         int preferred_dim, bool multiplication);

  int CalcRange(const CNodePtr &cnode, std::vector<float> *ranges, int preferred_dim);
  int CalcDataRange(const float *data, size_t element_cnt, const std::vector<int> &dims, int preferred_dim,
                    std::vector<float> *ranges);
  int CalcScaleWithTwoLayer(const CombinationLayer &layer_group, std::vector<double> *scales);
  int CalcScaleWithThreeLayer(const CombinationLayer &layer_group, std::vector<double> *scales12,
                              std::vector<double> *scales23);

 private:
  FuncGraphPtr func_graph_ = nullptr;
  CLEPattern *cle_pattern_ = nullptr;
};
}  // namespace mindspore::lite::quant

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CLE_STRATEGY_H_
