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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CLE_STRATEGY_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_CLE_STRATEGY_H

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
  explicit CLEStrategy(const FuncGraphPtr &func_graph, const std::shared_ptr<Calibrator> &calibrator,
                       const converter::Flags &flags)
      : func_graph_(func_graph), calibrator_(calibrator), flags_(flags) {}

  ~CLEStrategy();

  int Run();

 private:
  int ReplaceGraphRelu6ToRelu();
  int ReplaceCNodeRelu6ToRelu(const CNodePtr &cnode);
  int FindPattern();
  int WeightEqualization();
  int ClipHighBias();
  int ClipBackBiasLayer(const CNodePtr &cnode, float absorb_bias);
  int ClipFrontBiasLayer(const CNodePtr &cnode, float absorb_bias);
  int ClipHighBiasWithTwoLayer(const CombinationLayer &layer_group, const std::map<std::string, float> &absorb_biases);
  int ClipHighBiasWithThreeLayer(const CombinationLayer &layer_group,
                                 const std::map<std::string, float> &absorb_biases);
  int ReduceWeight(const CNodePtr &cnode, int weight_index, int preferred_dim, std::vector<float> *reduce_weight);

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
  int DoInference();

 private:
  FuncGraphPtr func_graph_ = nullptr;
  std::shared_ptr<Calibrator> calibrator_ = nullptr;
  converter::Flags flags_;
  CLEPattern *cle_pattern_ = nullptr;
  std::map<std::string, float> total_min_;

  bool depthwise_replace_relu6_flag_ = false;
  bool clip_bias_flag_ = false;
};
}  // namespace mindspore::lite::quant

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CLE_STRATEGY_H
