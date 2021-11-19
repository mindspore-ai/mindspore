/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_DIVERG_INFO_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_DIVERG_INFO_H
#include <vector>
#include <utility>
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/quantize_util.h"
namespace mindspore::lite::quant {
class DivergInfo {
 public:
  DivergInfo() = default;
  DivergInfo(CNodePtr cnode, int bins, size_t bits, int quant_max, int quant_min,
             ActivationQuantizedMethod activation_quant_method) {
    this->activation_quant_method = activation_quant_method;
    this->cnode = std::move(cnode);
    this->bin_num = bins;
    this->bit_num = bits;
    histogram.resize(bin_num);
    max = -FLT_MAX;
    min = FLT_MAX;
    this->quant_max = quant_max;
    this->quant_min = quant_min;
    std::fill(histogram.begin(), histogram.end(), 1.0e-7);
  }

  int RecordMaxMinValue(const std::vector<float> &data);

  int RecordMaxMinValueArray(const std::vector<float> &data);

  void UpdateInterval();

  int UpdateHistogram(const std::vector<float> &data);

  void DumpHistogram();

  void HandleBinForKL(int quant_bint_nums, int bin_index, std::vector<float> *quantized_histogram,
                      std::vector<float> *expanded_histogram);

  int ComputeThreshold();

  std::pair<CNodePtr, float> GetScale();

  std::pair<CNodePtr, int32_t> GetZeropoint();

  float GetMax() { return this->max; }

  float GetMin() { return this->min; }

  CNodePtr GetCNode() { return this->cnode; }

 private:
  std::vector<float> histogram;
  CNodePtr cnode;
  int bin_num = 0;
  float interval = 0;
  float max = 0.0f;
  float min = 0.0f;
  float best_T = 0.0f;
  size_t bit_num = 0;
  int quant_max = 255;
  int quant_min = 0;
  ActivationQuantizedMethod activation_quant_method = MAX_MIN;
  std::vector<float> min_datas;
  std::vector<float> max_datas;
  std::pair<float, float> percent_result{0.0, 0.0};
  float scale_tmp = 0;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_DIVERG_INFO_H
