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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_DATA_DISTRIBUTION_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_DATA_DISTRIBUTION_H
#include <vector>
#include <utility>
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/quantize_util.h"
namespace mindspore::lite::quant {
class DataDistribution {
 public:
  DataDistribution() = default;
  DataDistribution(CNodePtr cnode, int bins, size_t bits, int quant_max, int quant_min,
                   ActivationQuantizedMethod activation_quant_method, bool symmetry) {
    this->activation_quant_method_ = activation_quant_method;
    this->cnode_ = std::move(cnode);
    this->bin_num_ = bins;
    this->bit_num_ = bits;
    histogram_.resize(bin_num_);
    real_max_ = -FLT_MAX;
    real_min_ = FLT_MAX;
    this->quant_max_ = quant_max;
    this->quant_min_ = quant_min;
    std::fill(histogram_.begin(), histogram_.end(), 1.0e-7);
    if (this->activation_quant_method_ == KL) {
      symmetry_ = true;
    } else {
      symmetry_ = symmetry;
    }
  }

  int RecordMaxMinValueArray(const std::vector<float> &data);

  void UpdateInterval();

  int UpdateHistogram(const std::vector<float> &data);

  void DumpHistogram();

  void HandleBinForKL(int quant_bint_nums, int bin_index, std::vector<float> *quantized_histogram,
                      std::vector<float> *expanded_histogram);

  int ComputeThreshold();

  double GetScale();

  int32_t GetZeroPoint();

  float GetRealMax() { return this->real_max_; }

  float GetRealMin() { return this->real_min_; }

  float GetEncodeMin() { return this->encode_min_; }

  float GetEncodeMax() { return this->encode_max_; }

  CNodePtr GetCNode() { return this->cnode_; }

 private:
  double CalculateMinMaxScale();
  double CalculateRemovalOutlierScale();
  double CalculateKLScale();
  double CalculateScaleAndZp(float min_value, float max_value);

  std::pair<float, float> CalQuantileMinMax(const std::vector<float> &min_datas, const std::vector<float> &max_datas);

 private:
  std::vector<float> histogram_;
  CNodePtr cnode_;
  int bin_num_ = 0;
  float interval_ = 0;
  float real_max_ = FLT_MIN;
  float real_min_ = FLT_MAX;
  float best_T_ = 0.0f;
  size_t bit_num_ = 0;
  float encode_min_ = 0.0f;
  float encode_max_ = 0.0f;
  int quant_max_ = 255;
  int quant_min_ = 0;
  ActivationQuantizedMethod activation_quant_method_ = MAX_MIN;
  std::vector<float> min_datas_;
  std::vector<float> max_datas_;
  std::pair<float, float> percent_result_{0.0, 0.0};
  double scale_ = 0;
  int zero_point_ = 0;
  bool symmetry_ = true;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_DATA_DISTRIBUTION_H
