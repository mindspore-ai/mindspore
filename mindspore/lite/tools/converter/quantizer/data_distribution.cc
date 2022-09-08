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

#define USE_DEPRECATED_API
#include "tools/converter/quantizer/data_distribution.h"
#include <algorithm>
#include <vector>
#include <utility>
#include <set>
#include <cfloat>
#include <cmath>
#include "tools/common/statistic_utils.h"

namespace mindspore::lite::quant {
int DataDistribution::RecordMaxMinValueArray(const std::vector<float> &data) {
  if (data.empty()) {
    return RET_ERROR;
  }
  auto min_max = GetFloatMinMaxValue(data.data(), data.size());
  float min_num = min_max.first;
  float max_num = min_max.second;
  real_min_ = std::min(min_num, real_min_);
  real_max_ = std::max(max_num, real_max_);
  if (activation_quant_method_ == REMOVAL_OUTLIER) {
    auto bak_data(data);
    const float min_percentage = 0.0001;
    const float max_percentage = 0.9999;
    auto const quantile_min_index = static_cast<int>(min_percentage * bak_data.size());
    auto const quantile_max_index = static_cast<int>(max_percentage * bak_data.size());
    std::nth_element(bak_data.begin(), bak_data.begin() + quantile_min_index, bak_data.end());
    auto quantile_min = bak_data.at(quantile_min_index);
    std::nth_element(bak_data.begin() + quantile_min_index + 1, bak_data.begin() + quantile_max_index, bak_data.end());
    auto quantile_max = bak_data.at(quantile_max_index);
    MS_LOG(DEBUG) << "real_min_:" << real_min_ << " real_max_:" << real_max_ << " quantile_min:" << quantile_min
                  << " quantile_max:" << quantile_max;
    this->min_datas_.emplace_back(quantile_min);
    this->max_datas_.emplace_back(quantile_max);
  }
  return RET_OK;
}

void DataDistribution::UpdateInterval() {
  auto max_value = std::max(fabs(this->real_max_), fabs(this->real_min_));
  MS_CHECK_TRUE_RET_VOID(bin_num_ != 0);
  this->interval_ = max_value / static_cast<float>(bin_num_);
}

int DataDistribution::UpdateHistogram(const std::vector<float> &data) {
  for (auto value : data) {
    if (fabs(value) <= DBL_EPSILON) {
      continue;
    }
    if (fabs(this->interval_) <= DBL_EPSILON) {
      MS_LOG(ERROR) << "divisor 'interval' cannot be 0.";
      return RET_ERROR;
    }
    int bin_index = std::min(static_cast<int>(std::fabs(value) / this->interval_), bin_num_ - 1);
    this->histogram_[bin_index]++;
  }
  return RET_OK;
}

void DataDistribution::DumpHistogram() {
  MS_LOG(INFO) << "Print node " << cnode_->fullname_with_scope() << " histogram";
  for (float item : this->histogram_) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
}

void DataDistribution::HandleBinForKL(int quant_bint_nums, int bin_index, std::vector<float> *quantized_histogram,
                                      std::vector<float> *expanded_histogram) {
  CHECK_NULL_RETURN_VOID(quantized_histogram);
  CHECK_NULL_RETURN_VOID(expanded_histogram);
  MS_CHECK_TRUE_RET_VOID(quant_bint_nums != 0);
  const float bin_interval = static_cast<float>(bin_index) / static_cast<float>(quant_bint_nums);
  MS_CHECK_TRUE_RET_VOID(quant_bint_nums <= static_cast<int>(quantized_histogram->size()));
  // merge i bins to target bins
  for (int i = 0; i < quant_bint_nums; ++i) {
    const float start = i * bin_interval;
    const float end = start + bin_interval;
    const int left_upper = static_cast<int>(std::ceil(start));
    if (left_upper > start) {
      const double left_scale = left_upper - start;
      MS_ASSERT((left_upper - 1) < this->histogram_.size());
      quantized_histogram->at(i) += left_scale * this->histogram_[left_upper - 1];
    }
    const int right_lower = static_cast<int>(std::floor(end));
    if (right_lower < end) {
      const double right_scale = end - right_lower;
      quantized_histogram->at(i) += right_scale * this->histogram_[right_lower];
    }
    std::for_each(this->histogram_.begin() + left_upper, this->histogram_.begin() + right_lower,
                  [&quantized_histogram, i](float item) { quantized_histogram->at(i) += item; });
  }
  // expand target bins to i bins in order to calculate KL with reference_histogram
  for (int i = 0; i < quant_bint_nums; ++i) {
    const float start = i * bin_interval;
    const float end = start + bin_interval;
    float count = 0;
    const int left_upper = static_cast<int>(std::ceil(start));
    float left_scale = 0.0f;
    if (left_upper > start) {
      left_scale = left_upper - start;
      if (fabs(this->histogram_[left_upper - 1]) > DBL_EPSILON) {
        count += left_scale;
      }
    }
    const int right_lower = static_cast<int>(std::floor(end));
    double right_scale = 0.0f;
    if (right_lower < end) {
      right_scale = end - right_lower;
      if (fabs(this->histogram_[right_lower]) > DBL_EPSILON) {
        count += right_scale;
      }
    }
    std::for_each(this->histogram_.begin() + left_upper, this->histogram_.begin() + right_lower, [&count](float item) {
      bool is_zero = (item <= kEps && item >= -kEps);
      if (!is_zero) {
        count += 1;
      }
    });
    if (fabs(count) <= DBL_EPSILON) {
      continue;
    }
    const float average_num = quantized_histogram->at(i) / count;
    if (left_upper > start && fabs(this->histogram_[left_upper - 1]) > DBL_EPSILON) {
      expanded_histogram->at(left_upper - 1) += average_num * left_scale;
    }
    if (right_lower < end && fabs(this->histogram_[right_lower]) > DBL_EPSILON) {
      expanded_histogram->at(right_lower) += average_num * right_scale;
    }
    for (int k = left_upper; k < right_lower; ++k) {
      if (fabs(this->histogram_[k]) > DBL_EPSILON) {
        expanded_histogram->at(k) += average_num;
      }
    }
  }
}

int DataDistribution::ComputeThreshold() {
  if (activation_quant_method_ != KL) {
    return RET_OK;
  }

  int threshold = INT8_MAX + 1;
  float min_kl = FLT_MAX;
  float after_threshold_sum = std::accumulate(this->histogram_.begin() + INT8_MAX + 1, this->histogram_.end(), 0.0f);

  for (int i = INT8_MAX + 1; i < this->bin_num_; ++i) {
    std::vector<float> quantized_histogram(INT8_MAX + 1, 0);
    std::vector<float> reference_histogram(this->histogram_.begin(), this->histogram_.begin() + i);
    std::vector<float> expanded_histogram(i, 0);
    reference_histogram[i - 1] += after_threshold_sum;
    after_threshold_sum -= this->histogram_[i];
    // handle bins for computing KL.
    HandleBinForKL(INT8_MAX + 1, i, &quantized_histogram, &expanded_histogram);
    const float kl = lite::KLDivergence(reference_histogram, expanded_histogram);
    if (kl < min_kl) {
      min_kl = kl;
      threshold = i;
    }
  }
  this->best_T_ = (static_cast<float>(threshold) + 0.5f) * this->interval_;
  MS_LOG(DEBUG) << cnode_->fullname_with_scope() << " Best threshold bin index: " << threshold << " T: " << best_T_
                << " max: " << std::max(fabs(this->real_max_), fabs(this->real_min_));
  return RET_OK;
}

double DataDistribution::CalculateMinMaxScale() { return CalculateScale(this->real_min_, this->real_max_); }

double DataDistribution::CalculateRemovalOutlierScale() {
  this->percent_result_ = CalQuantileMinMax(min_datas_, max_datas_);
  return CalculateScale(percent_result_.first, percent_result_.second);
}

std::pair<float, float> DataDistribution::CalQuantileMinMax(const std::vector<float> &min_datas,
                                                            const std::vector<float> &max_datas) {
  MS_ASSERT(!min_datas.empty());
  MS_ASSERT(!max_datas.empty());
  auto avg_min = accumulate(min_datas.begin(), min_datas.end(), 0.0) / min_datas.size();
  auto avg_max = accumulate(max_datas.begin(), max_datas.end(), 0.0) / max_datas.size();
  return {avg_min, avg_max};
}

double DataDistribution::CalculateScale(float min_value, float max_value) {
  EncodeMinMax(min_value, max_value, quant_min_, quant_max_, symmetric_, &encode_min_, &encode_max_);
  auto q_range = quant_max_ - quant_min_;
  MS_ASSERT(quant_max_ - quant_min_ > 0);
  auto range = encode_max_ - encode_min_;
  return range / q_range;
}

double DataDistribution::CalculateKLScale() {
  return CalculateScale(-std::abs(this->best_T_), std::abs(this->best_T_));
}

double DataDistribution::GetScale() {
  switch (this->activation_quant_method_) {
    case MAX_MIN:
      this->scale_ = CalculateMinMaxScale();
      break;
    case KL:
      this->scale_ = CalculateKLScale();
      break;
    case REMOVAL_OUTLIER:
      this->scale_ = CalculateRemovalOutlierScale();
      break;
    default:
      MS_LOG(ERROR) << "Unsupported activation quant method " << this->activation_quant_method_;
      return FLT_MAX;
  }
  return this->scale_;
}

int32_t DataDistribution::GetZeroPoint() {
  if (symmetric_) {
    zero_point_ = 0;
  } else {
    MS_ASSERT(scale_ > 0);
    zero_point_ = static_cast<int32_t>(std::round(quant_min_ - encode_min_ / scale_));
  }
  return zero_point_;
}
}  // namespace mindspore::lite::quant
