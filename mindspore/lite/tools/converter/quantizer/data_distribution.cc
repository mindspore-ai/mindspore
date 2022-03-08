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

#include "tools/converter/quantizer/data_distribution.h"
#include <algorithm>
#include <vector>
#include <utility>
#include <set>
#include "tools/common/statistic_utils.h"

namespace mindspore::lite::quant {
int DataDistribution::RecordMaxMinValueArray(const std::vector<float> &data) {
  if (data.empty()) {
    return RET_ERROR;
  }
  float min_num = data.at(0);
  float max_num = data.at(0);
  for (float val : data) {
    min_num = std::min(val, min_num);
    max_num = std::max(val, max_num);
  }
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
  MS_ASSERT(bin_num_ != 0);
  this->interval_ = max_value / static_cast<float>(bin_num_);
}

int DataDistribution::UpdateHistogram(const std::vector<float> &data) {
  for (auto value : data) {
    if (value == 0) {
      continue;
    }
    if (this->interval_ == 0) {
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
  MS_ASSERT(quantized_histogram != nullptr && expanded_histogram != nullptr);
  MS_ASSERT(quant_bint_nums != 0);
  const float bin_interval = static_cast<float>(bin_index) / static_cast<float>(quant_bint_nums);
  MS_ASSERT(quant_bint_nums <= quantized_histogram->size());
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
      if (this->histogram_[left_upper - 1] != 0) {
        count += left_scale;
      }
    }
    const int right_lower = static_cast<int>(std::floor(end));
    double right_scale = 0.0f;
    if (right_lower < end) {
      right_scale = end - right_lower;
      if (this->histogram_[right_lower] != 0) {
        count += right_scale;
      }
    }
    std::for_each(this->histogram_.begin() + left_upper, this->histogram_.begin() + right_lower, [&count](float item) {
      if (item != 0) {
        count += 1;
      }
    });
    if (count == 0) {
      continue;
    }
    const float average_num = quantized_histogram->at(i) / count;
    if (left_upper > start && this->histogram_[left_upper - 1] != 0) {
      expanded_histogram->at(left_upper - 1) += average_num * left_scale;
    }
    if (right_lower < end && this->histogram_[right_lower] != 0) {
      expanded_histogram->at(right_lower) += average_num * right_scale;
    }
    for (int k = left_upper; k < right_lower; ++k) {
      if (this->histogram_[k] != 0) {
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

double DataDistribution::CalculateMinMaxScale() { return CalculateScaleAndZp(this->real_min_, this->real_max_); }

double DataDistribution::CalculateRemovalOutlierScale() {
  this->percent_result_ = CalQuantileMinMax(min_datas_, max_datas_);
  return CalculateScaleAndZp(percent_result_.first, percent_result_.second);
}

std::pair<float, float> DataDistribution::CalQuantileMinMax(const std::vector<float> &min_datas,
                                                            const std::vector<float> &max_datas) {
  MS_ASSERT(!min_datas.empty());
  MS_ASSERT(!max_datas.empty());
  auto avg_min = accumulate(min_datas.begin(), min_datas.end(), 0.0) / min_datas.size();
  auto avg_max = accumulate(max_datas.begin(), max_datas.end(), 0.0) / max_datas.size();
  return {avg_min, avg_max};
}

double DataDistribution::CalculateScaleAndZp(float min_value, float max_value) {
  if (symmetry_) {
    auto abs_max = std::max(fabs(min_value), fabs(max_value));
    encode_min_ = -abs_max;
    encode_max_ = abs_max;
  } else {
    encode_min_ = min_value;
    encode_max_ = max_value;
  }

  // Handling 0
  // Inputs are strictly positive, set the real min to 0. e.g. input range = [1.0, 5.0] -> [0.0, 5.0]
  if (encode_min_ > 0.0f) {
    MS_LOG(DEBUG) << "min " << encode_min_ << " is bigger then 0, set to 0, this may course low precision";
    encode_min_ = 0.0f;
  }
  // Inputs are strictly negative, set the real max to 0. e.g. input range = [-5.0, -1.0] -> [-5.0, 0.0]
  if (encode_max_ < 0.0f) {
    MS_LOG(DEBUG) << "real_max " << encode_max_ << " is smaller than 0, set to 0, this may course low precision";
    encode_max_ = 0.0f;
  }
  // Inputs are both negative and positive, real_min and real_max are slightly shifted to make the floating point zero
  // exactly representable. e.g. input range = [-5.1, 5.1] -> [-5.12, 5.08]

  // handle case where encode_min_ == encode_max_
  float epsilon = 1e-5;
  encode_max_ = std::max(encode_max_, encode_min_ + epsilon);
  auto range = encode_max_ - encode_min_;
  MS_ASSERT(quant_max_ - quant_min_ > 0);
  return range / (quant_max_ - quant_min_);
}

double DataDistribution::CalculateKLScale() {
  return CalculateScaleAndZp(-std::abs(this->best_T_), std::abs(this->best_T_));
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
  if (symmetry_) {
    zero_point_ = 0;
  } else {
    MS_ASSERT(scale_ > 0);
    zero_point_ = static_cast<int32_t>(std::round(quant_min_ - encode_min_ / scale_));
  }
  return zero_point_;
}
}  // namespace mindspore::lite::quant
