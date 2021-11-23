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
namespace mindspore::lite::quant {
int DataDistribution::RecordMaxMinValue(const std::vector<float> &data) {
  for (float val : data) {
    max_ = std::max(val, max_);
    min_ = std::min(val, min_);
  }
  return RET_OK;
}

int DataDistribution::RecordMaxMinValueArray(const std::vector<float> &data) {
  if (data.empty()) {
    return RET_ERROR;
  }
  float max_num = data.at(0);
  float min_num = data.at(0);
  for (float val : data) {
    max_num = std::max(val, max_num);
    min_num = std::min(val, min_num);
  }
  this->max_datas_.emplace_back(max_num);
  this->min_datas_.emplace_back(min_num);
  return RET_OK;
}

void DataDistribution::UpdateInterval() {
  auto max_value = std::max(fabs(this->max_), fabs(this->min_));
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
  // merge i bins to target bins
  for (int i = 0; i < quant_bint_nums; ++i) {
    const float start = i * bin_interval;
    const float end = start + bin_interval;
    const int left_upper = static_cast<int>(std::ceil(start));
    if (left_upper > start) {
      const double left_scale = left_upper - start;
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
  if (activation_quant_method_ == MAX_MIN) {
    this->best_T_ = std::max(fabs(this->max_), fabs(this->min_));
    MS_LOG(DEBUG) << "using MAX_MIN, T: " << this->best_T_;
    return RET_OK;
  }

  if (activation_quant_method_ == REMOVAL_OUTLIER && !this->min_datas_.empty()) {
    this->percent_result_ = OutlierMethod(min_datas_, max_datas_);
    this->best_T_ = std::max(std::fabs(percent_result_.first), std::fabs(percent_result_.second));
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
    auto KLDivergence = [](std::vector<float> p, std::vector<float> q) {
      auto sum = 0.0f;
      std::for_each(p.begin(), p.end(), [&sum](float item) { sum += item; });
      std::for_each(p.begin(), p.end(), [sum](float &item) { item /= sum; });
      sum = 0.0f;
      std::for_each(q.begin(), q.end(), [&sum](float item) { sum += item; });
      std::for_each(q.begin(), q.end(), [sum](float &item) { item /= sum; });

      float result = 0.0f;
      const int size = p.size();
      for (int i = 0; i < size; ++i) {
        if (p[i] != 0) {
          if (q[i] == 0) {
            result += 1.0f;
          } else {
            result += (p[i] * std::log((p[i]) / (q[i])));
          }
        }
      }
      return result;
    };
    const float kl = KLDivergence(reference_histogram, expanded_histogram);
    if (kl < min_kl) {
      min_kl = kl;
      threshold = i;
    }
  }
  this->best_T_ = (static_cast<float>(threshold) + 0.5f) * this->interval_;
  MS_LOG(DEBUG) << cnode_->fullname_with_scope() << " Best threshold bin index: " << threshold << " T: " << best_T_
                << " max: " << std::max(fabs(this->max_), fabs(this->min_));
  return RET_OK;
}

float DataDistribution::GetScale() {
  float max_value = this->best_T_;
  float min_value = -max_value;

  if (this->activation_quant_method_ == REMOVAL_OUTLIER) {
    min_value = percent_result_.first;
    max_value = percent_result_.second;
  }

  MS_CHECK_TRUE_MSG(quant_max_ - quant_min_ > 0, 0, "quant_max_ - quant_min_ <= 0");
  this->scale_ = (max_value - min_value) / (quant_max_ - quant_min_);
  MS_ASSERT(fabs(this->scale_) <= 0.0f);
  return this->scale_;
}

// Support for asymmetry in the future
int32_t DataDistribution::GetZeroPoint() {
  int zero_point = 0;
  if (quant_min_ == 0 && quant_max_ == UINT8_MAX) {
    zero_point = INT8_MAX + 1;
  } else if (quant_min_ == INT_LEAST8_MIN + 1 && quant_max_ == INT8_MAX) {
    zero_point = 0;
  } else {
    MS_LOG(WARNING) << "unexpected quant range, quant_min_: " << quant_min_ << " quant_max_: " << quant_max_;
  }
  if (this->activation_quant_method_ == REMOVAL_OUTLIER) {
    MS_CHECK_TRUE_MSG(fabs(scale_) <= 0.0f, 1, "fabs(scale) > 0.0f");
    zero_point = std::round(quant_max_ - percent_result_.second / scale_);
  }
  return zero_point;
}
}  // namespace mindspore::lite::quant
