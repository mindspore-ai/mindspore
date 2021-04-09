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

#include <math.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <bitset>
#include <tuple>
#include "debug/debugger/tensor_summary.h"

namespace mindspore {
using CONDITION_TYPE = DebugServices::CONDITION_TYPE;

RangeCountCalculator::RangeCountCalculator()
    : range_start_inclusive(-std::numeric_limits<double>::infinity()),
      range_end_inclusive(std::numeric_limits<double>::infinity()),
      count(0),
      total(0) {}

void RangeCountCalculator::ProcessElement(double element) {
  count += (element >= range_start_inclusive && element <= range_end_inclusive);
  total += 1;
}

double RangeCountCalculator::GetPercentInRange() const {
  if (total == 0) {
    return 0.0;
  }
  return 100.0 * count / total;
}

AllCloseCalculator::AllCloseCalculator() : atol(1.0e-8), rtol(1.0e-5), result(true) {}

void AllCloseCalculator::ProcessElement(double current, double previous) {
  result = result && (std::abs(current - previous) <= (atol + rtol * std::abs(previous)));
}

bool AllCloseCalculator::IsAllClose() { return result; }

MeanCalculator::MeanCalculator() : mean(0.0), count(0) {}

void MeanCalculator::ProcessElement(double value) {
  count += 1;
  double delta = value - mean;
  mean += delta / count;
}

double MeanCalculator::GetMean() { return mean; }

VarianceAndMeanCalculator::VarianceAndMeanCalculator() : mean(0.0), count(0), m2(0.0) {}

void VarianceAndMeanCalculator::ProcessElement(double value) {
  count += 1;
  double delta = value - mean;
  mean += delta / count;
  m2 += delta * (value - mean);
}

double VarianceAndMeanCalculator::GetMean() { return mean; }

double VarianceAndMeanCalculator::GetVariance() {
  if (count > 1) {
    return m2 / (count - 1);
  } else {
    return 0.0;
  }
}

double VarianceAndMeanCalculator::GetStandardDeviation() { return sqrt(GetVariance()); }

template <typename T>
TensorSummary<T>::TensorSummary(void *current_tensor_ptr, void *const previous_tensor_ptr, uint32_t num_elements)
    : current_tensor_ptr(reinterpret_cast<T *>(current_tensor_ptr)),
      prev_tensor_ptr(reinterpret_cast<T *>(previous_tensor_ptr)),
      num_elements(num_elements),
      min(std::numeric_limits<double>::max()),
      max(std::numeric_limits<double>::lowest()),
      inf_count(0),
      nan_count(0),
      zero_count(0),
      epsilon(1.0e-9),
      mean_sd_cal_enabled(false) {}

template <typename T>
void TensorSummary<T>::SummarizeTensor(const std::vector<DebugServices::watchpoint_t> &wps) {
  InitCalculators(wps);
  for (size_t i = 0; i < num_elements; ++i) {
    auto current_value = static_cast<double>(current_tensor_ptr[i]);
    double previous_value =
      prev_tensor_ptr ? static_cast<double>(prev_tensor_ptr[i]) : std::numeric_limits<double>::quiet_NaN();
    inf_count += std::isinf(current_value);
    nan_count += std::isnan(current_value);
    zero_count += (current_value == 0);
    max = std::max(max, current_value);
    min = std::min(min, current_value);
    if (mean_sd_cal_enabled) {
      current_mean_variance.ProcessElement(current_value);
    }
    for (auto &it : all_close) {
      it.second->ProcessElement(current_value, previous_value);
    }
    for (auto &range_count : range_counts) {
      range_count.second->ProcessElement(current_value);
    }
    for (auto &mean : means) {
      if (mean.first == "curr_prev_diff_mean") {
        mean.second->ProcessElement(std::abs(current_value - previous_value));
      } else if (mean.first == "abs_prev_mean") {
        mean.second->ProcessElement(std::abs(previous_value));
      } else if (mean.first == "abs_current_mean") {
        mean.second->ProcessElement(std::abs(current_value));
      }
    }
  }
}

template <typename T>
std::tuple<bool, int, std::vector<DebugServices::parameter_t>> TensorSummary<T>::IsWatchpointHit(
  DebugServices::watchpoint_t wp) {
  auto parameter_list = wp.parameter_list;
  bool hit = false;
  std::bitset<32> error_code;
  CONDITION_TYPE type = wp.condition.type;
  // bit 0 denotes presence of nan
  error_code.set(0, nan_count > 0);
  // bit 1 denotes presence of inf
  error_code.set(1, inf_count > 0);

  if (type == CONDITION_TYPE::HAS_NAN) {
    error_code.reset();
    hit = nan_count > 0;
  } else if (type == CONDITION_TYPE::HAS_INF) {
    error_code.reset();
    hit = inf_count > 0;
  } else if (type == CONDITION_TYPE::GENERAL_OVERFLOW) {
    error_code.reset();
    hit = (nan_count + inf_count) > 0;
  } else if (type == CONDITION_TYPE::NOT_CHANGED && prev_tensor_ptr && error_code.none()) {
    hit = all_close[wp.id]->IsAllClose();
  } else if ((type == CONDITION_TYPE::NOT_CHANGED || type == CONDITION_TYPE::CHANGE_TOO_LARGE ||
              type == CONDITION_TYPE::CHANGE_TOO_SMALL) &&
             !prev_tensor_ptr) {
    // bit 2 denotes absence of previous tensor
    error_code.set(2, true);
  }

  if (error_code.none()) {
    for (auto &parameter : parameter_list) {
      if (parameter.disabled || error_code.any()) {
        continue;
      }
      // extract inequality type from watchpoint for backward compatibility
      std::string inequality_type;
      if (wp.is_gt_wp()) {
        inequality_type = "gt";
      } else if (wp.is_lt_wp()) {
        inequality_type = "lt";
      }
      parameter.Evaluate(StatLookup(parameter.name, wp), inequality_type);
      hit = hit || parameter.hit;
    }
  }
  return std::make_tuple(hit, static_cast<int32_t>(error_code.to_ulong()), parameter_list);
}

template <typename T>
double_t TensorSummary<T>::StatLookup(const std::string &parameter_name, const DebugServices::watchpoint_t &wp) {
  if (parameter_name == "param") return StatLookup(wp);
  std::string param_type;
  auto pos = parameter_name.find_last_of('_');
  if (pos != std::string::npos) {
    param_type = parameter_name.substr(0, pos);
  }

  if (param_type == "max") {
    return max;
  } else if (param_type == "min") {
    return min;
  } else if (param_type == "max_min") {
    return max - min;
  } else if (param_type == "mean") {
    return current_mean_variance.GetMean();
  } else if (param_type == "sd") {
    return current_mean_variance.GetStandardDeviation();
  } else if (param_type == "abs_mean") {
    if (means.find("abs_current_mean") != means.end()) {
      return means["abs_current_mean"]->GetMean();
    }
  } else if (param_type == "abs_mean_update_ratio" && prev_tensor_ptr) {
    if (means.find("curr_prev_diff_mean") != means.end() && means.find("abs_prev_mean") != means.end()) {
      return means["curr_prev_diff_mean"]->GetMean() / (means["abs_prev_mean"]->GetMean() + epsilon);
    }
  } else if (param_type == "range_percentage") {
    if (range_counts.find(wp.id) != range_counts.end()) {
      return range_counts[wp.id]->GetPercentInRange();
    }
  } else if (param_type == "zero_percentage") {
    return GetZeroValPercent();
  }
  return std::numeric_limits<double_t>::quiet_NaN();
}

template <typename T>
double_t TensorSummary<T>::StatLookup(const DebugServices::watchpoint_t &wp) {
  CONDITION_TYPE type = wp.condition.type;
  if (type == CONDITION_TYPE::MAX_LT || type == CONDITION_TYPE::MAX_GT) {
    return max;
  } else if (type == CONDITION_TYPE::MIN_LT || type == CONDITION_TYPE::MIN_GT) {
    return min;
  } else if (type == CONDITION_TYPE::MEAN_LT || type == CONDITION_TYPE::MEAN_GT) {
    return current_mean_variance.GetMean();
  } else if (type == CONDITION_TYPE::SD_LT || type == CONDITION_TYPE::SD_GT) {
    return current_mean_variance.GetStandardDeviation();
  } else if (type == CONDITION_TYPE::MAX_MIN_GT || type == CONDITION_TYPE::MAX_MIN_LT) {
    return max - min;
  }
  return std::numeric_limits<double_t>::quiet_NaN();
}

template <typename T>
double_t TensorSummary<T>::GetZeroValPercent() {
  if (num_elements == 0) {
    return 0;
  }

  return (zero_count * 100.0) / num_elements;
}

template <typename T>
void TensorSummary<T>::InitCalculators(const std::vector<DebugServices::watchpoint_t> &wps) {
  for (auto &wp : wps) {
    auto wp_id = wp.id;
    mean_sd_cal_enabled = mean_sd_cal_enabled || wp.mean_sd_enabled();
    if (wp.allclose_enabled() && prev_tensor_ptr) {
      all_close[wp_id] = std::make_unique<AllCloseCalculator>();
      if (!wp.parameter_list[0].disabled) {
        all_close[wp_id]->set_atol(wp.parameter_list[0].value);
      }
      if (!wp.parameter_list[1].disabled) {
        all_close[wp_id]->set_rtol(wp.parameter_list[1].value);
      }
    } else if (wp.range_enabled()) {
      range_counts[wp_id] = std::make_unique<RangeCountCalculator>();
      if (!wp.parameter_list[0].disabled) {
        range_counts[wp_id]->set_range_start_inclusive(wp.parameter_list[0].value);
      }
      if (!wp.parameter_list[1].disabled) {
        range_counts[wp_id]->set_range_end_inclusive(wp.parameter_list[1].value);
      }
    } else if (wp.tensor_update_ratio_mean_enabled() && prev_tensor_ptr) {
      means.insert({"curr_prev_diff_mean", std::make_unique<MeanCalculator>()});
      means.insert({"abs_prev_mean", std::make_unique<MeanCalculator>()});
    } else if (wp.abs_mean_enabled()) {
      means.insert({"abs_current_mean", std::make_unique<MeanCalculator>()});
    }
  }
}
template class TensorSummary<uint8_t>;
template class TensorSummary<int8_t>;
template class TensorSummary<uint16_t>;
template class TensorSummary<int16_t>;
template class TensorSummary<uint32_t>;
template class TensorSummary<int32_t>;
template class TensorSummary<uint64_t>;
template class TensorSummary<int64_t>;
template class TensorSummary<float16>;
template class TensorSummary<float>;
template class TensorSummary<double>;
template class TensorSummary<bool>;
}  // namespace mindspore
