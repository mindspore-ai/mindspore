/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "debug/debugger/tensor_summary.h"
#include <cmath>
#include <algorithm>
#include <future>
#include <limits>
#include <memory>
#include <bitset>
#include <tuple>
#include <type_traits>

#ifdef OFFLINE_DBG_MODE
#include "base/float16.h"
#endif

namespace mindspore {
using CONDITION_TYPE = DebugServices::CONDITION_TYPE;

RangeCountCalculator::RangeCountCalculator()
    : range_start_inclusive(-std::numeric_limits<double>::infinity()),
      range_end_inclusive(std::numeric_limits<double>::infinity()),
      count(0),
      total(0) {}

void RangeCountCalculator::ProcessElement(double element) {
  if (element >= range_start_inclusive && element <= range_end_inclusive) {
    count += 1;
  }
  total += 1;
}

double RangeCountCalculator::GetPercentInRange() const {
  if (total == 0) {
    return 0.0;
  }
  const double factor = 100.0;
  return factor * count / total;
}

AllCloseCalculator::AllCloseCalculator() : atol(1.0e-8), rtol(1.0e-5), result(true) {}

void AllCloseCalculator::ProcessElement(double current, double previous) {
  result = result && (std::abs(current - previous) <= (atol + rtol * std::abs(previous)));
}

bool AllCloseCalculator::IsAllClose() const { return result; }

MeanCalculator::MeanCalculator() : mean(0.0), count(0) {}

void MeanCalculator::ProcessElement(double value) {
  count += 1;
  double delta = value - mean;
  mean += delta / count;
}

double MeanCalculator::GetMean() const { return mean; }

VarianceAndMeanCalculator::VarianceAndMeanCalculator() : mean(0.0), count(0), m2(0.0) {}

void VarianceAndMeanCalculator::ProcessElement(double value) {
  count += 1;
  double delta = value - mean;
  mean += delta / count;
  m2 += delta * (value - mean);
}

double VarianceAndMeanCalculator::GetMean() const { return mean; }

double VarianceAndMeanCalculator::GetVariance() const {
  if (count > 1) {
    return m2 / (count - 1);
  }
  return 0.0;
}

double VarianceAndMeanCalculator::GetStandardDeviation() const { return sqrt(GetVariance()); }

void L2Calculator::ProcessElement(double value) { squre_sum += value * value; }

void L2Calculator::ProcessElement(const L2Calculator &other) { this->squre_sum += other.squre_sum; }

double L2Calculator::GetL2Value() const { return std::sqrt(squre_sum); }

template <typename T>
TensorSummary<T>::TensorSummary(const void *current_tensor_ptr, const void *const previous_tensor_ptr,
                                uint64_t num_elements, uint64_t prev_num_elements)
    : current_tensor_ptr_(static_cast<const T *>(current_tensor_ptr)),
      prev_tensor_ptr_(static_cast<const T *>(previous_tensor_ptr)),
      num_elements_(num_elements),
      prev_num_elements_(prev_num_elements),
      min_(std::numeric_limits<double>::max()),
      max_(std::numeric_limits<double>::lowest()),
      avg_(0.0),
      is_bool_(false),
      neg_zero_count_(0),
      pos_zero_count_(0),
      pos_inf_count_(0),
      neg_inf_count_(0),
      inf_count_(0),
      nan_count_(0),
      zero_count_(0),
      epsilon_(1.0e-9),
      mean_sd_cal_enabled_(false) {}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Initialize watchpoints calculators based on the watchpoint category. Process all the elements within the
 * current tensor.
 */
template <typename T>
void TensorSummary<T>::SummarizeTensor(const std::vector<DebugServices::watchpoint_t> &wps) {
  InitCalculators(wps);
  for (size_t i = 0; i < num_elements_; ++i) {
    auto current_value = static_cast<double>(current_tensor_ptr_[i]);
    double previous_value = std::numeric_limits<double>::quiet_NaN();
    if (prev_tensor_ptr_) {
      if (num_elements_ == prev_num_elements_) {
        previous_value = static_cast<double>(prev_tensor_ptr_[i]);
      } else {
        MS_LOG(DEBUG) << "Current and previous tensor are not the same size.";
      }
    }
    if (std::isinf(current_value)) {
      inf_count_ += 1;
    }
    if (std::isnan(current_value)) {
      nan_count_ += 1;
    }
    if (current_value == 0.0) {
      zero_count_ += 1;
    }
    max_ = std::max(max_, current_value);
    min_ = std::min(min_, current_value);
    if (mean_sd_cal_enabled_) {
      current_mean_variance_.ProcessElement(current_value);
    }
    for (auto &it : all_close_) {
      it.second->ProcessElement(current_value, previous_value);
    }
    for (auto &range_count : range_counts_) {
      range_count.second->ProcessElement(current_value);
    }
    for (auto &mean : means_) {
      if (mean.first.compare("curr_prev_diff_mean") == 0) {
        mean.second->ProcessElement(std::abs(current_value - previous_value));
      } else if (mean.first.compare("abs_prev_mean") == 0) {
        mean.second->ProcessElement(std::abs(previous_value));
      } else if (mean.first.compare("abs_current_mean") == 0) {
        mean.second->ProcessElement(std::abs(current_value));
      }
    }
  }
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Calculates statistics on chunks of data.
 */
template <typename T>
void TensorSummary<T>::TensorStatistics(DbgDataType dtype_value) {
  if (dtype_value == DT_BOOL) {
    is_bool_ = true;
  }
  const uint64_t default_threads = 32;
  const uint64_t default_elements_per_thread = 10000;

  if (num_elements_ <= default_elements_per_thread) {
    return TensorStatisticsSingleThread();
  }
  uint64_t desired_threads = num_elements_ / default_elements_per_thread;
  uint64_t actual_threads = std::min(desired_threads, default_threads);
  uint64_t actual_elements_per_thread = num_elements_ / actual_threads;

  // Use multithread to calculate statistic on chunks of data
  void *previous_tensor_ptr = nullptr;
  size_t offset = 0;
  std::vector<std::unique_ptr<TensorSummary<T>>> summary_vec;
  std::vector<std::future<void>> summary_future_vec;
  for (uint64_t i = 0; i < actual_threads; i++) {
    uint64_t num_elements_for_thread;
    if (i == actual_threads - 1) {
      num_elements_for_thread = num_elements_ - offset;
    } else {
      num_elements_for_thread = actual_elements_per_thread;
    }
    (void)summary_vec.emplace_back(std::make_unique<TensorSummary<T>>(current_tensor_ptr_ + offset, previous_tensor_ptr,
                                                                      num_elements_for_thread, 0));
    (void)summary_future_vec.emplace_back(
      std::async(std::launch::async, &TensorSummary<T>::TensorStatisticsSingleThread, summary_vec[i].get()));
    offset += num_elements_for_thread;
  }

  // Aggregate results of all chunks
  num_elements_ = 0;  // Let current tensor weight 0 in the aggregation
  for (unsigned int i = 0; i < summary_future_vec.size(); i++) {
    summary_future_vec[i].wait();
    summary_future_vec[i].get();
    auto &cur_summary = *(summary_vec[i]);
    num_elements_ += cur_summary.num_elements_;
    min_ = std::min(min_, cur_summary.min_);
    max_ = std::max(max_, cur_summary.max_);
    double avg_delta = cur_summary.avg_ - avg_;
    avg_ += avg_delta * (cur_summary.num_elements_ / num_elements_);
    neg_zero_count_ += cur_summary.neg_zero_count_;
    pos_zero_count_ += cur_summary.pos_zero_count_;
    neg_inf_count_ += cur_summary.neg_inf_count_;
    pos_inf_count_ += cur_summary.pos_inf_count_;
    inf_count_ += cur_summary.inf_count_;
    nan_count_ += cur_summary.nan_count_;
    zero_count_ += cur_summary.zero_count_;
    l2_calc_.ProcessElement(cur_summary.l2_calc_);
  }
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Process all the elements of the chunked data and calculates the statistics.
 */
template <typename T>
void TensorSummary<T>::TensorStatisticsSingleThread() {
  MeanCalculator mean_calc = MeanCalculator();
  for (size_t i = 0; i < num_elements_; ++i) {
    auto current_value = static_cast<double>(current_tensor_ptr_[i]);
    l2_calc_.ProcessElement(current_value);
    if (std::isnan(current_value)) {
      nan_count_ += 1;
      max_ = current_value;
      min_ = current_value;
      mean_calc.ProcessElement(current_value);
      continue;
    }
    if (std::isinf(current_value)) {
      if (current_value > 0) {
        pos_inf_count_ += 1;
      } else {
        neg_inf_count_ += 1;
      }
    }
    if (current_value == 0.0) {
      zero_count_ += 1;
    }
    // only considering tensor elements with value
    if (std::signbit(current_value) && !(current_value == 0.0)) {
      neg_zero_count_ += 1;
    } else if (!(current_value == 0.0)) {
      pos_zero_count_ += 1;
    }
    max_ = std::max(max_, current_value);
    min_ = std::min(min_, current_value);
    mean_calc.ProcessElement(current_value);
  }
  avg_ = mean_calc.GetMean();
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Returns a tuple with three elements, the first element is a bool and it is true if the watchpoint is
 * hit. The second element is the error_code which is set in this function and the third element is the parameter_list
 * for the watchpoint.
 */
template <typename T>
std::tuple<bool, int, std::vector<DebugServices::parameter_t>> TensorSummary<T>::IsWatchpointHit(
  DebugServices::watchpoint_t wp) {
  auto parameter_list = wp.parameter_list;
  bool hit = false;
  const uint8_t bit_size = 32;
  std::bitset<bit_size> error_code;
  CONDITION_TYPE type = wp.condition.type;
  // bit 0 denotes presence of nan
  (void)error_code.set(0, nan_count_ > 0);
  // bit 1 denotes presence of inf
  (void)error_code.set(1, inf_count_ > 0);

  if (type == CONDITION_TYPE::HAS_NAN) {
    error_code.reset();
    hit = nan_count_ > 0;
  } else if (type == CONDITION_TYPE::HAS_INF) {
    error_code.reset();
    hit = inf_count_ > 0;
  } else if (type == CONDITION_TYPE::GENERAL_OVERFLOW) {
    error_code.reset();
    hit = (nan_count_ + inf_count_) > 0;
  } else if (type == CONDITION_TYPE::NOT_CHANGED && prev_tensor_ptr_ && error_code.none()) {
    hit = all_close_[wp.id]->IsAllClose();
  } else if ((type == CONDITION_TYPE::NOT_CHANGED || type == CONDITION_TYPE::CHANGE_TOO_LARGE ||
              type == CONDITION_TYPE::CHANGE_TOO_SMALL) &&
             !prev_tensor_ptr_) {
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
  if (parameter_name == "param") {
    return StatLookup(wp);
  }
  std::string param_type;
  auto pos = parameter_name.find_last_of('_');
  if (pos != std::string::npos) {
    param_type = parameter_name.substr(0, pos);
  }

  if (param_type == "max") {
    return max_;
  }
  if (param_type == "min") {
    return min_;
  }
  if (param_type == "max_min") {
    return max_ - min_;
  }
  if (param_type == "mean") {
    return current_mean_variance_.GetMean();
  }
  if (param_type == "sd") {
    return current_mean_variance_.GetStandardDeviation();
  }
  if (param_type == "abs_mean") {
    if (means_.find("abs_current_mean") != means_.end()) {
      return means_["abs_current_mean"]->GetMean();
    }
  }
  if (param_type == "abs_mean_update_ratio" && prev_tensor_ptr_) {
    if (means_.find("curr_prev_diff_mean") != means_.end() && means_.find("abs_prev_mean") != means_.end()) {
      return means_["curr_prev_diff_mean"]->GetMean() / (means_["abs_prev_mean"]->GetMean() + epsilon_);
    }
  }
  if (param_type == "range_percentage") {
    if (range_counts_.find(wp.id) != range_counts_.end()) {
      return range_counts_[wp.id]->GetPercentInRange();
    }
  }
  if (param_type == "zero_percentage") {
    return GetZeroValPercent();
  }
  return std::numeric_limits<double_t>::quiet_NaN();
}

template <typename T>
double_t TensorSummary<T>::StatLookup(const DebugServices::watchpoint_t &wp) const {
  CONDITION_TYPE type = wp.condition.type;
  if (type == CONDITION_TYPE::MAX_LT || type == CONDITION_TYPE::MAX_GT) {
    return max_;
  }
  if (type == CONDITION_TYPE::MIN_LT || type == CONDITION_TYPE::MIN_GT) {
    return min_;
  }
  if (type == CONDITION_TYPE::MEAN_LT || type == CONDITION_TYPE::MEAN_GT) {
    return current_mean_variance_.GetMean();
  }
  if (type == CONDITION_TYPE::SD_LT || type == CONDITION_TYPE::SD_GT) {
    return current_mean_variance_.GetStandardDeviation();
  }
  if (type == CONDITION_TYPE::MAX_MIN_GT || type == CONDITION_TYPE::MAX_MIN_LT) {
    return max_ - min_;
  }
  return std::numeric_limits<double_t>::quiet_NaN();
}

template <typename T>
double_t TensorSummary<T>::GetZeroValPercent() const {
  if (num_elements_ == 0) {
    return 0.0;
  }

  return (zero_count_ * 100.0) / num_elements_;
}

template <typename T>
void TensorSummary<T>::InitCalculators(const std::vector<DebugServices::watchpoint_t> &wps) {
  for (auto &wp : wps) {
    auto wp_id = wp.id;
    mean_sd_cal_enabled_ = mean_sd_cal_enabled_ || wp.mean_sd_enabled();
    if (wp.allclose_enabled() && prev_tensor_ptr_) {
      all_close_[wp_id] = std::make_unique<AllCloseCalculator>();
      if (!wp.parameter_list[0].disabled) {
        all_close_[wp_id]->set_rtol(wp.parameter_list[0].value);
      }
      if (!wp.parameter_list[1].disabled) {
        all_close_[wp_id]->set_atol(wp.parameter_list[1].value);
      }
    } else if (wp.range_enabled()) {
      range_counts_[wp_id] = std::make_unique<RangeCountCalculator>();
      if (!wp.parameter_list[0].disabled) {
        range_counts_[wp_id]->set_range_start_inclusive(wp.parameter_list[0].value);
      }
      if (!wp.parameter_list[1].disabled) {
        range_counts_[wp_id]->set_range_end_inclusive(wp.parameter_list[1].value);
      }
    } else if (wp.tensor_update_ratio_mean_enabled() && prev_tensor_ptr_) {
      (void)means_.emplace("curr_prev_diff_mean", std::make_unique<MeanCalculator>());
      (void)means_.emplace("abs_prev_mean", std::make_unique<MeanCalculator>());
    } else if (wp.abs_mean_enabled()) {
      (void)means_.emplace("abs_current_mean", std::make_unique<MeanCalculator>());
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
template class TensorSummary<bfloat16>;
template class TensorSummary<float>;
template class TensorSummary<double>;
template class TensorSummary<bool>;
}  // namespace mindspore
