/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_TENSOR_SUMMARY_H
#define MINDSPORE_TENSOR_SUMMARY_H

#include <vector>
#include <tuple>
#include <memory>
#include <string>

#include "utils/hash_map.h"
#include "debug/debug_services.h"

namespace mindspore {
class RangeCountCalculator {
 public:
  RangeCountCalculator();
  ~RangeCountCalculator() = default;
  void ProcessElement(double element);
  double GetPercentInRange() const;
  void set_range_start_inclusive(double value) { range_start_inclusive = value; }
  void set_range_end_inclusive(double value) { range_end_inclusive = value; }

 private:
  double range_start_inclusive;
  double range_end_inclusive;
  int count;
  int total;
};

class AllCloseCalculator {
 public:
  AllCloseCalculator();
  ~AllCloseCalculator() = default;
  void ProcessElement(double current, double previous);
  bool IsAllClose() const;
  void set_atol(double value) { atol = value; }
  void set_rtol(double value) { rtol = value; }

 private:
  double atol;
  double rtol;
  bool result;
};

class MeanCalculator {
 public:
  MeanCalculator();
  ~MeanCalculator() = default;
  void ProcessElement(double value);
  double GetMean() const;

 protected:
  double mean;
  int count;
};

class VarianceAndMeanCalculator {
 public:
  VarianceAndMeanCalculator();
  ~VarianceAndMeanCalculator() = default;
  void ProcessElement(double value);
  double GetStandardDeviation() const;
  double GetVariance() const;
  double GetMean() const;

 private:
  double mean;
  int count;
  double m2;
};

class L2Calculator {
 public:
  L2Calculator() : squre_sum(0.0) {}
  ~L2Calculator() = default;
  void ProcessElement(double value);
  void ProcessElement(const L2Calculator &other);
  double GetL2Value() const;

 private:
  // save (x^2 + y^2)/y^2, when y > x, to avoid itermidiate value overflow
  // the true l2 value should be sqrt(squre_sum_div_max_ * max_value_^2)
  double squre_sum;
};

class ITensorSummary {
 public:
  enum WatchpointPos { eHitPos = 0, eErrorCodePos = 1, eParamListPos = 2 };
  enum ErrorCode {
    NAN_TENSOR = 0,
    INF_TENSOR = 2,
    NULL_PREV_TENSOR = 4,
    OUT_OF_MEMORY = 8,
    HISTORY_NOT_FOUND = 16,
    NO_VALUE = 32
  };
  virtual ~ITensorSummary() = default;
  virtual void SummarizeTensor(const std::vector<DebugServices::watchpoint_t> &wps) = 0;
  virtual std::tuple<bool, int32_t, std::vector<DebugServices::parameter_t>> IsWatchpointHit(
    DebugServices::watchpoint_t) = 0;
  virtual void TensorStatistics(DbgDataType dtype_value) = 0;
  virtual const bool is_bool() const = 0;
  virtual const double max_value() const = 0;
  virtual const double min_value() const = 0;
  virtual const double avg_value() const = 0;
  virtual const double l2_value() const = 0;

  virtual const uint64_t count() const = 0;
  virtual const uint64_t neg_zero_count() const = 0;
  virtual const uint64_t pos_zero_count() const = 0;
  virtual const uint64_t nan_count() const = 0;
  virtual const uint64_t neg_inf_count() const = 0;
  virtual const uint64_t pos_inf_count() const = 0;
  virtual const uint64_t zero_count() const = 0;
};

template <typename T>
class TensorSummary : public ITensorSummary {
 public:
  TensorSummary() = default;
  ~TensorSummary() override = default;
  TensorSummary(const void *current_tensor_ptr, const void *const previous_tensor_ptr, uint64_t num_elements,
                uint64_t prev_num_elements);
  void SummarizeTensor(const std::vector<DebugServices::watchpoint_t> &wps) override;
  // returns hit, error_code, parameter_list
  std::tuple<bool, int, std::vector<DebugServices::parameter_t>> IsWatchpointHit(
    DebugServices::watchpoint_t wp) override;
  void TensorStatistics(DbgDataType dtype_value) override;
  const bool is_bool() const override { return is_bool_; }
  const double max_value() const override { return max_; }
  const double min_value() const override { return min_; }
  const double avg_value() const override { return avg_; }
  const uint64_t count() const override { return num_elements_; }
  const uint64_t neg_zero_count() const override { return neg_zero_count_; }
  const uint64_t pos_zero_count() const override { return pos_zero_count_; }
  const uint64_t nan_count() const override { return nan_count_; }
  const uint64_t neg_inf_count() const override { return neg_inf_count_; }
  const uint64_t pos_inf_count() const override { return pos_inf_count_; }
  const uint64_t zero_count() const override { return zero_count_; }
  const double l2_value() const override { return l2_calc_.GetL2Value(); }

 private:
  const T *current_tensor_ptr_;
  const T *prev_tensor_ptr_;
  uint64_t num_elements_;
  uint64_t prev_num_elements_;
  double min_;
  double max_;
  double avg_;
  bool is_bool_;
  uint64_t neg_zero_count_;
  uint64_t pos_zero_count_;
  uint64_t pos_inf_count_;
  uint64_t neg_inf_count_;
  uint64_t inf_count_;
  uint64_t nan_count_;
  uint64_t zero_count_;
  double epsilon_;
  bool mean_sd_cal_enabled_;
  VarianceAndMeanCalculator current_mean_variance_;
  L2Calculator l2_calc_;
  mindspore::HashMap<std::string, std::unique_ptr<MeanCalculator>> means_;
  mindspore::HashMap<uint32_t, std::unique_ptr<AllCloseCalculator>> all_close_;
  mindspore::HashMap<uint32_t, std::unique_ptr<RangeCountCalculator>> range_counts_;
  double_t StatLookup(const DebugServices::watchpoint_t &wp) const;
  double_t StatLookup(const std::string &parameter_name, const DebugServices::watchpoint_t &wp);
  double_t GetZeroValPercent() const;
  void TensorStatisticsSingleThread();
  void InitCalculators(const std::vector<DebugServices::watchpoint_t> &);
};
}  // namespace mindspore
#endif  // MINDSPORE_TENSOR_SUMMARY_H
