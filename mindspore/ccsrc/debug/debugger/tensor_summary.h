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
#ifndef MINDSPORE_TENSOR_SUMMARY_H
#define MINDSPORE_TENSOR_SUMMARY_H

#include <vector>
#include <unordered_map>
#include <tuple>
#include <memory>
#include <string>

#include "debug/debug_services.h"

#ifdef ONLINE_DBG_MODE
namespace mindspore {
#endif
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
  double GetStandardDeviation();
  double GetVariance() const;
  double GetMean() const;

 private:
  double mean;
  int count;
  double m2;
};

class ITensorSummary {
 public:
  enum WatchpointPos { eHitPos = 0, eErrorCodePos = 1, eParamListPos = 2 };
  virtual ~ITensorSummary() = default;
  virtual void SummarizeTensor(const std::vector<DebugServices::watchpoint_t> &) = 0;
  virtual std::tuple<bool, int32_t, std::vector<DebugServices::parameter_t>> IsWatchpointHit(
    DebugServices::watchpoint_t) = 0;
  virtual void TensorStatistics(DbgDataType) = 0;
  virtual const bool is_bool() const = 0;
  virtual const double max_value() const = 0;
  virtual const double min_value() const = 0;
  virtual const double avg_value() const = 0;
  virtual const int count() const = 0;
  virtual const int neg_zero_count() const = 0;
  virtual const int pos_zero_count() const = 0;
  virtual const int nan_count() const = 0;
  virtual const int neg_inf_count() const = 0;
  virtual const int pos_inf_count() const = 0;
  virtual const int zero_count() const = 0;
};

template <typename T>
class TensorSummary : public ITensorSummary {
 public:
  TensorSummary() = default;
  ~TensorSummary() override = default;
  TensorSummary(const void *, const void *, uint32_t, uint32_t);
  void SummarizeTensor(const std::vector<DebugServices::watchpoint_t> &) override;
  // returns hit, error_code, parameter_list
  std::tuple<bool, int, std::vector<DebugServices::parameter_t>> IsWatchpointHit(DebugServices::watchpoint_t) override;
  void TensorStatistics(DbgDataType) override;
  const bool is_bool() const override { return is_bool_; }
  const double max_value() const override { return max_; }
  const double min_value() const override { return min_; }
  const double avg_value() const override { return avg_; }
  const int count() const override { return num_elements_; }
  const int neg_zero_count() const override { return neg_zero_count_; }
  const int pos_zero_count() const override { return pos_zero_count_; }
  const int nan_count() const override { return nan_count_; }
  const int neg_inf_count() const override { return neg_inf_count_; }
  const int pos_inf_count() const override { return pos_inf_count_; }
  const int zero_count() const override { return zero_count_; }

 private:
  const T *current_tensor_ptr_;
  const T *prev_tensor_ptr_;
  uint32_t num_elements_;
  uint32_t prev_num_elements_;
  double min_;
  double max_;
  double avg_;
  bool is_bool_;
  uint32_t neg_zero_count_;
  uint32_t pos_zero_count_;
  uint32_t pos_inf_count_;
  uint32_t neg_inf_count_;
  uint32_t inf_count_;
  uint32_t nan_count_;
  uint32_t zero_count_;
  double epsilon_;
  bool mean_sd_cal_enabled_;
  VarianceAndMeanCalculator current_mean_variance_;
  std::unordered_map<std::string, std::unique_ptr<MeanCalculator>> means_;
  std::unordered_map<uint32_t, std::unique_ptr<AllCloseCalculator>> all_close_;
  std::unordered_map<uint32_t, std::unique_ptr<RangeCountCalculator>> range_counts_;
  double_t StatLookup(const DebugServices::watchpoint_t &);
  double_t StatLookup(const std::string &, const DebugServices::watchpoint_t &);
  double_t GetZeroValPercent();
  void InitCalculators(const std::vector<DebugServices::watchpoint_t> &);
};
#ifdef ONLINE_DBG_MODE
}  // namespace mindspore
#endif
#endif  // MINDSPORE_TENSOR_SUMMARY_H
