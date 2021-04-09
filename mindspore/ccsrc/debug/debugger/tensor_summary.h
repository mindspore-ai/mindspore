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
  bool IsAllClose();
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
  double GetMean();

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
  double GetVariance();
  double GetMean();

 private:
  double mean;
  int count;
  double m2;
};

class ITensorSummary {
 public:
  virtual ~ITensorSummary() = default;
  virtual void SummarizeTensor(const std::vector<DebugServices::watchpoint_t> &) = 0;
  virtual std::tuple<bool, int32_t, std::vector<DebugServices::parameter_t>> IsWatchpointHit(
    DebugServices::watchpoint_t) = 0;
};

template <typename T>
class TensorSummary : public ITensorSummary {
 public:
  TensorSummary() = default;
  ~TensorSummary() override = default;
  TensorSummary(void *, void *const, uint32_t);
  void SummarizeTensor(const std::vector<DebugServices::watchpoint_t> &) override;
  // returns hit, error_code, parameter_list
  std::tuple<bool, int, std::vector<DebugServices::parameter_t>> IsWatchpointHit(DebugServices::watchpoint_t) override;

 private:
  T *current_tensor_ptr;
  T *prev_tensor_ptr;
  uint32_t num_elements;
  double min;
  double max;
  uint32_t inf_count;
  uint32_t nan_count;
  uint32_t zero_count;
  double epsilon;
  bool mean_sd_cal_enabled;
  VarianceAndMeanCalculator current_mean_variance;
  std::unordered_map<std::string, std::unique_ptr<MeanCalculator>> means;
  std::unordered_map<uint32_t, std::unique_ptr<AllCloseCalculator>> all_close;
  std::unordered_map<uint32_t, std::unique_ptr<RangeCountCalculator>> range_counts;
  double_t StatLookup(const DebugServices::watchpoint_t &);
  double_t StatLookup(const std::string &, const DebugServices::watchpoint_t &);
  double_t GetZeroValPercent();
  void InitCalculators(const std::vector<DebugServices::watchpoint_t> &);
};
}  // namespace mindspore
#endif  // MINDSPORE_TENSOR_SUMMARY_H
