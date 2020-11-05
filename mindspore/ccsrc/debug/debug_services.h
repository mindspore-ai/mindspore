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
#ifndef MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_
#define MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_

#include <math.h>
#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <mutex>
#include <limits>
#include "debug/tensor_load.h"
#include "debug/tensor_data.h"
#include "ir/dtype.h"

namespace mindspore {
class DebugServices {
 public:
  DebugServices();

  DebugServices(const DebugServices &other);

  DebugServices &operator=(const DebugServices &other);

  ~DebugServices();

  enum CONDITION_TYPE {
    HAS_NAN,
    HAS_INF,
    IS_OVERFLOW,
    MAX_GT,
    MAX_LT,
    MIN_GT,
    MIN_LT,
    MAX_MIN_GT,
    MAX_MIN_LT,
    MEAN_GT,
    MEAN_LT,
    SD_GT,
    SD_LT,
    GENERAL_OVERFLOW,
    INIT,
    TOO_LARGE,
    TOO_SMALL,
    ALL_ZERO,
    CHANGE_TOO_LARGE,
    CHANGE_TOO_SMALL,
    NOT_CHANGED
  };

  enum STAT_TYPE {
    STAT_MIN,
    STAT_MAX,
    STAT_MEAN,
    STAT_ZERO_PERCENTAGE,
    STAT_TENSOR_UPDATE_RATIO_MEAN,
    STAT_ALLCLOSE,
    STAT_ABS_MEAN
  };

  typedef struct condition {
    CONDITION_TYPE type;
    float parameter = 0;
    std::string comparison;
  } condition_t;

  typedef struct parameter {
    std::string name;
    bool disabled;
    double_t value;
    bool hit;
  } parameter_t;

  typedef struct watchpoint {
    unsigned int id;
    condition_t condition;
    std::vector<std::tuple<std::string, bool>> check_node_list;
    std::vector<parameter_t> parameter_list;
    size_t location = 0;

    bool IsNodeIncluded(const std::string &tensor_name) {
      std::string node_name = tensor_name.substr(0, tensor_name.find_first_of(':'));
      for (auto check_node : check_node_list) {
        std::string w_name = std::get<0>(check_node);
        bool w_type = std::get<1>(check_node);
        auto found = w_name.find_last_of('/');
        if (found != std::string::npos && w_name.substr(found + 1) == tensor_name) return true;
        if ((w_type && (tensor_name.find(w_name) == location || w_name == "*")) || (!w_type && node_name == w_name)) {
          return true;
        }
      }
      return false;
    }

    bool min_max_enabled() {
      return condition.type == MAX_LT || condition.type == MAX_GT || condition.type == MIN_LT ||
             condition.type == MIN_GT || condition.type == MAX_MIN_LT || condition.type == MAX_MIN_GT ||
             (condition.type == INIT && (!parameter_list[1].disabled || !parameter_list[2].disabled)) ||
             (condition.type == TOO_LARGE && (!parameter_list[1].disabled || !parameter_list[2].disabled)) ||
             (condition.type == TOO_SMALL && (!parameter_list[1].disabled || !parameter_list[2].disabled));
    }
    // inf or nan related condition set
    bool inf_nan_enabled() {
      return condition.type == HAS_INF || condition.type == HAS_NAN || condition.type == GENERAL_OVERFLOW;
    }
    // mean or sd related condition set
    bool mean_sd_enabled() {
      return condition.type == MEAN_LT || condition.type == MEAN_GT || condition.type == SD_LT ||
             condition.type == SD_GT || (condition.type == TOO_LARGE && !parameter_list[3].disabled) ||
             (condition.type == TOO_SMALL && !parameter_list[3].disabled);
    }
    bool abs_mean_enabled() {
      return (condition.type == TOO_LARGE && !parameter_list[0].disabled) ||
             (condition.type == TOO_SMALL && !parameter_list[0].disabled);
    }
    bool zero_percentage_enabled() { return condition.type == ALL_ZERO || condition.type == INIT; }
    bool tensor_update_ratio_mean_enabled() {
      return condition.type == CHANGE_TOO_LARGE || condition.type == CHANGE_TOO_SMALL;
    }
    bool allclose_enabled() { return condition.type == NOT_CHANGED; }
  } watchpoint_t;

  struct tensor_stats {
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::lowest();
    bool has_inf = false;
    bool has_nan = false;
    unsigned int n = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double zero_percentage = 0.0;
    double tensor_update_ratio_mean = -1;
    bool allclose = false;
    double abs_mean = 0.0;

    double statLookup(CONDITION_TYPE type) const {
      if (type == MAX_GT || type == MAX_LT) return max;
      if (type == MIN_GT || type == MIN_LT) return min;
      if (type == MAX_MIN_GT || type == MAX_MIN_LT) return (max - min);
      if (type == MEAN_GT || type == MEAN_LT) return mean;
      if (type == SD_GT || type == SD_LT) return getStandardDeviation();
      return std::numeric_limits<double>::quiet_NaN();
    }

    double parmLookup(STAT_TYPE type) const {
      if (type == STAT_MAX) return max;
      if (type == STAT_MIN) return min;
      if (type == STAT_MEAN) return mean;
      if (type == STAT_ZERO_PERCENTAGE) return zero_percentage;
      if (type == STAT_TENSOR_UPDATE_RATIO_MEAN) return tensor_update_ratio_mean;
      if (type == STAT_ALLCLOSE) return allclose;
      if (type == STAT_ABS_MEAN) return abs_mean;
      return std::numeric_limits<double>::quiet_NaN();
    }

    double getMean() const { return mean; }

    double getVariance() const {
      if (n > 1) {
        return m2 / (n - 1);
      } else {
        return 0.0;
      }
    }

    double getStandardDeviation() const { return sqrt(getVariance()); }
  };

  void AddWatchpoint(unsigned int id, unsigned int watch_condition, float parameter,
                     const std::vector<std::tuple<std::string, bool>> &check_node_list,
                     const std::vector<parameter_t> &parameter_list);

  void RemoveWatchpoint(unsigned int id);

  void CheckWatchpoints(std::vector<std::string> *name, std::vector<std::string> *slot, std::vector<int> *condition,
                        std::vector<unsigned int> *watchpoint_id, std::vector<std::vector<parameter_t>> *parameters,
                        const std::vector<std::string> &op_overflows,
                        const std::vector<std::shared_ptr<TensorData>> &tensor_list, bool init_dbg_suspend);

  void ReadNodesTensors(std::vector<std::string> name, std::vector<std::string> *ret_name,
                        std::vector<char *> *data_ptr, std::vector<unsigned int> *data_size,
                        std::vector<TypePtr> *dtype, std::vector<std::vector<int64_t>> *shape);

  bool IsWatchPoint(std::string kernel_name, const CNodePtr &kernel = nullptr);

  bool IsWatchPointNodeInput(std::string w_name, const CNodePtr &kernel);

  void AddWeightsBiasInputs(std::vector<std::shared_ptr<TensorData>> *tensor_list, const CNodePtr &kernel);

  TensorLoader *tensor_loader() const;

  std::unordered_map<unsigned int, watchpoint_t> GetWatchpointTable();

 private:
  std::mutex lock_;

  std::unordered_map<unsigned int, watchpoint_t> watchpoint_table;
  std::vector<std::string> condition_label = {
    "HAS_NAN",    "HAS_INF",   "IS_OVERFLOW", "MAX_GT",           "MAX_LT",
    "MIN_GT",     "MIN_LT",    "MAX_MIN_GT",  "MAX_MIN_LT",       "MEAN_GT",
    "MEAN_LT",    "SD_GT",     "SD_LT",       "GENERAL_OVERFLOW", "INIT",
    "TOO_LARGE",  "TOO_SMALL", "ALL_ZERO",    "CHANGE_TOO_LARGE", "CHANGE_TOO_SMALL",
    "NOT_CHANGED"};

  TensorLoader *tensor_loader_;

  template <typename T>
  static tensor_stats SummarizeTensor(const T *start, const T *start_prev, unsigned int n, bool need_min_max,
                                      bool need_mean_sd, bool need_zero_percentage, bool need_tensor_update_ratio_mean,
                                      bool need_allclose, bool need_abs_mean_sd);
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_
