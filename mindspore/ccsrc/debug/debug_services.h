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
    SD_LT
  };

  typedef struct condition {
    CONDITION_TYPE type;
    float parameter = 0;
    std::string comparison;
  } condition_t;

  typedef struct watchpoint {
    unsigned int id;
    condition_t condition;
    std::vector<std::tuple<std::string, bool>> check_node_list;
    size_t location = 0;

    bool IsNodeIncluded(const std::string &tensor_name) {
      std::string node_name = tensor_name.substr(0, tensor_name.find_first_of(':'));
      for (auto check_node : check_node_list) {
        std::string w_name = std::get<0>(check_node);
        bool w_type = std::get<1>(check_node);
        if ((w_type && (tensor_name.find(w_name) == location || w_name == "*")) || (!w_type && node_name == w_name)) {
          return true;
        }
      }
      return false;
    }

    bool min_max_enabled() {
      return condition.type == MAX_LT || condition.type == MAX_GT || condition.type == MIN_LT ||
             condition.type == MIN_GT || condition.type == MAX_MIN_LT || condition.type == MAX_MIN_GT;
    }
    // inf or nan related condition set
    bool inf_nan_enabled() { return condition.type == HAS_INF || condition.type == HAS_NAN; }
    // mean or sd related condition set
    bool mean_sd_enabled() {
      return condition.type == MEAN_LT || condition.type == MEAN_GT || condition.type == SD_LT ||
             condition.type == SD_GT;
    }
  } watchpoint_t;

  struct tensor_stats {
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::lowest();
    bool has_inf = false;
    bool has_nan = false;
    unsigned int n = 0;
    double mean = 0.0;
    double m2 = 0.0;

    double statLookup(CONDITION_TYPE type) const {
      if (type == MAX_GT || type == MAX_LT) return max;
      if (type == MIN_GT || type == MIN_LT) return min;
      if (type == MAX_MIN_GT || type == MAX_MIN_LT) return (max - min);
      if (type == MEAN_GT || type == MEAN_LT) return mean;
      if (type == SD_GT || type == SD_LT) return getStandardDeviation();
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
                     const std::vector<std::tuple<std::string, bool>> &check_node_list);

  void RemoveWatchpoint(unsigned int id);

  void CheckWatchpoints(std::vector<std::string> *name, std::vector<std::string> *slot, std::vector<int> *condition,
                        std::vector<unsigned int> *watchpoint_id, const std::vector<std::string> &op_overflows,
                        const std::vector<std::shared_ptr<TensorData>> &tensor_list);

  void ReadNodesTensors(std::vector<std::string> name, std::vector<std::string> *ret_name,
                        std::vector<char *> *data_ptr, std::vector<unsigned int> *data_size,
                        std::vector<TypePtr> *dtype, std::vector<std::vector<int>> *shape);

  bool IsWatchPoint(std::string kernel_name, std::unordered_map<unsigned int, watchpoint_t> watchpoint_table);

  TensorLoader *tensor_loader() const;

  std::unordered_map<unsigned int, watchpoint_t> GetWatchpointTable();

 private:
  std::mutex lock_;

  std::unordered_map<unsigned int, watchpoint_t> watchpoint_table;
  std::vector<std::string> condition_label = {"HAS_NAN", "HAS_INF", "IS_OVERFLOW", "MAX_GT",     "MAX_LT",
                                              "MIN_GT",  "MIN_LT",  "MAX_MIN_GT",  "MAX_MIN_LT", "MEAN_GT",
                                              "MEAN_LT", "SD_GT",   "SD_LT"};

  TensorLoader *tensor_loader_;

  template <typename T>
  static tensor_stats SummarizeTensor(const T *start, unsigned int n, bool need_min_max, bool need_mean_sd);
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_
