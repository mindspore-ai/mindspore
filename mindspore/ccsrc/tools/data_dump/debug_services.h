/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEBUG_SERVICES_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEBUG_SERVICES_H_

#include <cmath>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/float16.h"
#include "tools/data_dump/tensor_load.h"
#include "tools/tensor_data.h"

namespace mindspore {
class DebugServices {
 public:
  DebugServices();

  DebugServices(const DebugServices &other);

  DebugServices &operator=(const DebugServices &other);

  ~DebugServices() = default;

  struct TensorBase {
    TensorBase(uint64_t data_size, int dtype, const std::vector<int64_t> &shape)
        : data_size(data_size), dtype(dtype), shape(shape) {}
    TensorBase() = default;
    uint64_t data_size = 0;
    int dtype = 0;
    std::vector<int64_t> shape;
  };

  struct TensorStat {
    TensorStat(uint64_t data_size, int dtype, const std::vector<int64_t> &shape, bool is_bool, double max_value,
               double min_value, double avg_value, uint64_t count, uint64_t neg_zero_count, uint64_t pos_zero_count,
               uint64_t nan_count, uint64_t neg_inf_count, uint64_t pos_inf_count, uint64_t zero_count, double l2_value,
               std::string sha1, std::string md5 = "")
        : data_size(data_size),
          dtype(dtype),
          shape(shape),
          is_bool(is_bool),
          max_value(max_value),
          min_value(min_value),
          avg_value(avg_value),
          count(count),
          neg_zero_count(neg_zero_count),
          pos_zero_count(pos_zero_count),
          nan_count(nan_count),
          neg_inf_count(neg_inf_count),
          pos_inf_count(pos_inf_count),
          zero_count(zero_count),
          l2_value(l2_value),
          sha1(sha1),
          md5(md5) {}

    TensorStat() = default;

    uint64_t data_size = 0;
    int dtype = 0;
    std::vector<int64_t> shape;
    bool is_bool = false;
    double max_value = std::numeric_limits<double>::lowest();
    double min_value = std::numeric_limits<double>::max();
    double avg_value = 0.0;
    uint64_t count = 0;
    uint64_t neg_zero_count = 0;
    uint64_t pos_zero_count = 0;
    uint64_t nan_count = 0;
    uint64_t neg_inf_count = 0;
    uint64_t pos_inf_count = 0;
    uint64_t zero_count = 0;
    double l2_value = 0.0;
    std::string sha1 = "";
    std::string md5 = "";
    std::map<std::string, std::string> header_item_map;
    std::string DoubleToString(double value) {
      std::ostringstream ss;
      ss << value;
      return ss.str();
    }
    void UpdateHeaderItemMap() {
      header_item_map = {{"max", DoubleToString(max_value)},
                         {"min", DoubleToString(min_value)},
                         {"avg", DoubleToString(avg_value)},
                         {"count", std::to_string(count)},
                         {"negative zero count", std::to_string(neg_zero_count)},
                         {"positive zero count", std::to_string(pos_zero_count)},
                         {"nan count", std::to_string(nan_count)},
                         {"negative inf count", std::to_string(neg_inf_count)},
                         {"positive inf count", std::to_string(pos_inf_count)},
                         {"zero count", std::to_string(zero_count)},
                         {"l2norm", DoubleToString(l2_value)},
                         {"sha1", sha1},
                         {"md5", md5}};
    }
  };

  static TensorStat GetTensorStatistics(const std::shared_ptr<TensorData> &tensor);

  std::shared_ptr<TensorData> GetTensor(const std::string &tensor_name) const;

  void EmptyCurrentTensor();

  bool DumpTensorToFile(const std::string &filepath, const std::string &tensor_name, size_t slot) const;

  bool LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev);

  void ResetLoadedTensors();

  bool TensorExistsInCurrent(const std::string &tensor_name);

 private:
  std::shared_ptr<TensorLoader> tensor_loader_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEBUG_SERVICES_H_
