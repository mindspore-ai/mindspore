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
#ifndef DEBUG_DBG_SERVICES_H_
#define DEBUG_DBG_SERVICES_H_

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <tuple>
#include <iostream>
#include <variant>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "utils/ms_utils.h"
#include "debug/debug_services.h"
namespace py = pybind11;
namespace common = mindspore::common;

struct parameter_t {
  parameter_t(const std::string &name, bool disabled, double value, bool hit, double actual_value)
      : name(name), disabled(disabled), value(value), hit(hit), actual_value(actual_value) {}
  const std::string get_name() const { return name; }
  const bool get_disabled() const { return disabled; }
  const double get_value() const { return value; }
  const bool get_hit() const { return hit; }
  const double get_actual_value() const { return actual_value; }
  std::string name;
  bool disabled;
  double value;
  bool hit;
  double actual_value;
};

struct watchpoint_hit_t {
  watchpoint_hit_t(const std::string &name, uint32_t slot, int condition, uint32_t watchpoint_id,
                   const std::vector<parameter_t> &parameters, int32_t error_code, uint32_t rank_id,
                   uint32_t root_graph_id)
      : name(name),
        slot(slot),
        condition(condition),
        watchpoint_id(watchpoint_id),
        parameters(parameters),
        error_code(error_code),
        rank_id(rank_id),
        root_graph_id(root_graph_id) {}
  const std::string get_name() const { return name; }
  const uint32_t get_slot() const { return slot; }
  const int get_condition() const { return condition; }
  const uint32_t get_watchpoint_id() const { return watchpoint_id; }
  const std::vector<parameter_t> get_parameters() const { return parameters; }
  const int32_t get_error_code() const { return error_code; }
  const uint32_t get_rank_id() const { return rank_id; }
  const uint32_t get_root_graph_id() const { return root_graph_id; }
  std::string name;
  uint32_t slot;
  int condition;
  uint32_t watchpoint_id;
  std::vector<parameter_t> parameters;
  int32_t error_code;
  uint32_t rank_id;
  uint32_t root_graph_id;
};

struct tensor_info_t {
  tensor_info_t(const std::string &node_name, uint32_t slot, uint32_t iteration, uint32_t rank_id,
                uint32_t root_graph_id, bool is_output)
      : node_name(node_name),
        slot(slot),
        iteration(iteration),
        rank_id(rank_id),
        root_graph_id(root_graph_id),
        is_output(is_output) {}
  const std::string get_node_name() const { return node_name; }
  const uint32_t get_slot() const { return slot; }
  const uint32_t get_iteration() const { return iteration; }
  const uint32_t get_rank_id() const { return rank_id; }
  const uint32_t get_root_graph_id() const { return root_graph_id; }
  const bool get_is_output() const { return is_output; }
  std::string node_name;
  uint32_t slot;
  uint32_t iteration;
  uint32_t rank_id;
  uint32_t root_graph_id;
  bool is_output;
};

struct tensor_data_t {
  tensor_data_t(const char *data_ptr, uint64_t data_size, int dtype, const std::vector<int64_t> &shape)
      : data_size(data_size), dtype(dtype), shape(shape) {
    if (data_ptr != nullptr) {
      this->data_ptr = py::bytes(data_ptr, data_size);
    } else {
      this->data_ptr = py::bytes();
    }
  }
  const py::bytes get_data_ptr() const { return data_ptr; }
  const uint64_t get_data_size() const { return data_size; }
  const int get_dtype() const { return dtype; }
  const std::vector<int64_t> &get_shape() const { return shape; }
  py::bytes data_ptr;
  uint64_t data_size;
  int dtype;
  std::vector<int64_t> shape;
};

struct TensorBaseData {
  TensorBaseData(uint64_t data_size, int dtype, const std::vector<int64_t> &shape)
      : data_size_(data_size), dtype_(dtype), shape_(shape) {}

  const uint64_t data_size() const { return data_size_; }
  const int dtype() const { return dtype_; }
  const std::vector<int64_t> &shape() const { return shape_; }
  uint64_t data_size_;
  int dtype_;
  std::vector<int64_t> shape_;
};

struct TensorStatData {
  TensorStatData(uint64_t data_size, int dtype, const std::vector<int64_t> &shape, bool is_bool, double max_value,
                 double min_value, double avg_value, int count, int neg_zero_count, int pos_zero_count, int nan_count,
                 int neg_inf_count, int pos_inf_count, int zero_count)
      : data_size_(data_size),
        dtype_(dtype),
        shape_(shape),
        is_bool_(is_bool),
        max_value_(max_value),
        min_value_(min_value),
        avg_value_(avg_value),
        count_(count),
        neg_zero_count_(neg_zero_count),
        pos_zero_count_(pos_zero_count),
        nan_count_(nan_count),
        neg_inf_count_(neg_inf_count),
        pos_inf_count_(pos_inf_count),
        zero_count_(zero_count) {}

  const uint64_t data_size() const { return data_size_; }
  const int dtype() const { return dtype_; }
  const std::vector<int64_t> &shape() const { return shape_; }
  const bool is_bool() const { return is_bool_; }
  const double max_value() const { return max_value_; }
  const double min_value() const { return min_value_; }
  const double avg_value() const { return avg_value_; }
  const int count() const { return count_; }
  const int neg_zero_count() const { return neg_zero_count_; }
  const int pos_zero_count() const { return pos_zero_count_; }
  const int nan_count() const { return nan_count_; }
  const int neg_inf_count() const { return neg_inf_count_; }
  const int pos_inf_count() const { return pos_inf_count_; }
  const int zero_count() const { return zero_count_; }

  uint64_t data_size_;
  int dtype_;
  std::vector<int64_t> shape_;
  bool is_bool_;
  double max_value_;
  double min_value_;
  double avg_value_;
  int count_;
  int neg_zero_count_;
  int pos_zero_count_;
  int nan_count_;
  int neg_inf_count_;
  int pos_inf_count_;
  int zero_count_;
};

class DbgServices {
 public:
  DbgServices();

  DbgServices(const DbgServices &other);

  DbgServices &operator=(const DbgServices &other);

  ~DbgServices();

  int32_t Initialize(const std::string net_name, const std::string dump_folder_path, bool is_sync_mode,
                     uint64_t max_mem_usage);

  int32_t AddWatchpoint(
    unsigned int id, unsigned int watch_condition,
    std::map<std::string, std::map<std::string, std::variant<bool, std::vector<std::string>>>> check_nodes,
    std::vector<parameter_t> parameter_list);

  int32_t RemoveWatchpoint(unsigned int id);

  std::vector<watchpoint_hit_t> CheckWatchpoints(unsigned int iteration);

  std::vector<std::shared_ptr<TensorData>> ReadTensorsUtil(std::vector<tensor_info_t> info);

  std::vector<tensor_data_t> ReadTensors(const std::vector<tensor_info_t> info);

  std::vector<TensorBaseData> ReadTensorsBase(const std::vector<tensor_info_t> info);

  std::vector<TensorStatData> ReadTensorsStat(const std::vector<tensor_info_t> info);

  std::string GetVersion() const;

 private:
  std::shared_ptr<DebugServices> debug_services_ = nullptr;
};

#endif  // DEBUG_DBG_SERVICES_H_
