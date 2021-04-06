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

#include "debug/debug_services.h"
namespace py = pybind11;

typedef struct parameter {
  parameter(const std::string &name, bool disabled, double value, bool hit, double actual_value)
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
} parameter_t;

typedef struct watchpoint_hit {
  watchpoint_hit(const std::string &name, uint32_t slot, int condition, uint32_t watchpoint_id,
                 const std::vector<parameter_t> &parameters, int32_t error_code, uint32_t device_id,
                 uint32_t root_graph_id)
      : name(name),
        slot(slot),
        condition(condition),
        watchpoint_id(watchpoint_id),
        parameters(parameters),
        error_code(error_code),
        device_id(device_id),
        root_graph_id(root_graph_id) {}
  const std::string get_name() const { return name; }
  const uint32_t get_slot() const { return slot; }
  const int get_condition() const { return condition; }
  const uint32_t get_watchpoint_id() const { return watchpoint_id; }
  const std::vector<parameter_t> get_parameters() const { return parameters; }
  const int32_t get_error_code() const { return error_code; }
  const uint32_t get_device_id() const { return device_id; }
  const uint32_t get_root_graph_id() const { return root_graph_id; }
  std::string name;
  uint32_t slot;
  int condition;
  uint32_t watchpoint_id;
  std::vector<parameter_t> parameters;
  int32_t error_code;
  uint32_t device_id;
  uint32_t root_graph_id;
} watchpoint_hit_t;

typedef struct tensor_info {
  tensor_info(const std::string &node_name, uint32_t slot, uint32_t iteration, uint32_t device_id,
              uint32_t root_graph_id, bool is_parameter)
      : node_name(node_name),
        slot(slot),
        iteration(iteration),
        device_id(device_id),
        root_graph_id(root_graph_id),
        is_parameter(is_parameter) {}
  const std::string get_node_name() const { return node_name; }
  const uint32_t get_slot() const { return slot; }
  const uint32_t get_iteration() const { return iteration; }
  const uint32_t get_device_id() const { return device_id; }
  const uint32_t get_root_graph_id() const { return root_graph_id; }
  const bool get_is_parameter() const { return is_parameter; }
  std::string node_name;
  uint32_t slot;
  uint32_t iteration;
  uint32_t device_id;
  uint32_t root_graph_id;
  bool is_parameter;
} tensor_info_t;

typedef struct tensor_data {
  tensor_data(char *data_ptr, uint64_t data_size, int dtype, const std::vector<int64_t> &shape)
      : data_size(data_size), dtype(dtype), shape(shape) {
    if (data_ptr != NULL) {
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
} tensor_data_t;

class DbgServices {
 private:
  DebugServices *debug_services;

 public:
  explicit DbgServices(bool verbose = false);

  DbgServices(const DbgServices &other);

  DbgServices &operator=(const DbgServices &other);

  ~DbgServices();

  int32_t Initialize(std::string net_name, std::string dump_folder_path, bool is_sync_mode);

  int32_t AddWatchpoint(
    unsigned int id, unsigned int watch_condition,
    std::map<std::string, std::map<std::string, std::variant<bool, std::vector<std::string>>>> check_nodes,
    std::vector<parameter_t> parameter_list);

  int32_t RemoveWatchpoint(unsigned int id);

  std::vector<watchpoint_hit_t> CheckWatchpoints(unsigned int iteration);

  std::vector<tensor_data_t> ReadTensors(std::vector<tensor_info_t> info);

  std::string GetVersion();
};

#endif  // DEBUG_DBG_SERVICES_H_
