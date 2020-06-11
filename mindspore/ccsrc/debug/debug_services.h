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

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <mutex>
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

  void add_watchpoint(unsigned int id, unsigned int watch_condition,
                      const std::vector<std::tuple<std::string, bool>> &check_node_list);

  void remove_watchpoint(unsigned int id);

  void check_watchpoints(std::vector<std::string> *name, std::vector<std::string> *slot, std::vector<char *> *data_ptr,
                         std::vector<unsigned int> *data_size, std::vector<int> *condition,
                         std::vector<unsigned int> *wacthpoint_id);

  void read_nodes_tensors(std::vector<std::string> name, std::vector<std::string> *ret_name,
                          std::vector<char *> *data_ptr, std::vector<unsigned int> *data_size,
                          std::vector<TypePtr> *dtype, std::vector<std::vector<int>> *shape);

  TensorLoader *get_tensor_loader() const;

 private:
  typedef struct condition_no_param {
    bool enabled = false;
  } condition_no_param_t;

  typedef struct condition_with_param {
    bool enabled = false;
    float parameter = 0;
  } condition_with_param_t;

  typedef struct conditions {
    condition_no_param_t inf;
    condition_no_param_t neg_inf;
    condition_no_param_t nan;
    condition_with_param_t max_below;
    condition_with_param_t max_above;
    condition_with_param_t min_below;
    condition_with_param_t min_above;
    condition_with_param_t max_minus_min_below;
    condition_with_param_t max_minus_min_above;
    condition_with_param_t mean_below;
    condition_with_param_t mean_above;
    condition_with_param_t std_dev_below;
    condition_with_param_t std_dev_above;
  } conditions_t;

  typedef struct watchpoint {
    unsigned int id;
    conditions_t conditions;
    std::vector<std::tuple<std::string, bool>> check_node_list;
  } watchpoint_t;

  std::mutex lock_;

  std::unordered_map<unsigned int, watchpoint_t> watchpoint_table;

  TensorLoader *tensor_loader_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_
