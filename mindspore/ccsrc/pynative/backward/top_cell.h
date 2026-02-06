/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <stack>
#include <set>
#include <map>

#include "utils/convert_utils.h"
#include "tools/profiler/profiler.h"
#include "utils/hash_map.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "mindspore/ccsrc/utils/base_ref_py.h"
#include "ir/anf.h"
#include "include/frontend/jit/ps/resource_interface.h"
#include "pynative/utils/base.h"
#include "utils/ms_context.h"
#include "include/utils/pynative/variable.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
class GradExecutor;
struct PyNGraphInfo {
  OrderedMap<std::string, ParameterPtr> input_params;   // Hold input parameters
  OrderedMap<std::string, ParameterPtr> weight_params;  // Hold weights parameters
  // Hold op op output or combination of output
  mindspore::HashMap<std::string, std::pair<AnfNodePtr, std::vector<int64_t>>> node_map;
};
using GraphInfoPtr = std::shared_ptr<PyNGraphInfo>;
class TopCellInfo {
 public:
  TopCellInfo() = default;
  ~TopCellInfo() = default;
  TopCellInfo(bool is_high_order_top_cell, size_t grad_order, std::string cellid, std::string already_run_cell_id,
              pipeline::ResourcePtr r, FuncGraphPtr fg, size_t reserve_size)
      : is_high_order_top_cell_(is_high_order_top_cell),
        grad_order_(grad_order),
        cell_id_(std::move(cellid)),
        ready_run_cell_id_(std::move(already_run_cell_id)),
        fg_(std::move(fg)) {}

  inline size_t grad_order() const { return grad_order_; }
  inline bool is_high_order_top_cell() const { return is_high_order_top_cell_; }
  inline FuncGraphPtr fg() const { return fg_; }
  inline void ClearForwardGraph() { fg_ = nullptr; }
  inline bool jit_out_has_dict() const { return jit_out_has_dict_; }
  inline void set_jit_out_has_dict(bool jit_out_has_dict) { jit_out_has_dict_ = jit_out_has_dict; }
  inline const std::string &cell_id() const { return cell_id_; }
  inline const std::string &ready_run_cell_id() const { return ready_run_cell_id_; }
  inline void set_input_args_id(const std::string &input_args_id) { input_args_id_ = input_args_id; }
  inline const std::string &input_args_id() const { return input_args_id_; }
  inline void SetGraphInfoMap(const FuncGraphPtr &fg, const GraphInfoPtr &graph_info) {
    graph_info_map_[fg] = graph_info;
  }
  inline const OrderedMap<FuncGraphPtr, GraphInfoPtr> &graph_info_map() const { return graph_info_map_; }
  inline size_t op_index() const { return op_index_; }
  inline InputArgsInfoPtr input_args_info() { return input_args_info_; }
  inline void set_input_args_info(const InputArgsInfoPtr &input_args_info) { input_args_info_ = input_args_info; }
  void DeleteParamNodeInfo(const FuncGraphPtr &g, const std::string &id) const;
  void SetParamNodeMapInGraphInfoMap(const std::string &id, const ParameterPtr &param, bool is_weight = false) const;
  void SetNodeMapInGraphInfoMap(const std::string &id, const AnfNodePtr &node, int64_t index = -1,
                                bool need_save_sub_id = true) const;
  void Clear();
  inline void set_grad_is_running(bool grad_is_running) { grad_is_running_ = grad_is_running; }
  bool grad_is_running() const { return grad_is_running_; }
  inline void set_grad_first(bool grad_first) { grad_first_ = grad_first; }
  bool grad_first() const { return grad_first_; }

 private:
  void SetMultipleOutputToGraphInfoMap(const string &id, const AnfNodePtr &node) const;
  void SetNestedMultipleOutputToGraphInfoMap(const string &id, const AnfNodePtr &node,
                                             const std::vector<int64_t> &index_sequence) const;
  void SetUnpackOutputToGraphInfoMap(const std::string &id, const AnfNodePtr &node,
                                     const std::vector<int64_t> &index) const;
  bool is_high_order_top_cell_{false};
  bool jit_out_has_dict_{false};
  bool use_dynamic_shape_process_{false};

  // Top cell is running backward
  bool grad_is_running_{false};
  // if call grad not set_grad first, grad first is true
  bool grad_first_{false};

  size_t grad_order_{0};
  mutable size_t op_index_{0};

  // id with cell shape and type
  std::string cell_id_;

  // cell_id_ add grad_operation_ and grad_order_
  std::string ready_run_cell_id_;

  // cell inputs args id
  std::string input_args_id_;

  FuncGraphPtr fg_{nullptr};

  OrderedMap<FuncGraphPtr, GraphInfoPtr> graph_info_map_;

  InputArgsInfoPtr input_args_info_{nullptr};
};
using TopCellInfoPtr = std::shared_ptr<TopCellInfo>;
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_
