/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "include/common/utils/convert_utils.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "pybind_api/ir/base_ref_py.h"
#include "ir/anf.h"
#include "frontend/optimizer/ad/auto_grad.h"
#include "frontend/operator/composite/composite.h"
#include "pipeline/jit/resource.h"
#include "pipeline/pynative/base.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
class GradExecutor;
using CellIdWithBackwardHookOp = mindspore::HashMap<std::string, std::vector<AnfNodePtr>>;

struct GraphInfo {
  OrderedMap<std::string, ParameterPtr> input_params;   // Hold input parameters
  OrderedMap<std::string, ParameterPtr> weight_params;  // Hold weights parameters
  // Hold op op output or combination of output
  mindspore::HashMap<std::string, std::pair<AnfNodePtr, std::vector<int64_t>>> node_map;
};
using GraphInfoPtr = std::shared_ptr<GraphInfo>;

class TopCellInfo {
 public:
  ~TopCellInfo() = default;
  TopCellInfo(bool is_high_order_top_cell, size_t grad_order, std::string cellid, std::string already_run_cell_id,
              pipeline::ResourcePtr r, FuncGraphPtr fg)
      : is_high_order_top_cell_(is_high_order_top_cell),
        grad_order_(grad_order),
        cell_id_(std::move(cellid)),
        already_run_cell_id_(std::move(already_run_cell_id)),
        resource_(std::move(r)),
        fg_(std::move(fg)) {}

  inline bool is_init_kpynative() const { return is_init_kpynative_; }
  inline void set_init_kpynative(bool init) { is_init_kpynative_ = init; }
  inline size_t grad_order() const { return grad_order_; }
  inline void set_hook_changed(bool hook_changed) { hook_changed_ = hook_changed; }
  inline void set_sub_cell_hook_changed(const std::string &sub_cell) { (void)sub_cell_hook_changed_.emplace(sub_cell); }
  inline const CellIdWithBackwardHookOp &cell_backward_hook_op() const { return cell_backward_hook_op_; }
  void RecordCellBackwardHookOp(const std::string &cell_order, const AnfNodePtr &hook_op);
  inline void ClearCellHookOp() { cell_backward_hook_op_.clear(); }
  inline bool forward_already_run() const { return forward_already_run_; }
  inline void set_forward_already_run(bool set_forward_already_run) { forward_already_run_ = set_forward_already_run; }
  inline bool is_high_order_top_cell() const { return is_high_order_top_cell_; }
  inline void set_need_do_final_opt(bool need_do_final_opt) { need_do_final_opt_ = need_do_final_opt; }
  inline bool need_do_final_opt() const { return need_do_final_opt_; }
  inline pipeline::ResourcePtr resource() const { return resource_; }
  inline FuncGraphPtr fg() const {
    MS_EXCEPTION_IF_NULL(fg_);
    return fg_;
  }
  inline void set_fg(const FuncGraphPtr &fg) { fg_ = fg; }
  inline const std::string &cell_id() const { return cell_id_; }
  inline const std::string &already_run_cell_id() const { return already_run_cell_id_; }
  inline void set_input_args_id(const std::string &input_args_id) { input_args_id_ = input_args_id; }
  inline const std::string &input_args_id() const { return input_args_id_; }
  inline void CheckSubCellHookChanged() { sub_cell_hook_changed_.clear(); }
  inline void SetGraphInfoMap(const FuncGraphPtr &fg, const GraphInfoPtr &graph_info) {
    graph_info_map_[fg] = graph_info;
  }
  inline const OrderedMap<FuncGraphPtr, GraphInfoPtr> &graph_info_map() const { return graph_info_map_; }
  inline ad::AutoGradCellImplPtr auto_grad_cell_ptr() const { return auto_grad_cell_ptr_; }
  void set_auto_grad_cell_ptr(const ad::AutoGradCellImplPtr &auto_grad_cell_ptr) {
    auto_grad_cell_ptr_ = auto_grad_cell_ptr;
  }
  void DeleteParamNodeInfo(const FuncGraphPtr &g, const std::string &id);
  void SetParamNodeMapInGraphInfoMap(const std::string &id, const ParameterPtr &param, bool is_weight = false) const;
  void SetNodeMapInGraphInfoMap(const std::string &id, const AnfNodePtr &node, int64_t index = -1,
                                bool save_flag = true) const;
  void ClearDeviceMemory() const;

 private:
  void SetMultipleOutputToGraphInfoMap(const string &id, const AnfNodePtr &node) const;
  void SetNestedMultipleOutputToGraphInfoMap(const string &id, const AnfNodePtr &node,
                                             const std::vector<int64_t> &index_sequence) const;
  void SetUnpackOutputToGraphInfoMap(const std::string &id, const AnfNodePtr &node,
                                     const std::vector<int64_t> &index) const;

  bool hook_changed_{false};
  bool is_init_kpynative_{false};
  bool forward_already_run_{false};
  bool is_high_order_top_cell_{false};
  bool need_do_final_opt_{false};
  size_t grad_order_{0};
  std::string cell_id_;
  std::string already_run_cell_id_;
  std::string input_args_id_;
  std::string grad_operation_;
  pipeline::ResourcePtr resource_{nullptr};
  FuncGraphPtr fg_{nullptr};
  ad::AutoGradCellImplPtr auto_grad_cell_ptr_{nullptr};
  OrderedMap<FuncGraphPtr, GraphInfoPtr> graph_info_map_;
  // Record `register hook` or `remove hook` function has been called by sub cell
  // The record range between the begin and end of top cell.
  mindspore::HashSet<std::string> sub_cell_hook_changed_;
  // Record backward hook ops for each cell object.
  // Each cell object has two backward hook ops.
  CellIdWithBackwardHookOp cell_backward_hook_op_;
};
using TopCellInfoPtr = std::shared_ptr<TopCellInfo>;
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_
