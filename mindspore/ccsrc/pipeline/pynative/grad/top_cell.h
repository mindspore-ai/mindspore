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
#include "pipeline/pynative/grad/auto_grad.h"
#include "frontend/operator/composite/composite.h"
#include "pipeline/jit/resource.h"
#include "pipeline/pynative/base.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
class GradExecutor;
using OpInfoWithTensorId = mindspore::HashMap<std::string, std::vector<std::string>>;
using TensorIdWithTensorObject = mindspore::HashMap<std::string, std::vector<tensor::TensorPtr>>;
using OpInfoWithMsFuncForwardTensors = mindspore::HashMap<std::string, std::vector<tensor::TensorPtr>>;
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
  TopCellInfo(bool is_high_order_top_cell, size_t grad_order, std::string obj_id_with_grad_order, std::string cellid,
              std::string already_run_cell_id, pipeline::ResourcePtr r, FuncGraphPtr fg)
      : is_high_order_top_cell_(is_high_order_top_cell),
        grad_order_(grad_order),
        obj_id_with_grad_order_(std::move(obj_id_with_grad_order)),
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
  void GetOpInfo(const FrontendOpRunInfoPtr &op_run_info);
  inline void ClearCellHookOp() { cell_backward_hook_op_.clear(); }
  inline bool forward_already_run() const { return forward_already_run_; }
  inline void set_forward_already_run(bool set_forward_already_run) { forward_already_run_ = set_forward_already_run; }
  inline bool need_compile_graph() const { return need_compile_graph_; }
  inline void set_need_compile_graph(bool need_compile_graph) { need_compile_graph_ = need_compile_graph; }
  inline bool vm_compile() const { return vm_compile_; }
  inline void set_force_top_cell_compile(bool force_top_cell_compile) {
    force_top_cell_compile_ = force_top_cell_compile;
  }
  inline bool force_top_cell_compile() const { return force_top_cell_compile_; }
  inline bool is_high_order_top_cell() const { return is_high_order_top_cell_; }
  inline void set_need_do_final_opt(bool need_do_final_opt) { need_do_final_opt_ = need_do_final_opt; }
  inline bool need_do_final_opt() const { return need_do_final_opt_; }
  inline void set_need_save_dynamic_detect_nodes(bool is_need_save_dynamic_detect_nodes) {
    is_need_save_dynamic_detect_nodes_ = is_need_save_dynamic_detect_nodes;
  }
  inline bool is_need_save_dynamic_detect_nodes() const { return is_need_save_dynamic_detect_nodes_; }
  inline pipeline::ResourcePtr resource() const { return resource_; }
  inline FuncGraphPtr fg() const {
    MS_EXCEPTION_IF_NULL(fg_);
    return fg_;
  }
  inline void set_fg(const FuncGraphPtr &fg) { fg_ = fg; }
  inline const std::string &cell_id() const { return cell_id_; }
  inline const std::string &obj_id_with_grad_order() const { return obj_id_with_grad_order_; }
  inline const std::string &already_run_cell_id() const { return already_run_cell_id_; }
  inline void set_input_args_id(const std::string &input_args_id) { input_args_id_ = input_args_id; }
  inline const std::string &input_args_id() const { return input_args_id_; }
  const std::string &grad_operation() const { return grad_operation_; }
  void set_grad_operation(const std::string &grad_operation) { grad_operation_ = grad_operation; }
  inline void CheckSubCellHookChanged() { sub_cell_hook_changed_.clear(); }
  inline void SetGraphInfoMap(const FuncGraphPtr &fg, const GraphInfoPtr &graph_info) {
    graph_info_map_[fg] = graph_info;
  }
  inline const OrderedMap<FuncGraphPtr, GraphInfoPtr> &graph_info_map() const { return graph_info_map_; }
  inline autograd::AutoGradCellImplPtr auto_grad_cell_ptr() const {
    MS_EXCEPTION_IF_NULL(auto_grad_cell_ptr_);
    return auto_grad_cell_ptr_;
  }
  void set_auto_grad_cell_ptr(const autograd::AutoGradCellImplPtr &auto_grad_cell_ptr) {
    auto_grad_cell_ptr_ = auto_grad_cell_ptr;
  }
  inline const OpInfoWithTensorId &op_info_with_tensor_id() const { return op_info_with_tensor_id_; }
  void set_opinfo_with_tensor_id(const std::string &op_info, const std::vector<tensor::TensorPtr> &op_out_tensors);
  inline const TensorIdWithTensorObject &tensor_id_with_tensor_object() const { return tensor_id_with_tensor_object_; }
  inline void set_tensor_id_with_tensor_object(const std::string &id, const tensor::TensorPtr &tensor) {
    (void)tensor_id_with_tensor_object_[id].emplace_back(tensor);
  }
  inline const OpInfoWithMsFuncForwardTensors &op_info_with_ms_func_forward_tensors() const {
    return op_info_with_ms_func_forward_tensors_;
  }
  inline void set_op_info_with_ms_func_forward_tensors(const std::string &op_info,
                                                       const std::vector<tensor::TensorPtr> &forward_tensors) {
    op_info_with_ms_func_forward_tensors_[op_info] = forward_tensors;
  }
  inline size_t op_index() const { return op_index_; }
  inline void IncreaseOpIndex() { ++op_index_; }

  inline void set_cnode_hash_with_op_index(const size_t &node_hash, const size_t &op_index) {
    cnode_hash_with_op_index_[node_hash] = op_index;
  }
  inline size_t get_op_index_by_cnode_hash(const size_t node_hash, const size_t node_idx) const {
    const auto iter = cnode_hash_with_op_index_.find(node_hash);
    if (iter == cnode_hash_with_op_index_.end()) {
      MS_LOG(DEBUG) << "hash:" << node_hash << " is not found in cnode_hash_with_op_index_";
      return node_idx;
    }
    return iter->second;
  }

  void DeleteParamNodeInfo(const FuncGraphPtr &g, const std::string &id);
  void SetParamNodeMapInGraphInfoMap(const std::string &id, const ParameterPtr &param, bool is_weight = false) const;
  void SetNodeMapInGraphInfoMap(const std::string &id, const AnfNodePtr &node, int64_t index = -1,
                                bool need_save_sub_id = true) const;
  void UpdateTopCellInfo(bool forward_already_run, bool need_compile_graph, bool vm_compile);
  void ClearDeviceMemory() const;
  void Clear();

  inline bool use_dynamic_shape_process() { return use_dynamic_shape_process_; }
  inline void set_use_dynamic_shape_process(bool use_dynamic_shape_process) {
    use_dynamic_shape_process_ = use_dynamic_shape_process;
  }

 private:
  void SetMultipleOutputToGraphInfoMap(const string &id, const AnfNodePtr &node) const;
  void SetNestedMultipleOutputToGraphInfoMap(const string &id, const AnfNodePtr &node,
                                             const std::vector<int64_t> &index_sequence) const;
  void SetUnpackOutputToGraphInfoMap(const std::string &id, const AnfNodePtr &node,
                                     const std::vector<int64_t> &index) const;

  bool hook_changed_{false};
  bool is_init_kpynative_{false};
  bool forward_already_run_{false};
  bool need_compile_graph_{false};
  bool vm_compile_{false};
  bool force_top_cell_compile_{false};
  bool is_high_order_top_cell_{false};
  bool need_do_final_opt_{false};
  bool is_need_save_dynamic_detect_nodes_{false};
  size_t grad_order_{0};
  size_t op_index_{0};
  std::string obj_id_with_grad_order_;
  std::string cell_id_;
  std::string already_run_cell_id_;
  std::string input_args_id_;
  std::string grad_operation_;
  pipeline::ResourcePtr resource_{nullptr};
  FuncGraphPtr fg_{nullptr};
  autograd::AutoGradCellImplPtr auto_grad_cell_ptr_{nullptr};
  OrderedMap<FuncGraphPtr, GraphInfoPtr> graph_info_map_;
  // Record `register hook` or `remove hook` function has been called by sub cell
  // The record range between the begin and end of top cell.
  mindspore::HashSet<std::string> sub_cell_hook_changed_;
  // Record backward hook ops for each cell object.
  // Each cell object has two backward hook ops.
  CellIdWithBackwardHookOp cell_backward_hook_op_;
  OpInfoWithTensorId op_info_with_tensor_id_;
  TensorIdWithTensorObject tensor_id_with_tensor_object_;
  OpInfoWithMsFuncForwardTensors op_info_with_ms_func_forward_tensors_;
  mindspore::HashMap<size_t, size_t> cnode_hash_with_op_index_;
  bool use_dynamic_shape_process_{false};
};
using TopCellInfoPtr = std::shared_ptr<TopCellInfo>;
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_
