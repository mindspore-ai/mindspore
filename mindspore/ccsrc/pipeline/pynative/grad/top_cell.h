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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_

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
#include "pybind_api/ir/base_ref_py.h"
#include "ir/anf.h"
#include "frontend/optimizer/ad/kpynative.h"
#include "frontend/operator/composite/composite.h"
#include "pipeline/jit/resource.h"
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_cache.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
using OpInfoWithTensorId = mindspore::HashMap<std::string, std::vector<std::string>>;
using TensorIdWithTensorObject = mindspore::HashMap<std::string, std::vector<tensor::TensorPtr>>;
using OpInfoWithMsFuncForwardTensors = mindspore::HashMap<std::string, std::vector<tensor::TensorPtr>>;
using CellIdWithBackwardHookOp = mindspore::HashMap<std::string, std::vector<AnfNodePtr>>;

struct GraphInfo {
  GraphInfo() = default;
  ~GraphInfo() = default;
  std::string cell_id;
  AnfNodePtr output;
  OrderedMap<std::string, ParameterPtr> params;  // hold input parameters and cell weights
  mindspore::HashMap<std::string, std::pair<AnfNodePtr, std::vector<int64_t>>> node_map;
  explicit GraphInfo(std::string id) : cell_id(std::move((id))) {}
};
using GraphInfoPtr = std::shared_ptr<GraphInfo>;

struct CellSelfInfo {
  CellSelfInfo() = default;
  ~CellSelfInfo() = default;
  CellSelfInfo(std::string cell_self_id, std::vector<std::string> args_id, std::vector<abstract::ShapePtr> args_shape,
               std::vector<TypePtr> args_type)
      : cell_self_id(std::move(cell_self_id)),
        args_id(std::move(args_id)),
        args_shape(std::move(args_shape)),
        args_type(std::move(args_type)) {}

  std::string cell_self_id;
  std::vector<std::string> args_id;
  std::vector<abstract::ShapePtr> args_shape;
  std::vector<TypePtr> args_type;
};
using CellSelfInfoPtr = std::shared_ptr<CellSelfInfo>;

class TopCellInfo {
 public:
  ~TopCellInfo() = default;
  TopCellInfo(bool topest, size_t grad_order, std::string cellid, std::string already_run_cell_id,
              pipeline::ResourcePtr r, FuncGraphPtr fg, FuncGraphPtr df)
      : is_topest_(topest),
        grad_order_(grad_order),
        cell_id_(std::move(cellid)),
        already_run_cell_id_(std::move(already_run_cell_id)),
        resource_(std::move(r)),
        fg_(std::move(fg)),
        df_builder_(std::move(df)) {}

  TopCellInfo(const TopCellInfo &top_cell, pipeline::ResourcePtr r, FuncGraphPtr fg, FuncGraphPtr df)
      : is_topest_(top_cell.is_topest_),
        grad_order_(top_cell.grad_order_),
        cell_id_(top_cell.cell_id_),
        already_run_cell_id_(top_cell.already_run_cell_id_),
        cell_self_info_(top_cell.cell_self_info_),
        resource_(std::move(r)),
        fg_(std::move(fg)),
        df_builder_(std::move(df)) {}

  bool is_init_kpynative() const { return is_init_kpynative_; }
  void set_init_kpynative(bool init) { is_init_kpynative_ = init; }
  bool is_topest() const { return is_topest_; }
  size_t grad_order() const { return grad_order_; }
  bool is_dynamic_structure() const { return is_dynamic_structure_; }
  void set_dynamic_structure(bool is_dynamic_structure) { is_dynamic_structure_ = is_dynamic_structure; }
  bool is_real_dynamic_structure() const { return is_real_dynamic_structure_; }
  void set_is_real_dynamic_structure(bool is_real_dynamic_structure) {
    is_real_dynamic_structure_ = is_real_dynamic_structure;
  }
  bool dynamic_shape() const { return dynamic_shape_; }
  void set_dynamic_shape(bool dynamic_shape) { dynamic_shape_ = dynamic_shape; }
  bool hook_changed() const { return hook_changed_; }
  void set_hook_changed(bool hook_changed) { hook_changed_ = hook_changed; }
  void set_sub_cell_hook_changed(const std::string &sub_cell) { (void)sub_cell_hook_changed_.emplace(sub_cell); }
  const CellIdWithBackwardHookOp &cell_backward_hook_op() const { return cell_backward_hook_op_; }
  void RecordCellBackwardHookOp(const std::string &cell_order, const AnfNodePtr &hook_op);
  void ClearCellHookOp() { cell_backward_hook_op_.clear(); }
  bool vm_compiled() const { return vm_compiled_; }
  void set_vm_compiled(bool vm_compiled) { vm_compiled_ = vm_compiled; }
  bool ms_function_flag() const { return ms_function_flag_; }
  void set_ms_function_flag(bool ms_function_flag) { ms_function_flag_ = ms_function_flag; }
  bool need_compile_graph() const { return need_compile_graph_; }
  void set_need_compile_graph(bool need_compile_graph) { need_compile_graph_ = need_compile_graph; }
  bool forward_already_run() const { return forward_already_run_; }
  void set_forward_already_run(bool set_forward_already_run) { forward_already_run_ = set_forward_already_run; }
  pipeline::ResourcePtr resource() const { return resource_; }
  FuncGraphPtr df_builder() const { return df_builder_; }
  inline FuncGraphPtr fg() const {
    MS_EXCEPTION_IF_NULL(fg_);
    return fg_;
  }
  void set_fg(const FuncGraphPtr &fg) { fg_ = fg; }
  const std::string &cell_id() const { return cell_id_; }
  void set_cell_id(const std::string &cell_id) { cell_id_ = cell_id; }
  const std::string &already_run_cell_id() const { return already_run_cell_id_; }
  void set_input_args_id(const std::string &input_args_id) { input_args_id_ = input_args_id; }
  const std::string &grad_operation() const { return grad_operation_; }
  void set_grad_operation(const std::string &grad_operation) { grad_operation_ = grad_operation; }
  const abstract::AbstractBasePtr &last_output_abs() const { return last_output_abs_; }
  void set_last_output_abs(const abstract::AbstractBasePtr &last_output_abs) { last_output_abs_ = last_output_abs; }
  CellSelfInfoPtr cell_self_info() const { return cell_self_info_; }
  void SetCellSelfInfoForTopCell(const py::object &cell, const py::args &args);
  void EraseFromSubCellList(const std::string &cell_id) { (void)sub_cell_list_.erase(cell_id); }
  void SetSubCellList(const std::string &cell_id) { (void)sub_cell_list_.emplace(cell_id); }
  const mindspore::HashSet<std::string> &sub_cell_list() const { return sub_cell_list_; }
  bool IsSubCell(const std::string &cell_id) const;
  void CheckSubCellHookChanged();
  void SetGraphInfoMap(const FuncGraphPtr &fg, const GraphInfoPtr &graph_info) { graph_info_map_[fg] = graph_info; }
  const OrderedMap<FuncGraphPtr, GraphInfoPtr> &graph_info_map() const { return graph_info_map_; }
  const OpInfoWithTensorId &op_info_with_tensor_id() const { return op_info_with_tensor_id_; }
  void SetTensorIdWithTensorObject(const std::string &id, const tensor::TensorPtr &tensor) {
    (void)tensor_id_with_tensor_object_[id].emplace_back(tensor);
  }
  const TensorIdWithTensorObject &tensor_id_with_tensor_object() const { return tensor_id_with_tensor_object_; }
  ad::KPynativeCellPtr k_pynative_cell_ptr() const { return k_pynative_cell_ptr_; }
  void set_k_pynative_cell_ptr(const ad::KPynativeCellPtr &k_pynative_cell_ptr) {
    k_pynative_cell_ptr_ = k_pynative_cell_ptr;
  }
  const OpInfoWithMsFuncForwardTensors &op_info_with_ms_func_forward_tensors() const {
    return op_info_with_ms_func_forward_tensors_;
  }
  void set_op_info_with_ms_func_forward_tensors(const std::string &op_info,
                                                const std::vector<tensor::TensorPtr> &forward_tensors) {
    op_info_with_ms_func_forward_tensors_[op_info] = forward_tensors;
  }
  const std::string &input_args_id() const { return input_args_id_; }
  const std::string &all_op_info() const { return all_op_info_; }
  void set_all_op_info(const std::string &all_op_info) { all_op_info_ = all_op_info; }
  void ResetTopCellInfo(const py::args &args);
  void SaveOpInfo(const std::string &op_info, const std::vector<tensor::TensorPtr> &op_out_tensors);
  void RecordGradOpInfo(const FrontendOpRunInfoPtr &op_run_info);
  void SetParamNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const ParameterPtr &param) const;
  void SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                int64_t index = -1) const;
  void SetNodeMapInGraphInfoMap(const FuncGraphPtr &g, const std::string &id, const AnfNodePtr &node,
                                const std::vector<int64_t> &index) const;
  void SetTupleArgsToGraphInfoMap(const FuncGraphPtr &g, const ValuePtr &v, const AnfNodePtr &node,
                                  bool is_param = false);
  void ChangeTopCellInfo(size_t args_size);
  void ClearDeviceMemory() const;
  void Clear();

 private:
  void IncreaseOpNum() { ++op_num_; }
  void AppendAllOpInfo(const std::string &op_info) { all_op_info_ += "-" + op_info; }
  void SetOpInfoWithTensorId(const std::string &op_info, const std::string &tensor_id) {
    (void)op_info_with_tensor_id_[op_info].emplace_back(tensor_id);
  }
  void SetTupleItemArgsToGraphInfoMap(const FuncGraphPtr &g, const ValuePtr &v, const AnfNodePtr &node,
                                      const std::vector<int64_t> &index_sequence, bool is_param);

 private:
  bool is_topest_{false};
  bool is_dynamic_structure_{false};
  // Set this flag to ture when all_op_info of top_cell is changed.
  bool is_real_dynamic_structure_{false};
  bool dynamic_shape_{false};
  bool vm_compiled_{false};
  bool hook_changed_{false};
  bool ms_function_flag_{false};
  bool is_init_kpynative_{false};
  bool forward_already_run_{false};
  bool need_compile_graph_{false};
  size_t op_num_{0};
  size_t grad_order_{0};
  std::string cell_id_;
  std::string already_run_cell_id_;
  std::string input_args_id_;
  std::string all_op_info_;
  std::string grad_operation_;
  CellSelfInfoPtr cell_self_info_{nullptr};
  pipeline::ResourcePtr resource_{nullptr};
  FuncGraphPtr fg_{nullptr};
  FuncGraphPtr df_builder_{nullptr};
  ad::KPynativeCellPtr k_pynative_cell_ptr_{nullptr};
  abstract::AbstractBasePtr last_output_abs_{nullptr};
  OrderedMap<FuncGraphPtr, GraphInfoPtr> graph_info_map_;
  mindspore::HashSet<std::string> sub_cell_list_;
  // Record `register hook` or `remove hook` function has been called by sub cell
  // The record range between the begin and end of top cell.
  mindspore::HashSet<std::string> sub_cell_hook_changed_;
  // Record backward hook ops for each cell object.
  // Each cell object has two backward hook ops.
  CellIdWithBackwardHookOp cell_backward_hook_op_;
  OpInfoWithTensorId op_info_with_tensor_id_;
  TensorIdWithTensorObject tensor_id_with_tensor_object_;
  OpInfoWithMsFuncForwardTensors op_info_with_ms_func_forward_tensors_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_TOP_CELL_H_
