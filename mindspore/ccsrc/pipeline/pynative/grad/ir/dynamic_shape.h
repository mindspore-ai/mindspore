/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_DYNAMIC_SHAPE_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_DYNAMIC_SHAPE_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "pipeline/pynative/grad/top_cell.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace pynative {

struct NodeInfo {
  // Is parameter or input or op's output
  InputType grad_type{InputType::kConstant};
  // Just op output tensor has op_index
  size_t op_index{0};
  // For scalar compare
  ValuePtr value{nullptr};
  // For Input is tuple or list
  std::vector<NodeInfo> seq_node;
};

struct AbsCompareInfo {
  AbsCompareInfo() = default;
  AbsCompareInfo(abstract::AbstractBasePtrList input_abs, abstract::AbstractBasePtr out_abs)
      : input_abs(std::move(input_abs)), out_abs(std::move(out_abs)) {}
  abstract::AbstractBasePtrList input_abs{};
  abstract::AbstractBasePtr out_abs{nullptr};
  std::vector<NodeInfo> inputs;
};

struct ValueCompareInfo {
  // ValueSimpleInfo
  ValueSimpleInfo input_value_simple_info;
  std::vector<NodeInfo> inputs;
};

struct DynamicDetectNodeInfo {
  explicit DynamicDetectNodeInfo(PrimitivePtr op_prim, std::string graph_phase, size_t op_index,
                                 bool is_value_compare = true)
      : op_prim(std::move(op_prim)), graph_phase(graph_phase), op_index(op_index), is_value_compare(is_value_compare) {}
  DynamicDetectNodeInfo(PrimitivePtr op_prim, std::string graph_phase, size_t op_index,
                        abstract::AbstractBasePtrList input_abs, abstract::AbstractBasePtr out_abs)
      : op_prim(std::move(op_prim)),
        graph_phase(graph_phase),
        op_index(op_index),
        abs_compare_info(std::move(input_abs), std::move(out_abs)) {}

  PrimitivePtr op_prim{nullptr};
  std::string graph_phase;
  // op or jit execute index
  size_t op_index{0};
  bool is_value_compare{false};
  AbsCompareInfo abs_compare_info;
  ValueCompareInfo value_compare_info;
};
using DynamicDetectNodeInfoPtr = std::shared_ptr<DynamicDetectNodeInfo>;
using CellIdWithDynamicNodesMap =
  mindspore::HashMap<std::string, mindspore::HashMap<std::string, std::vector<DynamicDetectNodeInfoPtr>>>;

class NodeDynamicDetect {
 public:
  NodeDynamicDetect() = default;
  ~NodeDynamicDetect() = default;
  void Clear() { cell_id_with_dynamic_detect_nodes_.clear(); }
  void CheckNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                        const DynamicDetectNodeInfoPtr &node);
  bool IsNeedSaveDynamicDetectNodes(const TopCellInfoPtr &top_cell, bool use_dynamic_shape_process);

 private:
  bool IsNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs, const DynamicDetectNodeInfoPtr &node);
  void SaveDynamicDetectNodeInfoInFirstTime(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                                            const DynamicDetectNodeInfoPtr &node);

  CellIdWithDynamicNodesMap cell_id_with_dynamic_detect_nodes_;
};
using NodeDynamicDetectPtr = std::shared_ptr<NodeDynamicDetect>;

class TopCellUnknownShapeDetect {
 public:
  TopCellUnknownShapeDetect() = default;
  ~TopCellUnknownShapeDetect() = default;

  void SetDynamicInput(const py::object &obj, const py::args &args);
  void TryChangeTopCellToUnknownShape(const std::string &obj_id, const abstract::BaseShapePtrList &arg_base_shape_vec,
                                      bool is_auto_detect);
  void UpdateArgsAbsToUnknownShapeAbs(const py::object &obj, const py::args &args);

  void Clear() {
    obj_with_by_inputs_.clear();
    obj_id_args_info_by_set_inputs_.clear();
  }

 private:
  // pre top cell is already unknown shape, args shape is current input, check whether the requirements are met through
  // shape comparison.
  bool CanFindMatchedUnknownShapeTopCell(const TopCellInfoPtr &pre_top_cell,
                                         const abstract::BaseShapePtrList &cur_args_shape);
  bool SetTopCellUnknownShape(const TopCellInfoPtr &cur_top_cell, const TopCellInfoPtr &pre_top_cell,
                              const abstract::BaseShapePtrList &args_shape);
  void ChangeTopCellToUnknownShape(const TopCellInfoPtr &top_cell,
                                   const abstract::BaseShapePtrList &args_unknown_shape);
  void UpdateUnknownShapeAbsCache(const std::vector<string> &input_arg_id_vec,
                                  const std::vector<ValuePtr> &input_arg_value_vec,
                                  const std::vector<abstract::BaseShapePtr> &args_base_shape);

  // Like TrainOneStep, it is a cell and run first, top cell create first, but set inputs set in main cell
  // and run later, so need change top cell to unknown shape too.
  void UpdatePossibleTopCellToUnknownShape(const TopCellInfoPtr &cur_top_cell,
                                           const std::vector<string> &cur_arg_id_vec,
                                           const abstract::BaseShapePtrList &cur_args_shape);

  // Obj id(cell or function) with set inputs
  mindspore::HashSet<std::string> obj_with_by_inputs_;
  // Obj id with its args base shape
  mindspore::HashMap<std::string, abstract::BaseShapePtrList> obj_id_args_info_by_set_inputs_;
};
using TopCellUnknownShapeDetectPtr = std::shared_ptr<TopCellUnknownShapeDetect>;

class DynamicShape {
 public:
  DynamicShape()
      : top_cell_dynamic_detect_ptr_(std::make_shared<TopCellUnknownShapeDetect>()),
        node_dynamic_detect_ptr_(std::make_shared<NodeDynamicDetect>()) {}
  ~DynamicShape() = default;

  void set_enable_unknown_shape(bool enable_unknown_shape) { enable_unknown_shape_ = enable_unknown_shape; }
  inline bool enable_unknown_shape() const { return enable_unknown_shape_; }
  py::object GetDynamicInput(const py::object &actual_input);
  void SaveUnknownShapeAbsFromJit(const ValuePtr &v, const AbstractBasePtr &abs, size_t index);

  // For node dynamic struct check
  void CheckNodeDynamic(const TopCellInfoPtr &top_cell, const OpGradInfoPtr &op_grad_info,
                        const std::string &graph_phase = "") {
    MS_EXCEPTION_IF_NULL(top_cell);
    MS_EXCEPTION_IF_NULL(op_grad_info);
    if (top_cell->use_dynamic_shape_process()) {
      return;
    }
    DynamicDetectNodeInfoPtr node_info;
    if (op_grad_info->output_value_simple_info != nullptr) {
      node_info = std::make_shared<DynamicDetectNodeInfo>(op_grad_info->op_prim, graph_phase, op_grad_info->op_index);
    } else {
      node_info = std::make_shared<DynamicDetectNodeInfo>(op_grad_info->op_prim, graph_phase, op_grad_info->op_index,
                                                          op_grad_info->input_abs, op_grad_info->out_abs);
    }
    top_cell->CheckBpropCutNode(op_grad_info->op_prim);
    node_dynamic_detect_ptr_->CheckNodeDynamic(top_cell, op_grad_info->input_value, node_info);
  }

  bool IsNeedSaveDynamicDetectNodes(const TopCellInfoPtr &top_cell, bool use_dynamic_shape_process) {
    return node_dynamic_detect_ptr_->IsNeedSaveDynamicDetectNodes(top_cell, use_dynamic_shape_process);
  }

  // For top cell unknown shape
  void SetDynamicInput(const py::object &obj, const py::args &args) {
    top_cell_dynamic_detect_ptr_->SetDynamicInput(obj, args);
  }

  void TryChangeTopCellToUnknownShape(const std::string &obj_id, const abstract::BaseShapePtrList &arg_base_shape_vec,
                                      bool is_auto_detect) {
    top_cell_dynamic_detect_ptr_->TryChangeTopCellToUnknownShape(obj_id, arg_base_shape_vec, is_auto_detect);
  }

  void UpdateArgsAbsToUnknownShapeAbs(const py::object &obj, const py::args &args) {
    top_cell_dynamic_detect_ptr_->UpdateArgsAbsToUnknownShapeAbs(obj, args);
  }

  void Clear() {
    top_cell_dynamic_detect_ptr_->Clear();
    node_dynamic_detect_ptr_->Clear();
  }

 private:
  bool enable_unknown_shape_{false};
  TopCellUnknownShapeDetectPtr top_cell_dynamic_detect_ptr_{nullptr};
  NodeDynamicDetectPtr node_dynamic_detect_ptr_{nullptr};
};
using DynamicShapePtr = std::shared_ptr<DynamicShape>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_DYNAMIC_SHAPE_H_
