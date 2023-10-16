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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_DYNAMIC_SHAPE_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_DYNAMIC_SHAPE_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "pipeline/pynative/grad/top_cell.h"

namespace mindspore {
namespace pynative {

struct NodeInfo {
  NodeInfo() = default;
  explicit NodeInfo(const TensorGradType &grad_type, size_t op_index = 0, ValuePtr value = nullptr)
      : grad_type(grad_type), op_index(op_index), value(std::move(value)) {}
  TensorGradType grad_type;
  size_t op_index{};
  ValuePtr value;
};

struct DynamicDetectNodeInfo {
  DynamicDetectNodeInfo(PrimitivePtr op_prim, abstract::AbstractBasePtrList input_abs,
                        abstract::AbstractBasePtr out_abs)
      : op_prim(std::move(op_prim)), input_abs(std::move(input_abs)), out_abs(std::move(out_abs)) {}
  PrimitivePtr op_prim{nullptr};
  abstract::AbstractBasePtrList input_abs{};
  abstract::AbstractBasePtr out_abs{nullptr};
  bool is_graph_node{false};
  std::vector<std::pair<std::string, NodeInfo>> inputs;
  std::string graph_phase;
};
using DynamicDetectNodeInfoPtr = std::shared_ptr<DynamicDetectNodeInfo>;
using CellIdWithDynamicNodesMap =
  mindspore::HashMap<std::string, mindspore::HashMap<std::string, std::vector<DynamicDetectNodeInfoPtr>>>;

class NodeDynamicDetect {
 public:
  NodeDynamicDetect() = default;
  ~NodeDynamicDetect() = default;
  void Clear() { cell_id_with_dynamic_detect_nodes_.clear(); }
  bool CheckNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                        const DynamicDetectNodeInfoPtr &node);
  bool IsNeedSaveDynamicDetectNodes(const TopCellInfoPtr &top_cell, bool use_dynamic_shape_process);

 private:
  bool IsNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs, const DynamicDetectNodeInfoPtr &node,
                     size_t node_idx);
  void SaveDynamicDetectNodeInfoInFirstTime(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                                            const DynamicDetectNodeInfoPtr &node, size_t node_idx);

  std::mutex async_mutex_;
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
  bool CheckNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                        const DynamicDetectNodeInfoPtr &node) {
    return node_dynamic_detect_ptr_->CheckNodeDynamic(top_cell, inputs, node);
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
    node_dynamic_detect_ptr_->Clear();
    top_cell_dynamic_detect_ptr_->Clear();
  }

 private:
  bool enable_unknown_shape_{false};
  TopCellUnknownShapeDetectPtr top_cell_dynamic_detect_ptr_{nullptr};
  NodeDynamicDetectPtr node_dynamic_detect_ptr_{nullptr};
};
using DynamicShapePtr = std::shared_ptr<DynamicShape>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_DYNAMIC_SHAPE_H_
