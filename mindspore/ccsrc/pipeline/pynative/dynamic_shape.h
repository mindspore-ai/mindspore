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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_DYNAMIC_SHAPE_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_DYNAMIC_SHAPE_H_

#include <memory>
#include <string>
#include <vector>
#include <set>
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/grad/top_cell.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace pynative {
class TopCellInfo;
using TopCellInfoPtr = std::shared_ptr<TopCellInfo>;
using TensorIdWithTensorObject = mindspore::HashMap<std::string, std::vector<tensor::TensorPtr>>;

// Dynamic shape
class DynamicShape {
 public:
  DynamicShape() = default;
  ~DynamicShape() = default;
  void SetDynamicInput(const py::object &cell, const py::args &args);
  void SetFeedDynamicInputAbs(const py::object &cell, const py::args &args);
  py::object GetDynamicInput(const py::object &actual_input) const;
  ValuePtr GetSensValueForDynamicShapeOutput(const TopCellInfoPtr &top_cell, const ValuePtr &v,
                                             const AnfNodePtr &node) const;
  void UpdateValueToDynamicShape(const ValuePtr &value) const;
  void UpdateInputTensorToDynamicShape(const FrontendOpRunInfoPtr &op_run_info);
  void SaveDynShapeAbsForMsFunction(const py::args &args, const py::object &out, const FuncGraphPtr &ms_func_graph);
  void SaveOutputDynamicShape(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v);
  void UpdateSensValueForDynamicShapeOutput(const TopCellInfoPtr &top_cell, const ValuePtr &v) const;
  TopCellInfoPtr GetTopCellWithDynamicShape(const py::object &cell, const py::args &args, bool is_auto);
  void CheckPreviousTopCellCanBeDynamicShape(const py::object &cell, const py::args &args);
  const OrderedMap<std::string, abstract::AbstractBasePtr> &id_with_dynamic_abs() const { return id_with_dynamic_abs_; }
  void SetIdWithDynamicAbs(const std::string &id, const abstract::AbstractBasePtr &abs) {
    id_with_dynamic_abs_[id] = abs;
  }
  void reset() { id_with_dynamic_abs_.clear(); }
  py::object GetDynShape(const py::args &args) const;

 private:
  ShapeVector GetTensorShape(const ValuePtr &v) const;
  abstract::ShapePtr GetShapeFromAbstract(const abstract::AbstractBasePtr &abs) const;
  void SaveIdWithDynamicAbstract(const ValuePtr &v, const AbstractBasePtr &abs);
  ValuePtr SetSensValue(const ValuePtr &value, const TopCellInfoPtr &top_cell) const;

  TopCellInfoPtr ChangeTopCellToDynamicShapeBySetInputs(const TopCellInfoPtr &top_cell,
                                                        const std::vector<ShapeVector> &new_args_shape,
                                                        const py::object &cell);
  TopCellInfoPtr ChangeTopCellToDynamicShapeByAuto(const TopCellInfoPtr &top_cell,
                                                   const std::vector<ShapeVector> &new_args_shape,
                                                   const py::object &cell, const py::args &args);
  void UpdateTopCellId(const py::args &args) const;
  void FindMatchTopCell(const TopCellInfoPtr &top_cell, const py::args &args,
                        std::vector<ShapeVector> *new_args_shape) const;
  bool HasFeedDynamicInput() const { return !feed_dynamic_input_.empty(); }

 private:
  OrderedMap<std::string, abstract::AbstractBasePtr> id_with_dynamic_abs_;
  mindspore::HashMap<std::string, std::vector<abstract::AbstractBasePtr>> feed_dynamic_input_;
};
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_DYNAMIC_SHAPE_H_
