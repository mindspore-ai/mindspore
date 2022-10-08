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
#include "pipeline/pynative/grad/top_cell.h"
#include "utils/check_convert_utils.h"
#include "pybind11/pytypes.h"

namespace mindspore {
namespace pynative {
// Dynamic shape
class DynamicShape {
 public:
  DynamicShape() = default;
  ~DynamicShape() = default;
  void SetDynamicInput(const py::object &cell, const py::args &args);
  void SetFeedDynamicInputAbs(const py::object &cell, const py::args &args);
  ValuePtr GetSensValueForDynamicShapeOutput(const TopCellInfoPtr &top_cell, const ValuePtr &v,
                                             const AnfNodePtr &node) const;
  void UpdateValueBaseShape(const ValuePtr &v, const AbstractBasePtr &abs) const;
  void SetValueBaseShape(const ValuePtr &v, const AbstractBasePtr &abs) const;
  void SaveDynShapeAbsForMsFunction(const py::args &args, const py::object &out,
                                    const FuncGraphPtr &ms_func_graph) const;
  void UpdateSensValueForDynamicShapeOutput(const TopCellInfoPtr &top_cell, const ValuePtr &v) const;
  TopCellInfoPtr GetTopCellWithDynamicShape(const py::object &cell, const py::args &args, bool is_auto);
  void CheckPreviousTopCellCanBeDynamicShape(const py::object &cell, const py::args &args);
  py::object GetDynShape(const py::args &args) const;

 private:
  abstract::ShapePtr GetShapeFromAbstract(const abstract::AbstractBasePtr &abs) const;
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
  inline bool HasFeedDynamicInput() const { return !feed_dynamic_input_.empty(); }
  bool SetFeedTupleDynamicInputAbs(const abstract::AbstractBasePtr &abs, const py::object &arg, size_t i);

 private:
  mindspore::HashMap<std::string, std::vector<abstract::AbstractBasePtr>> feed_dynamic_input_;
};
using DynamicShapePtr = std::shared_ptr<DynamicShape>;
using DynamicShapeWeakPtr = std::weak_ptr<DynamicShape>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_DYNAMIC_SHAPE_H_
