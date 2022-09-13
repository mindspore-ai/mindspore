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
#include "pipeline/pynative/dynamic_shape.h"
#include <utility>
#include <algorithm>
#include "pipeline/pynative/pynative_utils.h"
#include "include/common/utils/convert_utils_py.h"

namespace mindspore {
namespace pynative {
const char kSensInfo[] = "SensInfo";

ShapeVector DynamicShape::GetTensorShape(const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    return v->cast<tensor::TensorPtr>()->shape();
  }
  return {};
}

abstract::ShapePtr DynamicShape::GetShapeFromAbstract(const abstract::AbstractBasePtr &abs) const {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "Get tuple or list abs";
  }
  const auto &base_shape = abs->BuildShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  const auto &shape = base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  return shape;
}

void DynamicShape::SaveIdWithDynamicAbstract(const ValuePtr &v, const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(abs);
  if (v->isa<ValueSequence>() && abs->isa<abstract::AbstractTuple>()) {
    const auto &obj_tuple = v->cast<ValueSequencePtr>();
    const auto &abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
    if (obj_tuple->size() != abs_tuple->size()) {
      MS_LOG(EXCEPTION) << "Obj tuple size " << obj_tuple->size() << ", but abstract tuple size " << abs_tuple->size();
    }
    for (size_t i = 0; i < obj_tuple->size(); ++i) {
      SaveIdWithDynamicAbstract(obj_tuple->value()[i], abs_tuple->elements()[i]);
    }
  } else if (v->isa<ValueSequence>() && !abs->isa<abstract::AbstractTuple>()) {
    const auto &obj_tuple = v->cast<ValueSequencePtr>();
    if (obj_tuple->size() != 1) {
      MS_LOG(EXCEPTION) << "Not match: obj " << v->ToString() << " and abs " << abs->ToString();
    }
    // Like Unique, has two outputs, but one output is static shape, and should not be stored
    if (abs->BuildShape()->IsDynamic()) {
      (void)id_with_dynamic_abs_.emplace(
        std::make_pair(PyNativeAlgo::Common::GetIdByValue(obj_tuple->value()[0]), abs));
    }
  } else if (!v->isa<ValueSequence>() && !abs->isa<abstract::AbstractTuple>()) {
    if (abs->BuildShape()->IsDynamic()) {
      (void)id_with_dynamic_abs_.emplace(std::make_pair(PyNativeAlgo::Common::GetIdByValue(v), abs));
    }
  } else {
    MS_LOG(EXCEPTION) << "Not match: obj " << v->ToString() << " and abs " << abs->ToString();
  }
}

void DynamicShape::UpdateValueToDynamicShape(const ValuePtr &value) const {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<mindspore::tensor::Tensor>()) {
    auto tensor_value = value->cast<tensor::TensorPtr>();
    auto it = id_with_dynamic_abs_.find(tensor_value->id());
    if (it != id_with_dynamic_abs_.end()) {
      tensor_value->set_base_shape(GetShapeFromAbstract(it->second));
    }
  } else if (value->isa<ValueTuple>()) {
    auto value_tuple = value->cast<ValueTuplePtr>();
    for (const auto &v : value_tuple->value()) {
      UpdateValueToDynamicShape(v);
    }
  } else {
    MS_LOG(DEBUG) << "Out put is not a tensor";
  }
}

void DynamicShape::UpdateInputTensorToDynamicShape(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!op_run_info->base_op_run_info.has_dynamic_input) {
    return;
  }
  // Set tensor dynamic base shape
  for (auto &input_tensor : op_run_info->base_op_run_info.input_tensor) {
    const auto it = id_with_dynamic_abs_.find(input_tensor->id());
    if (it != id_with_dynamic_abs_.end()) {
      input_tensor->set_base_shape(GetShapeFromAbstract(it->second));
    }
  }
}

void DynamicShape::SaveDynShapeAbsForMsFunction(const py::args &args, const py::object &out,
                                                const FuncGraphPtr &ms_func_graph) {
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  auto output_node = ms_func_graph->output();
  MS_EXCEPTION_IF_NULL(output_node);

  // Update input to dynamic
  for (size_t i = 0; i < args.size(); ++i) {
    if (py::isinstance<tensor::Tensor>(args[i])) {
      const auto &input_i_tensor = args[i].cast<tensor::TensorPtr>();
      UpdateValueToDynamicShape(input_i_tensor);
    }
  }

  // Update output to dynamic
  const auto &output_value = PyNativeAlgo::DataConvert::PyObjToValue(out);
  SaveIdWithDynamicAbstract(output_value, output_node->abstract());
  UpdateValueToDynamicShape(output_value);

  // Save output by one id for abs get performance
  if (output_node->abstract()->BuildShape()->IsDynamic()) {
    id_with_dynamic_abs_[PyNativeAlgo::PyParser::GetIdByPyObj(out)] = output_node->abstract();
  }
}

void DynamicShape::SaveOutputDynamicShape(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  // Save dynamic abs
  if (op_run_info->base_op_run_info.has_dynamic_output) {
    SaveIdWithDynamicAbstract(v, op_run_info->base_op_run_info.abstract);
  }
}

void DynamicShape::SetDynamicInput(const py::object &cell, const py::args &args) {
  auto &dynamic_index = feed_dynamic_input_[PyNativeAlgo::PyParser::GetIdByPyObj(cell)];
  dynamic_index.resize(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    auto value = PyNativeAlgo::DataConvert::PyObjToValue(args[i]);
    auto abstract = value->ToAbstract()->Broaden();
    MS_EXCEPTION_IF_NULL(abstract);
    dynamic_index[i] = abstract;
  }
}

void DynamicShape::SetFeedDynamicInputAbs(const py::object &cell, const py::args &args, bool is_auto) {
  if (!HasFeedDynamicInput()) {
    return;
  }
  const auto &cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  auto it = feed_dynamic_input_.find(cell_id);
  if (it == feed_dynamic_input_.end()) {
    return;
  }
  if (it->second.size() != args.size()) {
    MS_LOG(DEBUG) << "Dynamic input size " << it->second.size() << " is not equal to real input size " << args.size();
    return;
  }
  bool id_changed = false;
  for (size_t i = 0; i < args.size(); i++) {
    auto abs = it->second.at(i);
    MS_EXCEPTION_IF_NULL(abs);
    auto shape = abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->IsDynamic()) {
      const auto &arg_id = PyNativeAlgo::PyParser::GetIdByPyObj(args[i]);
      MS_LOG(DEBUG) << "Set arg " << i << ", id " << arg_id << " to be dynamic shape; Arg self abs: "
                    << PyNativeAlgo::DataConvert::PyObjToValue(args[i])->ToAbstract()->Broaden()->ToString()
                    << ", dynamic abs: " << abs->ToString();
      id_with_dynamic_abs_[arg_id] = abs;
      PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->EraseFromNodeAbsMap(arg_id);
      id_changed = true;
    }
  }
  if (id_changed && !is_auto) {
    CheckPreviousTopCellCanBeDynamicShape(cell, args);
  }
}

py::object DynamicShape::GetDynamicInput(const py::object &actual_input) const {
  if (py::isinstance<py::tuple>(actual_input)) {
    py::tuple tuple_actual_args = py::cast<py::tuple>(actual_input);
    size_t args_size = tuple_actual_args.size();
    py::tuple dyn_shape_args = py::tuple(args_size);
    for (size_t i = 0; i < args_size; ++i) {
      dyn_shape_args[i] = GetDynamicInput(tuple_actual_args[i]);
    }
    return dyn_shape_args;
  } else if (py::isinstance<py::list>(actual_input)) {
    py::list list_actual_args = py::cast<py::list>(actual_input);
    size_t args_size = list_actual_args.size();
    py::list dyn_shape_args;
    for (size_t i = 0; i < args_size; ++i) {
      dyn_shape_args.append(GetDynamicInput(list_actual_args[i]));
    }
    return dyn_shape_args;
  } else if (py::isinstance<tensor::Tensor>(actual_input)) {
    const auto iter = id_with_dynamic_abs_.find(PyNativeAlgo::PyParser::GetIdByPyObj(actual_input));
    if (iter != id_with_dynamic_abs_.end()) {
      auto tensor_ptr = py::cast<tensor::TensorPtr>(actual_input);
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      auto dyn_tensor = std::make_shared<tensor::Tensor>(tensor_ptr->data_type(), tensor_ptr->shape_c());
      dyn_tensor->set_base_shape(DynamicShape::GetShapeFromAbstract(iter->second));
      auto py_dyn_tensor = ValueToPyData(dyn_tensor);
      return py_dyn_tensor;
    }
  }
  return actual_input;
}

ValuePtr DynamicShape::SetSensValue(const ValuePtr &value, const TopCellInfoPtr &top_cell) const {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(top_cell);
  if (value->isa<ValueTuple>()) {
    ValuePtrList values;
    auto value_tuple = value->cast<ValueTuplePtr>();
    (void)std::transform(value_tuple->value().begin(), value_tuple->value().end(), std::back_inserter(values),
                         [this, top_cell](const ValuePtr &elem) { return SetSensValue(elem, top_cell); });
    return std::make_shared<ValueTuple>(values);
  } else if (value->isa<ValueList>()) {
    ValuePtrList values;
    auto value_list = value->cast<ValueTuplePtr>();
    (void)std::transform(value_list->value().begin(), value_list->value().end(), std::back_inserter(values),
                         [this, top_cell](const ValuePtr &elem) { return SetSensValue(elem, top_cell); });
    return std::make_shared<ValueList>(values);
  } else if (value->isa<tensor::Tensor>()) {
    auto tensor_value = value->cast<tensor::TensorPtr>();
    // Sens tensor has the same shape and dtype with output tensor
    auto sens_tensor = std::make_shared<tensor::Tensor>(tensor_value->data_type(), tensor_value->shape());
    sens_tensor->set_base_shape(tensor_value->base_shape_ptr());
    sens_tensor->set_is_forward_output(true);
    sens_tensor->set_user_data(kTensorUserDataIsSensTensor, MakeValue(true));
    MS_LOG(DEBUG) << "Make new tensor for sens id " << sens_tensor->id() << ", abstract "
                  << sens_tensor->ToAbstract()->ToString();
    top_cell->SetTensorIdWithTensorObject(sens_tensor->id(), sens_tensor);
    return sens_tensor;
  } else {
    return value;
  }
}

ValuePtr DynamicShape::GetSensValueForDynamicShapeOutput(const TopCellInfoPtr &top_cell, const ValuePtr &v,
                                                         const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(top_cell);
  MS_EXCEPTION_IF_NULL(node);
  if (!PyNativeAlgo::Common::ValueHasDynamicShape(v)) {
    return v;
  }
  MS_LOG(DEBUG) << "Set sens value with op info: " << kSensInfo;
  // Create sens value
  auto sens_value = SetSensValue(v, top_cell);
  // Ready for replace
  std::vector<tensor::TensorPtr> all_op_tensors;
  // Get output tensors
  TensorValueToTensor(sens_value, &all_op_tensors);
  // Save all tensors info of current op
  top_cell->SaveOpInfo(kSensInfo, all_op_tensors);
  return sens_value;
}

void DynamicShape::UpdateSensValueForDynamicShapeOutput(const TopCellInfoPtr &top_cell, const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(top_cell);
  if (top_cell->op_info_with_tensor_id().count(kSensInfo) == 0) {
    return;
  }
  MS_LOG(DEBUG) << "Update sens value with op info: " << kSensInfo;
  std::vector<tensor::TensorPtr> new_tensors;
  TensorValueToTensor(v, &new_tensors);
  if (new_tensors.empty()) {
    MS_LOG(DEBUG) << "The size of added forward tensors is zero, no need to update.";
    return;
  }
  // Update new output tensor info in bprop graph
  const auto &pre_op_tensor_id = top_cell->op_info_with_tensor_id().at(kSensInfo);
  if (pre_op_tensor_id.size() != new_tensors.size()) {
    MS_LOG(EXCEPTION) << "The size of pre op tensor id: " << pre_op_tensor_id.size()
                      << " is not equal to the size of all tensors of current op " << new_tensors.size();
  }
  const auto &pre_tensor_id_with_tensor_object = top_cell->tensor_id_with_tensor_object();
  for (size_t i = 0; i < pre_op_tensor_id.size(); ++i) {
    auto pre_id = pre_op_tensor_id[i];
    if (pre_tensor_id_with_tensor_object.find(pre_id) == pre_tensor_id_with_tensor_object.end()) {
      continue;
    }
    const auto &old_tensor_list = pre_tensor_id_with_tensor_object.at(pre_id);
    if (old_tensor_list.empty()) {
      MS_LOG(EXCEPTION) << "Get empty old tensor list";
    }
    const auto &old_tensor = old_tensor_list.front();
    const auto &new_tensor = new_tensors[i];
    MS_EXCEPTION_IF_NULL(old_tensor);
    MS_EXCEPTION_IF_NULL(new_tensor);
    MS_LOG(DEBUG) << "Replace Old tensor id " << old_tensor->id() << ", shape and type "
                  << old_tensor->GetShapeAndDataTypeInfo() << "; With new tensor id " << new_tensor->id()
                  << ", shape and dtype " << new_tensor->GetShapeAndDataTypeInfo();
    (void)old_tensor->set_shape(new_tensor->shape());
    (void)old_tensor->set_data_type(new_tensor->data_type());
    // New tensor have no device address, let old tensor device address nullptr for realloc in later stage
    old_tensor->set_device_address(nullptr);
  }
}

TopCellInfoPtr DynamicShape::ChangeTopCellToDynamicShapeBySetInputs(const TopCellInfoPtr &top_cell,
                                                                    const std::vector<ShapeVector> &new_args_shape,
                                                                    const py::object &cell) {
  MS_EXCEPTION_IF_NULL(top_cell);
  // Change args shape
  for (size_t i = 0; i < new_args_shape.size(); ++i) {
    top_cell->cell_self_info()->args_shape[i] = std::make_shared<abstract::Shape>(new_args_shape[i]);
  }
  auto it = feed_dynamic_input_.find(PyNativeAlgo::PyParser::GetIdByPyObj(cell));
  if (it != feed_dynamic_input_.end()) {
    for (size_t i = 0; i < new_args_shape.size(); i++) {
      auto abs = it->second.at(i);
      MS_EXCEPTION_IF_NULL(abs);
      auto shape = abs->BuildShape();
      MS_EXCEPTION_IF_NULL(shape);
      if (shape->IsDynamic()) {
        const auto &arg_id = top_cell->cell_self_info()->args_id[i];
        MS_LOG(DEBUG) << "Set arg " << i << ", id " << arg_id << ", dynamic abs: " << abs->ToString();
        id_with_dynamic_abs_[arg_id] = abs;
        PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->EraseFromNodeAbsMap(arg_id);
      }
    }
  }
  top_cell->ChangeTopCellInfo(new_args_shape.size());
  return top_cell;
}

TopCellInfoPtr DynamicShape::ChangeTopCellToDynamicShapeByAuto(const TopCellInfoPtr &top_cell,
                                                               const std::vector<ShapeVector> &new_args_shape,
                                                               const py::object &cell, const py::args &args) {
  MS_EXCEPTION_IF_NULL(top_cell);
  // Change args shape
  for (size_t i = 0; i < args.size(); ++i) {
    top_cell->cell_self_info()->args_shape[i] = std::make_shared<abstract::Shape>(new_args_shape[i]);
    if (py::isinstance<tensor::Tensor>(args[i])) {
      auto tensor = args[i].cast<tensor::TensorPtr>();
      tensor->set_base_shape(top_cell->cell_self_info()->args_shape[i]);
    }
    const auto &arg_id = PyNativeAlgo::PyParser::GetIdByPyObj(args[i]);
    PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->EraseFromNodeAbsMap(arg_id);
  }
  // Set to feed dynamic map, later shapes can match it
  MS_LOG(DEBUG) << "Set dynamic input for auto dynamic shape";
  SetDynamicInput(cell, args);
  SetFeedDynamicInputAbs(cell, args, true);
  top_cell->ChangeTopCellInfo(new_args_shape.size());
  return top_cell;
}

void DynamicShape::UpdateTopCellId(const py::args &args) const {
  const auto &grad_executor = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor();
  if (grad_executor->TopCellIsNull() || grad_executor->top_cell()->cell_self_info() == nullptr) {
    return;
  }
  const auto &top_cell = grad_executor->top_cell();
  mindspore::HashMap<std::string, ShapeVector> id_with_shape;
  ShapeVector empty_shape;
  bool has_dynamic_id = false;
  for (size_t i = 0; i < args.size(); i++) {
    const auto &arg_id = PyNativeAlgo::PyParser::GetIdByPyObj(args[i]);
    const auto item = id_with_dynamic_abs_.find(arg_id);
    if (item != id_with_dynamic_abs_.end()) {
      id_with_shape[arg_id] = (GetShapeFromAbstract(item->second))->shape();
      has_dynamic_id = true;
    } else {
      id_with_shape[arg_id] = empty_shape;
    }
  }
  if (!has_dynamic_id) {
    return;
  }
  // Check current top cell need change id to dynamic id
  const auto &args_id = top_cell->cell_self_info()->args_id;
  bool need_change = std::any_of(args_id.begin(), args_id.end(), [&id_with_shape](const std::string &id) {
    return id_with_shape.find(id) != id_with_shape.end();
  });
  if (need_change) {
    // Change args shape
    for (size_t i = 0; i < id_with_shape.size(); ++i) {
      const auto it = std::next(id_with_shape.begin(), i);
      if (!it->second.empty() && top_cell->cell_self_info()->args_id[i] == it->first) {
        top_cell->cell_self_info()->args_shape[i] = std::make_shared<abstract::Shape>(it->second);
      }
    }
    top_cell->ChangeTopCellInfo(top_cell->cell_self_info()->args_id.size());
  }
}

TopCellInfoPtr DynamicShape::GetTopCellWithDynamicShape(const py::object &cell, const py::args &args, bool is_auto) {
  // Current return nullptr for disable auto dynamic shape feature; Later after a complete test will enable this
  if (is_auto && !py::isinstance<py::none>(cell)) {
    return nullptr;
  }
  const auto &cell_self_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  const auto &top_cell_list = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor()->top_cell_list();
  const auto it = std::find_if(top_cell_list.begin(), top_cell_list.end(), [&cell_self_id](const TopCellInfoPtr &elem) {
    return elem->cell_self_info() != nullptr && elem->cell_self_info()->cell_self_id == cell_self_id;
  });
  if (it != top_cell_list.end()) {
    const auto &elem = *it;
    if (elem->dynamic_shape()) {
      MS_LOG(DEBUG) << "Elem has already dynamic shape";
      return nullptr;
    }
    std::vector<ShapeVector> new_args_shape;
    FindMatchTopCell(elem, args, &new_args_shape);
    // Change top cell to be dynamic
    if (new_args_shape.size() == args.size()) {
      if (is_auto) {
        return ChangeTopCellToDynamicShapeByAuto(elem, new_args_shape, cell, args);
      } else {
        return ChangeTopCellToDynamicShapeBySetInputs(elem, new_args_shape, cell);
      }
    }
  }
  UpdateTopCellId(args);
  return nullptr;
}

void DynamicShape::CheckPreviousTopCellCanBeDynamicShape(const py::object &cell, const py::args &args) {
  if (!PyNativeAlgo::Common::GetPyNativeExecutor()->grad_flag()) {
    return;
  }
  // In ms_function, new graph run before construct, so top cell create first; After that, set_dynamic_input call
  // in construct, here change top cell to dynamic.
  if (GetTopCellWithDynamicShape(cell, args, false) != nullptr) {
    MS_LOG(DEBUG) << "Convert ms_function top cell to dynamic shape.";
  }
}

py::object DynamicShape::GetDynShape(const py::args &args) const {
  const auto &obj = args[0];
  // infer type
  const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(obj);
  auto abs = v->ToAbstract();
  std::set<TypePtr> valid_params_types = {kTensorType};
  (void)CheckAndConvertUtils::CheckSubClass("shape type", abs->BuildType(), valid_params_types, "Shape");
  // infer shape
  const auto &base_shape_ptr = obj.cast<tensor::TensorPtr>()->base_shape_ptr();
  if (base_shape_ptr != nullptr) {
    auto value = MakeValue(base_shape_ptr->cast<abstract::ShapePtr>()->shape());
    return ValueToPyData(value);
  }
  const auto &arg_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
  auto it = id_with_dynamic_abs_.find(arg_id);
  if (it != id_with_dynamic_abs_.end()) {
    auto value = MakeValue(GetShapeFromAbstract(it->second)->shape());
    return ValueToPyData(value);
  }
  auto value = MakeValue(GetTensorShape(obj.cast<tensor::TensorPtr>()));
  return ValueToPyData(value);
}

void DynamicShape::FindMatchTopCell(const TopCellInfoPtr &top_cell, const py::args &args,
                                    std::vector<ShapeVector> *new_args_shape) const {
  MS_EXCEPTION_IF_NULL(top_cell);
  MS_EXCEPTION_IF_NULL(new_args_shape);
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &cur_value_abs = PyNativeAlgo::DataConvert::PyObjToValue(args[i])->ToAbstract();
    MS_EXCEPTION_IF_NULL(cur_value_abs);
    const auto &cur_type = PyNativeAlgo::Common::GetTypeFromAbstract(cur_value_abs);
    const auto &elem_type = top_cell->cell_self_info()->args_type[i];
    // Type is not the same
    if (cur_type->hash() != elem_type->hash()) {
      MS_LOG(DEBUG) << "The " << i << "th args type is not the same, cur is " << cur_type->ToString()
                    << " and the elem is " << elem_type->ToString();
      return;
    }
    // Check shape
    const auto &cur_shape = GetShapeFromAbstract(cur_value_abs)->shape();
    auto elem_shape = top_cell->cell_self_info()->args_shape[i]->shape();
    if (cur_shape.size() != elem_shape.size()) {
      MS_LOG(DEBUG) << "The " << i << "th args shape size is not the same, cur is " << cur_shape.size()
                    << " and the elem is " << elem_shape.size();
      return;
    }
    ShapeVector new_shape;
    for (size_t j = 0; j < cur_shape.size(); ++j) {
      if (cur_shape[j] == elem_shape[j]) {
        (void)new_shape.emplace_back(cur_shape[j]);
      } else {
        (void)new_shape.emplace_back(-1);
      }
    }
    // All shape can not be -1, and all shape can not be actual.
    bool is_any_unknown = std::any_of(new_shape.begin(), new_shape.end(), [](int64_t s) { return s == -1; });
    bool is_any_actual = std::any_of(new_shape.begin(), new_shape.end(), [](int64_t s) { return s != -1; });
    if (is_any_unknown && is_any_actual) {
      (void)new_args_shape->emplace_back(new_shape);
    } else {
      MS_LOG(DEBUG) << "Not support all shape unknown or actual.Cur shape " << cur_shape << ", elem shape "
                    << elem_shape << ", and new shape is " << new_shape;
    }
  }
}
}  // namespace pynative
}  // namespace mindspore
