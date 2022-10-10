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

void DynamicShape::UpdateValueBaseShape(const ValuePtr &v, const AbstractBasePtr &abs) const {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(abs);
  if (v->isa<ValueSequence>() && abs->isa<abstract::AbstractTuple>()) {
    const auto &obj_tuple = v->cast<ValueSequencePtr>();
    const auto &abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
    if (obj_tuple->size() != abs_tuple->size()) {
      MS_LOG(EXCEPTION) << "Obj tuple size " << obj_tuple->size() << ", but abstract tuple size " << abs_tuple->size();
    }
    for (size_t i = 0; i < obj_tuple->size(); ++i) {
      UpdateValueBaseShape(obj_tuple->value()[i], abs_tuple->elements()[i]);
    }
  } else if (v->isa<ValueSequence>() && !abs->isa<abstract::AbstractTuple>()) {
    const auto &obj_tuple = v->cast<ValueSequencePtr>();
    if (obj_tuple->size() != 1) {
      MS_LOG(EXCEPTION) << "Not match: obj " << v->ToString() << " and abs " << abs->ToString();
    }
    SetValueBaseShape(obj_tuple->value()[0], abs);
  } else if (!v->isa<ValueSequence>() && !abs->isa<abstract::AbstractTuple>()) {
    SetValueBaseShape(v, abs);
  } else {
    MS_LOG(EXCEPTION) << "Not match: obj " << v->ToString() << " and abs " << abs->ToString();
  }
}

void DynamicShape::SetValueBaseShape(const ValuePtr &v, const AbstractBasePtr &abs) const {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->BuildShape()->IsDynamic()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    tensor->set_base_shape(GetShapeFromAbstract(abs));
  }
}

void DynamicShape::SaveDynShapeAbsForMsFunction(const py::args &args, const py::object &out,
                                                const FuncGraphPtr &ms_func_graph) const {
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  auto output_node = ms_func_graph->output();
  if (!common::AnfAlgo::IsDynamicShape(output_node)) {
    return;
  }

  // Update output to dynamic
  const auto &output_value = PyNativeAlgo::DataConvert::PyObjToValue(out);
  MS_EXCEPTION_IF_NULL(output_node);
  UpdateValueBaseShape(output_value, output_node->abstract());
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

void DynamicShape::SetFeedDynamicInputAbs(const py::object &cell, const py::args &args) {
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
  for (size_t i = 0; i < args.size(); ++i) {
    auto abs = it->second.at(i);
    MS_EXCEPTION_IF_NULL(abs);
    if (!SetFeedTupleDynamicInputAbs(abs, args[i], i)) {
      break;
    }
  }
}

bool DynamicShape::SetFeedTupleDynamicInputAbs(const abstract::AbstractBasePtr &abs, const py::object &arg, size_t i) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_list = abs->cast<abstract::AbstractSequencePtr>();
    if (!py::isinstance<py::tuple>(arg) && !py::isinstance<py::list>(arg)) {
      MS_LOG(EXCEPTION) << "Arg is not tuple or list " << std::string(py::str(arg));
    }
    auto args_list = py::cast<py::tuple>(arg);
    if (args_list.size() != abs_list->elements().size()) {
      MS_LOG(EXCEPTION) << "Dynamic abs size " << abs_list->elements().size() << " is not equal to real args size "
                        << args_list.size();
    }
    for (size_t j = 0; j < args_list.size(); ++j) {
      if (!SetFeedTupleDynamicInputAbs(abs_list->elements()[j], args_list[j], j)) {
        return false;
      }
    }
  } else {
    auto shape = abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->IsDynamic()) {
      auto tensor = arg.cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto dynamic_shape = shape->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(dynamic_shape);
      if (tensor->shape().size() != dynamic_shape->shape().size()) {
        // If shape size not equal, change do dynamic rank in make new top automaticity
        return false;
      }
      const auto &arg_id = PyNativeAlgo::PyParser::GetIdByPyObj(arg);
      const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(arg);
      MS_LOG(DEBUG) << "Set arg " << i << ", id " << arg_id
                    << " to be dynamic shape; Arg self abs: " << v->ToAbstract()->Broaden()->ToString()
                    << ", dynamic abs: " << abs->ToString();
      PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->EraseFromNodeAbsMap(arg_id);
      UpdateValueBaseShape(v, abs);
    }
  }
  return true;
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
  const auto it = feed_dynamic_input_.find(PyNativeAlgo::PyParser::GetIdByPyObj(cell));
  if (it == feed_dynamic_input_.end()) {
    return nullptr;
  }
  for (size_t i = 0; i < new_args_shape.size(); ++i) {
    top_cell->cell_self_info()->args_shape[i] = std::make_shared<abstract::Shape>(new_args_shape[i]);
    const auto &arg_id = top_cell->cell_self_info()->args_id[i];
    it->second.at(i)->set_shape(top_cell->cell_self_info()->args_shape[i]);
    MS_LOG(DEBUG) << "Change cur top cell arg " << i << ", id " << arg_id
                  << ", dynamic abs: " << it->second.at(i)->ToString();
    PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->SetNodeAbsMapById(arg_id, it->second.at(i));
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
  top_cell->ChangeTopCellInfo(new_args_shape.size());
  return top_cell;
}

void DynamicShape::UpdateTopCellId(const py::args &args) const {
  const auto &grad_executor = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor();
  if (grad_executor->TopCellIsNull() || grad_executor->top_cell()->cell_self_info() == nullptr) {
    return;
  }
  const auto &top_cell = grad_executor->top_cell();
  const auto &args_id = top_cell->cell_self_info()->args_id;
  size_t arg_size = args.size();
  if (args_id.size() != arg_size) {
    return;
  }
  bool change_to_dynamic_shape = false;
  for (size_t i = 0; i < arg_size; i++) {
    const auto &arg_id = PyNativeAlgo::PyParser::GetIdByPyObj(args[i]);
    if (arg_id != args_id[i]) {
      continue;
    }
    const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(args[i]);
    MS_EXCEPTION_IF_NULL(v);
    const auto &abs = v->ToAbstract()->Broaden();
    if (!abs->BuildShape()->IsDynamic()) {
      continue;
    }
    change_to_dynamic_shape = true;
    auto shape = GetShapeFromAbstract(abs)->shape();
    top_cell->cell_self_info()->args_shape[i] = std::make_shared<abstract::Shape>(shape);
  }
  if (change_to_dynamic_shape) {
    top_cell->ChangeTopCellInfo(arg_size);
  }
}

TopCellInfoPtr DynamicShape::GetTopCellWithDynamicShape(const py::object &cell, const py::args &args, bool is_auto) {
  const auto &cell_self_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  const auto grad_executor = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor();
  size_t grad_order = grad_executor->grad_order();
  std::vector<TopCellInfoPtr> match_top_cell_list;
  std::copy_if(grad_executor->top_cell_list().begin(), grad_executor->top_cell_list().end(),
               std::back_inserter(match_top_cell_list), [&cell_self_id, grad_order](const TopCellInfoPtr &elem) {
                 return elem->cell_self_info() != nullptr && elem->cell_self_info()->cell_self_id == cell_self_id &&
                        grad_order == elem->grad_order() && !elem->cell_self_info()->is_unknown_rank;
               });
  for (const auto &it : match_top_cell_list) {
    std::vector<ShapeVector> new_args_shape;
    FindMatchTopCell(it, args, &new_args_shape);
    if (args.size() != new_args_shape.size()) {
      continue;
    }
    // Change top cell to be dynamic
    if (is_auto) {
      return ChangeTopCellToDynamicShapeByAuto(it, new_args_shape, cell, args);
    } else {
      return ChangeTopCellToDynamicShapeBySetInputs(it, new_args_shape, cell);
    }
  }
  // Like TrainOneStep, it is top cell, but set_inputs set for Net, so need change TrainOneStep top cell to dynamic
  // shape
  if (!is_auto) {
    UpdateTopCellId(args);
  }
  return nullptr;
}

void DynamicShape::CheckPreviousTopCellCanBeDynamicShape(const py::object &cell, const py::args &args) {
  if (!PyNativeAlgo::Common::GetPyNativeExecutor()->grad_flag()) {
    return;
  }
  // In ms_function, new graph run before construct, so top cell create first; After that, set_dynamic_input call
  // in construct, here change top cell to dynamic.
  if (GetTopCellWithDynamicShape(cell, args, false) != nullptr) {
    MS_LOG(DEBUG) << "Convert cur top cell to dynamic shape.";
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
  ShapeVector shape_out{};
  auto tensor = obj.cast<tensor::TensorPtr>();
  if (tensor != nullptr) {
    if (tensor->base_shape_ptr() != nullptr) {
      auto shape = tensor->base_shape_ptr()->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(shape);
      shape_out = shape->shape();
    } else {
      shape_out = tensor->shape();
    }
  }
  py::tuple ret(shape_out.size());
  for (size_t i = 0; i < shape_out.size(); ++i) {
    ret[i] = shape_out[i];
  }
  return std::move(ret);
}

void DynamicShape::FindMatchTopCell(const TopCellInfoPtr &top_cell, const py::args &args,
                                    std::vector<ShapeVector> *new_args_shape) const {
  MS_EXCEPTION_IF_NULL(top_cell);
  MS_EXCEPTION_IF_NULL(new_args_shape);
  bool change_to_unknown_rank = false;
  std::vector<TypePtr> cur_type_list;
  std::vector<ShapeVector> cur_shape_list;
  std::vector<ShapeVector> elem_shape_list;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &cur_value_abs = PyNativeAlgo::DataConvert::PyObjToValue(args[i])->ToAbstract();
    MS_EXCEPTION_IF_NULL(cur_value_abs);
    (void)cur_type_list.emplace_back(PyNativeAlgo::Common::GetTypeFromAbstract(cur_value_abs));
    (void)cur_shape_list.emplace_back(GetShapeFromAbstract(cur_value_abs)->shape());
    (void)elem_shape_list.emplace_back(top_cell->cell_self_info()->args_shape[i]->shape());
  }
  if (cur_shape_list == elem_shape_list) {
    MS_LOG(DEBUG) << "All shape is same";
    return;
  }
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &elem_type = top_cell->cell_self_info()->args_type[i];
    // Type is not the same
    if (cur_type_list[i]->hash() != elem_type->hash()) {
      MS_LOG(DEBUG) << "The " << i << "th args type is not the same, cur is " << cur_type_list[i]->ToString()
                    << " and the elem is " << elem_type->ToString();
      return;
    }
    size_t cur_shape_size = cur_shape_list[i].size();
    ShapeVector new_shape;
    // Rank dynamic
    if (change_to_unknown_rank || cur_shape_size != elem_shape_list[i].size()) {
      MS_LOG(DEBUG) << "The " << i << "th args shape size is not the same, cur is " << cur_shape_size
                    << " and the elem is " << elem_shape_list[i].size() << ", change shape to dynamic rank";
      new_shape.emplace_back(abstract::Shape::kShapeRankAny);
      change_to_unknown_rank = true;
    } else {
      // Shape dynamic
      std::fill_n(std::back_inserter(new_shape), cur_shape_size, abstract::Shape::kShapeDimAny);
    }
    (void)new_args_shape->emplace_back(new_shape);
    MS_LOG(DEBUG) << "Cur shape " << cur_shape_list[i] << ", elem shape " << elem_shape_list[i] << ", new shape "
                  << new_shape;
  }
  // Change UNKNOWN_DIM to UNKNOWN_RANK if both exist
  if (change_to_unknown_rank) {
    for (auto &shape : *new_args_shape) {
      if (shape.back() == abstract::Shape::kShapeDimAny) {
        shape.clear();
        (void)shape.emplace_back(abstract::Shape::kShapeRankAny);
      }
    }
    top_cell->cell_self_info()->is_unknown_rank = true;
  }
}
}  // namespace pynative
}  // namespace mindspore
