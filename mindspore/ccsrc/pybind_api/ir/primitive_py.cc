/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "pybind_api/ir/primitive_py.h"
#include <mutex>
#include "ir/signature.h"
#include "./common.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pybind11/pytypes.h"
#include "utils/convert_utils_base.h"
#include "utils/primitive_utils.h"
#include "utils/base_ref_extends.h"
#include "pybind_api/api_register.h"
#include "pybind_api/export_flags.h"
#include "pybind_api/ir/base_ref_py.h"

namespace mindspore {
namespace {
constexpr auto kBpropAttrName = "bprop";
constexpr auto kCellHookAttrName = "cell_hook";
constexpr auto kCellIDAttrName = "cell_id";
void SyncData(const py::object &arg) {
  if (py::isinstance<py::tuple>(arg)) {
    py::tuple arg_list = py::cast<py::tuple>(arg);
    for (size_t i = 0; i < arg_list.size(); i++) {
      SyncData(arg_list[i]);
    }
  }
  if (py::isinstance<tensor::Tensor>(arg)) {
    auto tensor = py::cast<tensor::TensorPtr>(arg);
    (void)tensor->data_sync();
  }
}
}  // namespace
std::map<std::string, py::object> PrimitivePy::hook_grad_;
static ValuePtr PyArgToValue(const py::object &arg) {
  if (py::isinstance<SignatureEnumKind>(arg) &&
      py::cast<SignatureEnumKind>(arg) == SignatureEnumKind::kKindEmptyDefaultValue) {
    return nullptr;
  }
  return parse::data_converter::PyDataToValue(arg);
}

void PrimitivePy::set_signatures(
  std::vector<std::tuple<std::string, SignatureEnumRW, SignatureEnumKind, py::object, SignatureEnumDType>> signatures) {
  signatures_.clear();
  for (auto &signature : signatures) {
    auto [name, rw, kind, arg_default, dtype] = signature;
    auto default_value = PyArgToValue(arg_default);
    signatures_.emplace_back(name, rw, kind, default_value, dtype);
  }
  set_has_signature(true);
}

py::function PrimitivePy::GetBpropFunction() {
  static const char *const get_bprop_func_name = "get_bprop";
  if (py::hasattr(python_obj_, get_bprop_func_name)) {
    py::function fn = python_obj_.attr(get_bprop_func_name)().cast<py::function>();
    return fn;
  } else {
    auto fn = GetBpropFunctionByObj(python_obj_);
    return fn;
  }
}

BaseRef PrimitivePy::RunHookFunction(const VectorRef &args) const {
  py::tuple py_args = ConvertDatatoPyTuple(args);
  py::object obj;
  bool is_bprop = this->HasAttr(kBpropAttrName);
  if (is_bprop) {
    SyncData(py_args);
    py::tuple convert_args(py_args.size());
    for (size_t i = 0; i < py_args.size(); i++) {
      convert_args[i] = py::isinstance<tensor::Tensor>(py_args[i])
                          ? parse::python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE,
                                                            parse::PYTHON_MOD_CONVERT_TO_MS_TENSOR, py_args[i])
                          : py_args[i];
    }
    obj = hook_(*convert_args);
    return std::make_shared<PyObjectRef>(obj);
  }
  SyncData(py_args[2]);
  bool is_cell = this->HasAttr(kCellHookAttrName);
  if (is_cell) {
    auto cell_id = GetValue<std::string>(this->GetAttr(kCellIDAttrName));
    auto iter = hook_grad_.find(cell_id);
    if (iter != hook_grad_.end()) {
      auto hook_args = py::tuple(3);
      hook_args[0] = cell_id;
      hook_args[1] = py::make_tuple(iter->second);
      hook_args[2] = py::make_tuple(py_args[2]);
      obj = hook_(*hook_args);
      if (py::isinstance<py::none>(obj)) {
        obj = py_args[2];
      }
      hook_grad_.erase(cell_id);
    } else {
      hook_grad_[cell_id] = py_args[2];
      obj = py_args[2];
    }
  } else {
    // Hook operator for execute variable hook function
    obj = hook_(py::make_tuple(py_args[2]));
    if (py::isinstance<py::none>(obj)) {
      obj = py_args[2];
    }
  }
  obj = py::make_tuple(obj);
  return std::make_shared<PyObjectRef>(obj);
}

py::function PrimitivePy::GetComputeFunction() const {
  static const char *const compute_func_name = "vm_impl";

  if (py::hasattr(python_obj_, compute_func_name)) {
    MS_LOG(INFO) << name() << " compute_func_name";
    py::function fn = python_obj_.attr(compute_func_name).cast<py::function>();
    return fn;
  }

  static const std::string vm_module = "mindspore.ops.vm_impl_registry";
  static const std::string get_vm_impl_fn = "get_vm_impl_fn";
  MS_LOG(INFO) << name() << ": get_vm_impl_fn";
  py::function get_fn = parse::python_adapter::GetPyFn(vm_module, get_vm_impl_fn);
  py::function vm_fn = get_fn(python_obj_);
  if (py::isinstance<py::none>(vm_fn)) {
    MS_LOG(WARNING) << "Cannot find " << python_obj_.attr("__class__").attr("__name__").cast<std::string>();
    vm_fn = mindspore::GetComputeFunction(Primitive::name());
  }
  return vm_fn;
}

void PrimitivePy::AddPyAttr(const py::str &name, const py::object &obj) {
  std::string attr_name = name;
  ValuePtr converted_ret = nullptr;
  if (py::isinstance<py::module>(obj)) {
    MS_LOG(EXCEPTION) << "AddPyAttr failed, obj should not be py::module";
  }
  bool converted = parse::ConvertData(obj, &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type: " << std::string(py::str(obj));
  }
  (void)this->AddAttr(attr_name, converted_ret);
}

py::dict PrimitivePy::GetAttrDict() {
  py::dict attr_dict;
  for (auto &attr : attrs_) {
    attr_dict[py::str(attr.first)] = ValuePtrToPyData(attr.second);
  }
  return attr_dict;
}

void PrimitivePy::CopyHookFunction(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (!primitive->isa<PrimitivePy>()) {
    MS_LOG(EXCEPTION) << "Cannot copy a primtive which is not python primitive hook function to python primitive!";
  }
  auto primitive_py = primitive->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(primitive_py);
  this->set_hook(primitive_py->hook());
}

BaseRef PrimitivePy::RunComputeFunction(const VectorRef &args) const {
  auto py_args = ConvertDatatoPyTuple(args);
  auto result = this->RunPyComputeFunction(py_args);
  if (py::isinstance<py::none>(result)) {
    return std::make_shared<BaseRef>(nullptr);
  }
  return std::make_shared<PyObjectRef>(result);
}

py::object PrimitivePy::RunPyComputeFunction(const py::tuple &py_args) const {
  auto func = this->GetComputeFunction();
  if (py::isinstance<py::none>(func)) {
    return py::none();
  }
  auto result = func(*py_args);
  return result;
}

bool PrimitivePy::HasComputeFunction() const {
  auto func = GetComputeFunction();
  if (py::isinstance<py::none>(func)) {
    return false;
  }
  return true;
}

PrimitivePtr PrimitivePy::Clone() {
  auto clone_fn = python_obj_.attr("_clone");
  py::object new_obj = clone_fn();
  auto cloned_prim = new_obj.cast<PrimitivePyPtr>();
  return cloned_prim;
}

py::dict PrimitivePy::RunInfer(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  auto infer_fuc = python_obj_.attr("__infer__");
  return infer_fuc(*args);
}

REGISTER_PYBIND_DEFINE(Primitive_, ([](const py::module *m) {
                         (void)py::enum_<PrimType>(*m, "prim_type", py::arithmetic())
                           .value("unknown", PrimType::kPrimTypeUnknown)
                           .value("builtin", PrimType::kPrimTypeBuiltIn)
                           .value("py_infer_shape", PrimType::kPrimTypePyInferShape)
                           .value("user_custom", PrimType::kPrimTypeUserCustom);
                         (void)py::class_<PrimitivePy, std::shared_ptr<PrimitivePy>>(*m, "Primitive_")
                           .def_readonly(PYTHON_PRIMITIVE_FLAG, &PrimitivePy::parse_info_)
                           .def(py::init<py::str &, py::object>())
                           .def("add_attr", &PrimitivePy::AddPyAttr, "add primitive attr")
                           .def("get_attr_dict", &PrimitivePy::GetAttrDict, "get primitive attr")
                           .def("set_prim_type", &PrimitivePy::set_prim_type, "Set primitive type.")
                           .def("set_is_const_value", &PrimitivePy::set_is_const_value, "Set primitive is const value.")
                           .def("set_signatures", &PrimitivePy::set_signatures, "Set primitive inputs signature.")
                           .def("register_hook", &PrimitivePy::set_hook, "Set primitive hook function.")
                           .def("set_instance_name", &PrimitivePy::set_instance_name, "Set primitive instance name.");
                       }));
}  // namespace mindspore
