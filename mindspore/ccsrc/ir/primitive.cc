/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "ir/primitive.h"
#include <mutex>
#include <utility>
#include "ir/signature.h"
#include "operator/ops.h"
#include "./common.h"
#include "pipeline/parse/python_adapter.h"
#include "pipeline/parse/data_converter.h"
#include "pybind11/pytypes.h"
#include "utils/convert_utils.h"

#include "pybind_api/api_register.h"
#include "pybind_api/export_flags.h"

namespace mindspore {
using mindspore::abstract::AbstractFunction;

abstract::AbstractBasePtr Primitive::ToPrimAbstract(const AnfNodePtr &anf_node) {
  auto prim_func = std::make_shared<abstract::PrimitiveAbstractClosure>(shared_from_base<Primitive>(), anf_node);
  return prim_func;
}

static py::function GetBpropFunctionByObj(py::object obj) {
  static const std::string get_bprop_fn = "get_bprop_fn";
  static const std::string ad_module = "mindspore.ops._grad";
  py::function fn = parse::python_adapter::GetPyFn(ad_module, get_bprop_fn)(obj);
  return fn;
}

py::function Primitive::GetBpropFunction() {
  auto fn = GetBpropFunctionByObj(py::str(name()));
  if (fn.is_none()) {
    MS_LOG(WARNING) << "Can't find bprop function for " << name();
  }
  return fn;
}

py::function Primitive::GetComputeFunction() {
  static const std::string module = "mindspore._extends.builtin_operations";
  py::module mod = py::module::import(common::SafeCStr(module));
  if (!py::hasattr(mod, common::SafeCStr(name()))) {
    PyErr_SetString(PyExc_NotImplementedError, common::SafeCStr(name()));
    // If raise AttributeError, user can't understand. This case need raise NotImplementedError.
    throw py::error_already_set();
  }
  py::object fn = mod.attr(common::SafeCStr(name()));
  return fn;
}

bool Primitive::operator==(const Value &other) const {
  if (other.isa<Primitive>()) {
    auto other_prim = static_cast<const Primitive &>(other);
    return *this == other_prim;
  } else {
    return false;
  }
}

bool Primitive::operator==(const Primitive &other) const {
  if (name() != other.name()) {
    return false;
  }
  if (attrs_.size() != other.attrs_.size()) {
    return false;
  }
  auto all = std::all_of(attrs_.begin(), attrs_.end(), [&other](const std::pair<std::string, ValuePtr> &item) -> bool {
    if (item.second == nullptr) {
      return false;
    }
    auto iter = other.attrs_.find(item.first);
    if (iter == other.attrs_.end()) {
      return false;
    }
    return *item.second == *iter->second;
  });
  return all;
}

void Primitive::set_signatures(
  std::vector<std::tuple<std::string, SignatureEnumRW, SignatureEnumKind, py::object, SignatureEnumDType>> signatures) {
  signatures_.clear();
  for (auto &signature : signatures) {
    std::string name;
    SignatureEnumRW rw;
    SignatureEnumKind kind;
    py::object default_value;
    SignatureEnumDType dtype;
    std::tie(name, rw, kind, default_value, dtype) = signature;
    signatures_.emplace_back(Signature(name, rw, kind, default_value, dtype));
  }
}

std::string Primitive::GetAttrsText() const {
  if (attrs_.empty()) {
    return "";
  }

  std::ostringstream oss;
  oss << "[";
  bool is_first = true;
  for (auto &attr : attrs_) {
    if (is_first) {
      is_first = false;
    } else {
      oss << ", ";
    }
    oss << attr.first << "=" << attr.second->DumpText();
  }
  oss << "]";

  return oss.str();
}

py::function PrimitivePy::GetBpropFunction() {
  static const char *const get_bprop_func_name = "get_bprop";
  if (py::hasattr(python_obj_, get_bprop_func_name)) {
    py::function fn = python_obj_.attr(get_bprop_func_name)().cast<py::function>();
    return fn;
  } else {
    auto fn = GetBpropFunctionByObj(python_obj_);
    if (fn.is_none()) {
      MS_LOG(WARNING) << "Can't find bprop function for " << name();
    }
    return fn;
  }
}

py::function PrimitivePy::GetComputeFunction() {
  static const char *const compute_func_name = "vm_impl";

  if (py::hasattr(python_obj_, compute_func_name)) {
    MS_LOG(INFO) << "" << name() << " compute_func_name";
    py::function fn = python_obj_.attr(compute_func_name).cast<py::function>();
    return fn;
  }

  static const std::string vm_module = "mindspore.ops.vm_impl_registry";
  static const std::string get_vm_impl_fn = "get_vm_impl_fn";
  MS_LOG(INFO) << "" << name() << ": get_vm_impl_fn";
  py::function get_fn = parse::python_adapter::GetPyFn(vm_module, get_vm_impl_fn);
  py::function vm_fn = get_fn(python_obj_);

  if (py::isinstance<py::none>(vm_fn)) {
    MS_LOG(DEBUG) << "Cannot find " << python_obj_.attr("__class__").attr("__name__").cast<std::string>();
    vm_fn = Primitive::GetComputeFunction();
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
    MS_LOG(EXCEPTION) << "Attribute convert error with type:" << std::string(py::str(obj));
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
                           .def("set_signatures", &PrimitivePy::set_signatures, "Set primitive inputs signature.")
                           .def("set_instance_name", &PrimitivePy::set_instance_name, "Set primitive instance name.");
                       }));
}  // namespace mindspore
