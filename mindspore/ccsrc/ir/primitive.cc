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

#include "ir/primitive.h"
#include <mutex>
#include <utility>
#include "ir/signature.h"
#include "operator/ops.h"
#include "./common.h"
#include "pipeline/parse/python_adapter.h"
#include "pipeline/parse/data_converter.h"
#include "pybind11/pytypes.h"
#include "utils/convert_utils_base.h"
#include "utils/primitive_utils.h"

#include "pybind_api/api_register.h"
#include "pybind_api/export_flags.h"

namespace mindspore {
void PrimitivePy::set_signatures(
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
  set_has_signature(true);
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
    MS_LOG(DEBUG) << "Cannot find " << python_obj_.attr("__class__").attr("__name__").cast<std::string>();
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
                           .def("register_hook", &PrimitivePy::set_hook, "Set primitive hook function.")
                           .def("set_instance_name", &PrimitivePy::set_instance_name, "Set primitive instance name.");
                       }));
}  // namespace mindspore
