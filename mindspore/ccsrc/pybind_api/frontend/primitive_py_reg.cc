/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/frontend/operator/primitive_py.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace py = pybind11;
void RegPrimitive(const py::module *m) {
  (void)py::enum_<HookType>(*m, "HookType", py::arithmetic())
    .value("CustomOpBprop", HookType::kCustomOpBprop)
    .value("CellCustomBprop", HookType::kCellCustomBprop)
    .value("HookBackward", HookType::kHookBackwardOp)
    .value("TensorHook", HookType::kTensorHook)
    .value("BackwardPreHook", HookType::kBackwardPreHook)
    .value("BackwardHook", HookType::kBackwardHook);
  (void)py::enum_<PrimType>(*m, "prim_type", py::arithmetic())
    .value("unknown", PrimType::kPrimTypeUnknown)
    .value("builtin", PrimType::kPrimTypeBuiltIn)
    .value("py_infer_shape", PrimType::kPrimTypePyInfer)
    .value("user_custom", PrimType::kPrimTypeUserCustom)
    .value("py_infer_check", PrimType::kPrimTypePyCheck);
  (void)py::class_<PrimitivePyAdapter, std::shared_ptr<PrimitivePyAdapter>>(*m, "Primitive_")
    .def_readonly("__primitive_flag__", &PrimitivePyAdapter::parse_info_)
    .def(py::init<py::str &>())
    .def("add_attr", &PrimitivePyAdapter::AddPyAttr, "add primitive attr")
    .def("del_attr", &PrimitivePyAdapter::DelPyAttr, "del primitive attr")
    .def("get_attr_dict", &PrimitivePyAdapter::GetAttrDict, "get primitive attr")
    .def("set_prim_type", &PrimitivePyAdapter::set_prim_type, "Set primitive type.")
    .def("set_const_prim", &PrimitivePyAdapter::set_const_prim, "Set primitive is const.")
    .def("set_inplace_prim", &PrimitivePyAdapter::set_inplace_prim, "Set primitive is inplace primitive.")
    .def("set_const_input_indexes", &PrimitivePyAdapter::set_const_input_indexes, "Set primitive const input indexes.")
    .def("set_signatures", &PrimitivePyAdapter::set_signatures, "Set primitive inputs signature.")
    .def("set_hook_fn", &PrimitivePyAdapter::SetHookFn, "Add primitive hook function.")
    .def("set_instance_name", &PrimitivePyAdapter::set_instance_name, "Set primitive instance name.")
    .def("set_user_data", &PrimitivePyAdapter::SetUserData, "Set primitive user data.")
    .def("get_user_data", &PrimitivePyAdapter::GetUserData, "Get primitive user data.");
}

void RegPrimitiveFunction(const py::module *m) {
  (void)py::class_<PrimitiveFunctionAdapter, std::shared_ptr<PrimitiveFunctionAdapter>>(*m, "PrimitiveFunction_")
    .def_readonly("__primitive_function_flag__", &PrimitiveFunctionAdapter::parse_info_)
    .def(py::init<>())
    .def_property_readonly("name", &PrimitiveFunctionAdapter::name, "Get function name.")
    .def("has_label", &PrimitiveFunctionAdapter::has_label, "Has function attr.")
    .def("set_label", &PrimitiveFunctionAdapter::set_label, "Set function attr.")
    .def("get_label", &PrimitiveFunctionAdapter::get_label, "Get function attr.")
    .def("clone", &PrimitiveFunctionAdapter::clone, "Clone a Primitive and create a PrimitiveFunctionAdapter.");
}
}  // namespace mindspore
