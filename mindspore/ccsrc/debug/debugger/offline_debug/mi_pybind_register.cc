/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "debugger/offline_debug/dbg_services.h"

PYBIND11_MODULE(_mindspore_offline_debug, m) {
  m.doc() = "pybind11 debug services api";
  py::class_<DbgServices>(m, "DbgServices")
    .def(py::init<bool>())
    .def("Initialize", &DbgServices::Initialize)
    .def("AddWatchpoint", &DbgServices::AddWatchpoint)
    .def("RemoveWatchpoint", &DbgServices::RemoveWatchpoint)
    .def("CheckWatchpoints", &DbgServices::CheckWatchpoints)
    .def("ReadTensors", &DbgServices::ReadTensors)
    .def("GetVersion", &DbgServices::GetVersion);

  py::class_<parameter>(m, "parameter")
    .def(py::init<std::string, bool, double, bool, double>())
    .def("get_name", &parameter::get_name)
    .def("get_disabled", &parameter::get_disabled)
    .def("get_value", &parameter::get_value)
    .def("get_hit", &parameter::get_hit)
    .def("get_actual_value", &parameter::get_actual_value);

  py::class_<watchpoint_hit>(m, "watchpoint_hit")
    .def(py::init<std::string, uint32_t, int, uint32_t, std::vector<parameter_t>, int32_t, uint32_t, uint32_t>())
    .def("get_name", &watchpoint_hit::get_name)
    .def("get_slot", &watchpoint_hit::get_slot)
    .def("get_condition", &watchpoint_hit::get_condition)
    .def("get_watchpoint_id", &watchpoint_hit::get_watchpoint_id)
    .def("get_parameters", &watchpoint_hit::get_parameters)
    .def("get_error_code", &watchpoint_hit::get_error_code)
    .def("get_device_id", &watchpoint_hit::get_device_id)
    .def("get_root_graph_id", &watchpoint_hit::get_root_graph_id);

  py::class_<tensor_info>(m, "tensor_info")
    .def(py::init<std::string, uint32_t, uint32_t, uint32_t, uint32_t, bool>())
    .def("get_node_name", &tensor_info::get_node_name)
    .def("get_slot", &tensor_info::get_slot)
    .def("get_iteration", &tensor_info::get_iteration)
    .def("get_device_id", &tensor_info::get_device_id)
    .def("get_root_graph_id", &tensor_info::get_root_graph_id)
    .def("get_is_parameter", &tensor_info::get_is_parameter);

  py::class_<tensor_data>(m, "tensor_data")
    .def(py::init<char *, uint64_t, int, std::vector<int64_t>>())
    .def("get_data_ptr", &tensor_data::get_data_ptr)
    .def("get_data_size", &tensor_data::get_data_size)
    .def("get_dtype", &tensor_data::get_dtype)
    .def("get_shape", &tensor_data::get_shape);
}
