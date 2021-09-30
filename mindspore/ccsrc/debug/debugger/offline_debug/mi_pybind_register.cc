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
  (void)py::class_<DbgServices>(m, "DbgServices")
    .def(py::init())
    .def("Initialize", &DbgServices::Initialize)
    .def("AddWatchpoint", &DbgServices::AddWatchpoint)
    .def("RemoveWatchpoint", &DbgServices::RemoveWatchpoint)
    .def("CheckWatchpoints", &DbgServices::CheckWatchpoints)
    .def("ReadTensors", &DbgServices::ReadTensors)
    .def("ReadTensorsBase", &DbgServices::ReadTensorsBase)
    .def("ReadTensorsStat", &DbgServices::ReadTensorsStat)
    .def("GetVersion", &DbgServices::GetVersion);

  (void)py::class_<parameter_t>(m, "parameter")
    .def(py::init<std::string, bool, double, bool, double>())
    .def("get_name", &parameter_t::get_name)
    .def("get_disabled", &parameter_t::get_disabled)
    .def("get_value", &parameter_t::get_value)
    .def("get_hit", &parameter_t::get_hit)
    .def("get_actual_value", &parameter_t::get_actual_value);

  (void)py::class_<watchpoint_hit_t>(m, "watchpoint_hit")
    .def(py::init<std::string, uint32_t, int, uint32_t, std::vector<parameter_t>, int32_t, uint32_t, uint32_t>())
    .def("get_name", &watchpoint_hit_t::get_name)
    .def("get_slot", &watchpoint_hit_t::get_slot)
    .def("get_condition", &watchpoint_hit_t::get_condition)
    .def("get_watchpoint_id", &watchpoint_hit_t::get_watchpoint_id)
    .def("get_parameters", &watchpoint_hit_t::get_parameters)
    .def("get_error_code", &watchpoint_hit_t::get_error_code)
    .def("get_rank_id", &watchpoint_hit_t::get_rank_id)
    .def("get_root_graph_id", &watchpoint_hit_t::get_root_graph_id);

  (void)py::class_<tensor_info_t>(m, "tensor_info")
    .def(py::init<std::string, uint32_t, uint32_t, uint32_t, uint32_t, bool>())
    .def("get_node_name", &tensor_info_t::get_node_name)
    .def("get_slot", &tensor_info_t::get_slot)
    .def("get_iteration", &tensor_info_t::get_iteration)
    .def("get_rank_id", &tensor_info_t::get_rank_id)
    .def("get_root_graph_id", &tensor_info_t::get_root_graph_id)
    .def("get_is_output", &tensor_info_t::get_is_output);

  (void)py::class_<tensor_data_t>(m, "tensor_data")
    .def(py::init<char *, uint64_t, int, std::vector<int64_t>>())
    .def("get_data_ptr", &tensor_data_t::get_data_ptr)
    .def("get_data_size", &tensor_data_t::get_data_size)
    .def("get_dtype", &tensor_data_t::get_dtype)
    .def("get_shape", &tensor_data_t::get_shape);

  (void)py::class_<TensorBaseData>(m, "TensorBaseData")
    .def(py::init<uint64_t, int, std::vector<int64_t>>())
    .def("data_size", &TensorBaseData::data_size)
    .def("dtype", &TensorBaseData::dtype)
    .def("shape", &TensorBaseData::shape);

  (void)py::class_<TensorStatData>(m, "TensorStatData")
    .def(
      py::init<uint64_t, int, std::vector<int64_t>, bool, double, double, double, int, int, int, int, int, int, int>())
    .def("data_size", &TensorStatData::data_size)
    .def("dtype", &TensorStatData::dtype)
    .def("shape", &TensorStatData::shape)
    .def("is_bool", &TensorStatData::is_bool)
    .def("max_value", &TensorStatData::max_value)
    .def("min_value", &TensorStatData::min_value)
    .def("avg_value", &TensorStatData::avg_value)
    .def("count", &TensorStatData::count)
    .def("neg_zero_count", &TensorStatData::neg_zero_count)
    .def("pos_zero_count", &TensorStatData::pos_zero_count)
    .def("nan_count", &TensorStatData::nan_count)
    .def("neg_inf_count", &TensorStatData::neg_inf_count)
    .def("pos_inf_count", &TensorStatData::pos_inf_count)
    .def("zero_count", &TensorStatData::zero_count);
}
