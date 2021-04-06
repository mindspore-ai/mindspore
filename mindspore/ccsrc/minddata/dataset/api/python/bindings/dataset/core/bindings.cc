/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/client.h"  // DE client
#include "minddata/dataset/util/status.h"
#include "pybind11/numpy.h"
#include "minddata/dataset/include/constants.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(GlobalContext, 0, ([](const py::module *m) {
                  (void)py::class_<GlobalContext>(*m, "GlobalContext")
                    .def_static("config_manager", &GlobalContext::config_manager, py::return_value_policy::reference);
                }));

PYBIND_REGISTER(ConfigManager, 0, ([](const py::module *m) {
                  (void)py::class_<ConfigManager, std::shared_ptr<ConfigManager>>(*m, "ConfigManager")
                    .def("__str__", &ConfigManager::ToString)
                    .def("get_auto_num_workers", &ConfigManager::auto_num_workers)
                    .def("get_callback_timeout", &ConfigManager::callback_timeout)
                    .def("get_monitor_sampling_interval", &ConfigManager::monitor_sampling_interval)
                    .def("get_num_parallel_workers", &ConfigManager::num_parallel_workers)
                    .def("get_numa_enable", &ConfigManager::numa_enable)
                    .def("set_numa_enable", &ConfigManager::set_numa_enable)
                    .def("get_op_connector_size", &ConfigManager::op_connector_size)
                    .def("get_rows_per_buffer", &ConfigManager::rows_per_buffer)
                    .def("get_seed", &ConfigManager::seed)
                    .def("set_rank_id", &ConfigManager::set_rank_id)
                    .def("get_worker_connector_size", &ConfigManager::worker_connector_size)
                    .def("set_auto_num_workers", &ConfigManager::set_auto_num_workers)
                    .def("set_auto_worker_config", &ConfigManager::set_auto_worker_config_)
                    .def("set_callback_timeout", &ConfigManager::set_callback_timeout)
                    .def("set_monitor_sampling_interval", &ConfigManager::set_monitor_sampling_interval)
                    .def("stop_dataset_profiler", &ConfigManager::stop_dataset_profiler)
                    .def("get_profiler_file_status", &ConfigManager::get_profiler_file_status)
                    .def("set_num_parallel_workers", &ConfigManager::set_num_parallel_workers)
                    .def("set_op_connector_size", &ConfigManager::set_op_connector_size)
                    .def("set_sending_batches", &ConfigManager::set_sending_batches)
                    .def("set_rows_per_buffer", &ConfigManager::set_rows_per_buffer)
                    .def("set_seed", &ConfigManager::set_seed)
                    .def("set_worker_connector_size", &ConfigManager::set_worker_connector_size)
                    .def("load", [](ConfigManager &c, std::string s) { THROW_IF_ERROR(c.LoadFile(s)); });
                }));

PYBIND_REGISTER(Tensor, 0, ([](const py::module *m) {
                  (void)py::class_<Tensor, std::shared_ptr<Tensor>>(*m, "Tensor", py::buffer_protocol())
                    .def(py::init([](py::array arr) {
                      std::shared_ptr<Tensor> out;
                      THROW_IF_ERROR(Tensor::CreateFromNpArray(arr, &out));
                      return out;
                    }))
                    .def_buffer([](Tensor &tensor) {
                      py::buffer_info info;
                      THROW_IF_ERROR(Tensor::GetBufferInfo(&tensor, &info));
                      return info;
                    })
                    .def("__str__", &Tensor::ToString)
                    .def("shape", &Tensor::shape)
                    .def("type", &Tensor::type)
                    .def("as_array", [](py::object &t) {
                      auto &tensor = py::cast<Tensor &>(t);
                      if (tensor.type() == DataType::DE_STRING) {
                        py::array res;
                        THROW_IF_ERROR(tensor.GetDataAsNumpyStrings(&res));
                        return res;
                      }
                      py::buffer_info info;
                      THROW_IF_ERROR(Tensor::GetBufferInfo(&tensor, &info));
                      return py::array(pybind11::dtype(info), info.shape, info.strides, info.ptr, t);
                    });
                }));

PYBIND_REGISTER(TensorShape, 0, ([](const py::module *m) {
                  (void)py::class_<TensorShape>(*m, "TensorShape")
                    .def(py::init<py::list>())
                    .def("__str__", &TensorShape::ToString)
                    .def("as_list", &TensorShape::AsPyList)
                    .def("is_known", &TensorShape::known);
                }));

PYBIND_REGISTER(DataType, 0, ([](const py::module *m) {
                  (void)py::class_<DataType>(*m, "DataType")
                    .def(py::init<std::string>())
                    .def(py::self == py::self)
                    .def("__str__", &DataType::ToString)
                    .def("__deepcopy__", [](py::object &t, py::dict memo) { return t; });
                }));

PYBIND_REGISTER(BorderType, 0, ([](const py::module *m) {
                  (void)py::enum_<BorderType>(*m, "BorderType", py::arithmetic())
                    .value("DE_BORDER_CONSTANT", BorderType::kConstant)
                    .value("DE_BORDER_EDGE", BorderType::kEdge)
                    .value("DE_BORDER_REFLECT", BorderType::kReflect)
                    .value("DE_BORDER_SYMMETRIC", BorderType::kSymmetric)
                    .export_values();
                }));

PYBIND_REGISTER(InterpolationMode, 0, ([](const py::module *m) {
                  (void)py::enum_<InterpolationMode>(*m, "InterpolationMode", py::arithmetic())
                    .value("DE_INTER_LINEAR", InterpolationMode::kLinear)
                    .value("DE_INTER_CUBIC", InterpolationMode::kCubic)
                    .value("DE_INTER_AREA", InterpolationMode::kArea)
                    .value("DE_INTER_NEAREST_NEIGHBOUR", InterpolationMode::kNearestNeighbour)
                    .export_values();
                }));

PYBIND_REGISTER(ImageBatchFormat, 0, ([](const py::module *m) {
                  (void)py::enum_<ImageBatchFormat>(*m, "ImageBatchFormat", py::arithmetic())
                    .value("DE_IMAGE_BATCH_FORMAT_NHWC", ImageBatchFormat::kNHWC)
                    .value("DE_IMAGE_BATCH_FORMAT_NCHW", ImageBatchFormat::kNCHW)
                    .export_values();
                }));

}  // namespace dataset
}  // namespace mindspore
