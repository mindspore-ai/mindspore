/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"

#include "minddata/dataset/core/client.h"  // DE client
#include "minddata/dataset/core/global_context.h"

#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
PYBIND_REGISTER(GlobalContext, 0, ([](const py::module *m) {
                  (void)py::class_<GlobalContext>(*m, "GlobalContext")
                    .def_static("config_manager", &GlobalContext::config_manager, py::return_value_policy::reference)
                    .def_static("profiling_manager", &GlobalContext::profiling_manager,
                                py::return_value_policy::reference);
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
                    .def("get_seed", &ConfigManager::seed)
                    .def("set_rank_id", &ConfigManager::set_rank_id)
                    .def("get_rank_id", &ConfigManager::rank_id)
                    .def("get_worker_connector_size", &ConfigManager::worker_connector_size)
                    .def("set_auto_num_workers", &ConfigManager::set_auto_num_workers)
                    .def("set_auto_worker_config", &ConfigManager::set_auto_worker_config_)
                    .def("set_callback_timeout", &ConfigManager::set_callback_timeout)
                    .def("set_monitor_sampling_interval", &ConfigManager::set_monitor_sampling_interval)
                    .def("set_num_parallel_workers",
                         [](ConfigManager &c, int32_t num) { THROW_IF_ERROR(c.set_num_parallel_workers(num)); })
                    .def("set_op_connector_size", &ConfigManager::set_op_connector_size)
                    .def("set_sending_batches", &ConfigManager::set_sending_batches)
                    .def("set_seed", &ConfigManager::set_seed)
                    .def("set_worker_connector_size", &ConfigManager::set_worker_connector_size)
                    .def("set_enable_shared_mem", &ConfigManager::set_enable_shared_mem)
                    .def("get_enable_shared_mem", &ConfigManager::enable_shared_mem)
                    .def("set_auto_offload", &ConfigManager::set_auto_offload)
                    .def("get_auto_offload", &ConfigManager::get_auto_offload)
                    .def("set_enable_autotune",
                         [](ConfigManager &c, bool enable, bool save_autoconfig, std::string json_filepath) {
                           THROW_IF_ERROR(c.set_enable_autotune(enable, save_autoconfig, json_filepath));
                         })
                    .def("get_enable_autotune", &ConfigManager::enable_autotune)
                    .def("set_autotune_interval", &ConfigManager::set_autotune_interval)
                    .def("get_autotune_interval", &ConfigManager::autotune_interval)
                    .def("set_enable_watchdog", &ConfigManager::set_enable_watchdog)
                    .def("get_enable_watchdog", &ConfigManager::enable_watchdog)
                    .def("set_multiprocessing_timeout_interval", &ConfigManager::set_multiprocessing_timeout_interval)
                    .def("get_multiprocessing_timeout_interval", &ConfigManager::multiprocessing_timeout_interval)
                    .def("set_dynamic_shape", &ConfigManager::set_dynamic_shape)
                    .def("get_dynamic_shape", &ConfigManager::dynamic_shape)
                    .def("set_fast_recovery", &ConfigManager::set_fast_recovery)
                    .def("get_fast_recovery", &ConfigManager::fast_recovery)
                    .def("set_debug_mode", &ConfigManager::set_debug_mode)
                    .def("get_debug_mode", &ConfigManager::get_debug_mode)
                    .def("set_error_samples_mode", &ConfigManager::set_error_samples_mode)
                    .def("get_error_samples_mode", &ConfigManager::get_error_samples_mode)
                    .def("load", [](ConfigManager &c, const std::string &s) { THROW_IF_ERROR(c.LoadFile(s)); });
                }));

PYBIND_REGISTER(Tensor, 0, ([](const py::module *m) {
                  (void)py::class_<Tensor, std::shared_ptr<Tensor>>(*m, "Tensor", py::buffer_protocol())
                    .def(py::init([](const py::array &arr) {
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
                    .def("as_python",
                         [](py::object &t) {
                           auto &tensor = py::cast<Tensor &>(t);
                           py::dict res;
                           THROW_IF_ERROR(tensor.GetDataAsPythonObject(&res));
                           return res;
                         })
                    .def("as_array", [](py::object &t) {
                      auto &tensor = py::cast<Tensor &>(t);
                      if (tensor.type().IsString()) {
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
                    .def("is_python", &DataType::IsPython)
                    .def("__str__", &DataType::ToString)
                    .def("__deepcopy__", [](py::object &t, const py::dict &memo) { return t; });
                }));

PYBIND_REGISTER(AutoAugmentPolicy, 0, ([](const py::module *m) {
                  (void)py::enum_<AutoAugmentPolicy>(*m, "AutoAugmentPolicy", py::arithmetic())
                    .value("DE_AUTO_AUGMENT_POLICY_IMAGENET", AutoAugmentPolicy::kImageNet)
                    .value("DE_AUTO_AUGMENT_POLICY_CIFAR10", AutoAugmentPolicy::kCifar10)
                    .value("DE_AUTO_AUGMENT_POLICY_SVHN", AutoAugmentPolicy::kSVHN)
                    .export_values();
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
                    .value("DE_INTER_PILCUBIC", InterpolationMode::kCubicPil)
                    .export_values();
                }));

PYBIND_REGISTER(ImageBatchFormat, 0, ([](const py::module *m) {
                  (void)py::enum_<ImageBatchFormat>(*m, "ImageBatchFormat", py::arithmetic())
                    .value("DE_IMAGE_BATCH_FORMAT_NHWC", ImageBatchFormat::kNHWC)
                    .value("DE_IMAGE_BATCH_FORMAT_NCHW", ImageBatchFormat::kNCHW)
                    .export_values();
                }));

PYBIND_REGISTER(SliceMode, 0, ([](const py::module *m) {
                  (void)py::enum_<SliceMode>(*m, "SliceMode", py::arithmetic())
                    .value("DE_SLICE_PAD", SliceMode::kPad)
                    .value("DE_SLICE_DROP", SliceMode::kDrop)
                    .export_values();
                }));

PYBIND_REGISTER(ConvertMode, 0, ([](const py::module *m) {
                  (void)py::enum_<ConvertMode>(*m, "ConvertMode", py::arithmetic())
                    .value("DE_COLOR_BGR2BGRA", ConvertMode::COLOR_BGR2BGRA)
                    .value("DE_COLOR_RGB2RGBA", ConvertMode::COLOR_RGB2RGBA)
                    .value("DE_COLOR_BGRA2BGR", ConvertMode::COLOR_BGRA2BGR)
                    .value("DE_COLOR_RGBA2RGB", ConvertMode::COLOR_RGBA2RGB)
                    .value("DE_COLOR_BGR2RGBA", ConvertMode::COLOR_BGR2RGBA)
                    .value("DE_COLOR_RGB2BGRA", ConvertMode::COLOR_RGB2BGRA)
                    .value("DE_COLOR_RGBA2BGR", ConvertMode::COLOR_RGBA2BGR)
                    .value("DE_COLOR_BGRA2RGB", ConvertMode::COLOR_BGRA2RGB)
                    .value("DE_COLOR_BGR2RGB", ConvertMode::COLOR_BGR2RGB)
                    .value("DE_COLOR_RGB2BGR", ConvertMode::COLOR_RGB2BGR)
                    .value("DE_COLOR_BGRA2RGBA", ConvertMode::COLOR_BGRA2RGBA)
                    .value("DE_COLOR_RGBA2BGRA", ConvertMode::COLOR_RGBA2BGRA)
                    .value("DE_COLOR_BGR2GRAY", ConvertMode::COLOR_BGR2GRAY)
                    .value("DE_COLOR_RGB2GRAY", ConvertMode::COLOR_RGB2GRAY)
                    .value("DE_COLOR_GRAY2BGR", ConvertMode::COLOR_GRAY2BGR)
                    .value("DE_COLOR_GRAY2RGB", ConvertMode::COLOR_GRAY2RGB)
                    .value("DE_COLOR_GRAY2BGRA", ConvertMode::COLOR_GRAY2BGRA)
                    .value("DE_COLOR_GRAY2RGBA", ConvertMode::COLOR_GRAY2RGBA)
                    .value("DE_COLOR_BGRA2GRAY", ConvertMode::COLOR_BGRA2GRAY)
                    .value("DE_COLOR_RGBA2GRAY", ConvertMode::COLOR_RGBA2GRAY)
                    .export_values();
                }));

PYBIND_REGISTER(ImageReadMode, 0, ([](const py::module *m) {
                  (void)py::enum_<ImageReadMode>(*m, "ImageReadMode", py::arithmetic())
                    .value("DE_IMAGE_READ_MODE_UNCHANGED", ImageReadMode::kUNCHANGED)
                    .value("DE_IMAGE_READ_MODE_GRAYSCALE", ImageReadMode::kGRAYSCALE)
                    .value("DE_IMAGE_READ_MODE_COLOR", ImageReadMode::kCOLOR)
                    .export_values();
                }));

PYBIND_REGISTER(ErrorSamplesMode, 0, ([](const py::module *m) {
                  (void)py::enum_<ErrorSamplesMode>(*m, "ErrorSamplesMode", py::arithmetic())
                    .value("DE_ERROR_SAMPLES_MODE_RETURN", ErrorSamplesMode::kReturn)
                    .value("DE_ERROR_SAMPLES_MODE_REPLACE", ErrorSamplesMode::kReplace)
                    .value("DE_ERROR_SAMPLES_MODE_SKIP", ErrorSamplesMode::kSkip)
                    .export_values();
                }));
}  // namespace dataset
}  // namespace mindspore
