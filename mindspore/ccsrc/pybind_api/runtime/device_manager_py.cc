/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <memory>
#include "utils/device_manager_conf.h"
#include "pybind_api/runtime/runtime_api.h"

namespace mindspore {
void RegDeviceManagerConf(const py::module *m) {
  (void)py::class_<DeviceManagerConf, std::shared_ptr<DeviceManagerConf>>(*m, "DeviceManagerConf")
    .def_static("get_instance", &DeviceManagerConf::GetInstance, "Get DeviceManagerConf instance.")
    .def("set_device", &DeviceManagerConf::set_device, "Set device target and device id.")
    .def("get_device_target", &DeviceManagerConf::GetDeviceTarget, "Get device target.")
    .def("get_device_id", &DeviceManagerConf::device_id, "Get device id.")
    .def("is_device_enable", &DeviceManagerConf::IsDeviceEnable, "Is device enable.")
    .def("set_deterministic", &DeviceManagerConf::set_deterministic, "Set deterministic.")
    .def("is_deterministic_configured", &DeviceManagerConf::IsDeterministicConfigured, "Set deterministic.");
}
}  // namespace mindspore
