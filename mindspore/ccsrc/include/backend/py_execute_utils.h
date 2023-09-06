/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_PY_EXECUTE_UTILS_H
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_PY_EXECUTE_UTILS_H

#include "include/backend/device_address.h"
#include "include/common/utils/python_adapter.h"

namespace mindspore {
namespace pyexecute {
using DeviceAddress = device::DeviceAddress;
using PyDataConverter = bool (*)(const py::object &, ValuePtr *);
void set_pydata_converter(const PyDataConverter &set_pydata_converter);
abstract::AbstractBasePtr GenerateAbstractFromPyObject(const py::object &obj);
void UserDataToRawMemory(DeviceAddress *const device_address);
}  // namespace pyexecute
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_PY_EXECUTE_UTILS_H
