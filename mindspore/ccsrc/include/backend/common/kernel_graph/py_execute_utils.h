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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_KERNEL_GRAPH_PY_EXECUTE_UTILS_H
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_KERNEL_GRAPH_PY_EXECUTE_UTILS_H

#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "include/utils/python_adapter.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace pyexecute {
using DeviceAddress = device::DeviceAddress;
using KernelTensor = kernel::KernelTensor;
using PyDataConverter = bool (*)(const py::object &, ValuePtr *);
BACKEND_COMMON_EXPORT void set_pydata_converter(const PyDataConverter &set_pydata_converter);
BACKEND_COMMON_EXPORT tensor::TensorPtr GetValueByPyObj(const py::object &obj);
BACKEND_COMMON_EXPORT abstract::AbstractBasePtr GenerateAbstractFromPyObject(const py::object &obj);
void UserDataToRawMemory(KernelTensor *const kernel_tensor);
ValuePtr GetValueFromUserData(const UserDataPtr &user_data);
}  // namespace pyexecute
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_KERNEL_GRAPH_PY_EXECUTE_UTILS_H
