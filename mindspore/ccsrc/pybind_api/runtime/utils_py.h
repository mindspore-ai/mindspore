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

#ifndef MINDSPORE_CCSRC_PYBIND_API_HAL_UTILS_PY_H
#define MINDSPORE_CCSRC_PYBIND_API_HAL_UTILS_PY_H
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ir/tensor.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace py = pybind11;
namespace mindspore {
namespace hal {
device::DeviceContext *GetDeviceCtx();
// Alloc device memory for tensor list
py::object AllocDeviceMemoryForTensorList(const py::object &tensor_list, bool enable_mem_align = True);
py::object GetSliceByTensorListIndexHandle(const py::object &object, const py::object &before_size_obj,
                                           const py::object &after_size_obj, size_t start, size_t end);
py::object GetSliceByPaddingShapeHandle(const py::object &object, size_t start, size_t end);
}  // namespace hal
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_HAL_UTILS_PY_H
