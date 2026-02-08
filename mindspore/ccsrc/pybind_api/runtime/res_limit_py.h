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

#ifndef MINDSPORE_CCSRC_PYBIND_API_HAL_RES_LIMIT_PY_H
#define MINDSPORE_CCSRC_PYBIND_API_HAL_RES_LIMIT_PY_H
#include "pybind11/pybind11.h"
#include "pybind_api/runtime/stream_py.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace hal {
namespace py = pybind11;
py::tuple GetDeviceLimit(int32_t device_id);
void SetDeviceLimit(int32_t device_id, int32_t cube_num, int32_t vector_num);
py::tuple GetStreamLimit(const StreamPyPtr &stream);
void SetStreamLimit(const StreamPyPtr &stream, int32_t cube_num, int32_t vector_num);
void ResetStreamLimit(const StreamPyPtr &stream);
void DispatchSetStreamLimitTask(const StreamPyPtr &stream, int32_t cube_num, int32_t vector_num);
}  // namespace hal
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_HAL_RES_LIMIT_PY_H
