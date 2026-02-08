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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_TRACE_RECORDER_INTERFACE_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_TRACE_RECORDER_INTERFACE_H_

#include <memory>
#include <map>
#include <stack>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "pybind11/pybind11.h"
#include "include/frontend/operator/primitive_py.h"
#include "include/utils/visible.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "frontend/jit/ps/parse/resolve.h"

namespace mindspore {
namespace trace {
FRONTEND_EXPORT void Capture(const py::args &args, py::object *res);
FRONTEND_EXPORT void Capture(const py::list &args, const PrimitivePtr &prim, py::object *res);
FRONTEND_EXPORT void Capture(const std::vector<py::object> &args_vec, const PrimitivePtr &prim, py::object *res);
FRONTEND_EXPORT void CapturePy(PyObject *args, PyObject **res);
FRONTEND_EXPORT void CapturePy(PyObject *args, const PrimitivePtr &prim, PyObject **res);
FRONTEND_EXPORT void CapturePy(const std::vector<PyObject *> &args_vec, const PrimitivePtr &prim, PyObject **res);
FRONTEND_EXPORT void CaptureResolveOperation(const py::tuple &args, const std::string &named_primitive,
                                             py::object *res);
FRONTEND_EXPORT bool IsTracing();
FRONTEND_EXPORT void PassNodeInArg(PyObject *origin_obj, PyObject *new_obj);
}  // namespace trace
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_TRACE_RECORDER_INTERFACE_H_
