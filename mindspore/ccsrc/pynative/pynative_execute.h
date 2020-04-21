/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PYNATIVE_PYNATIVE_EXECUTE_H_
#define MINDSPORE_CCSRC_PYNATIVE_PYNATIVE_EXECUTE_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <unordered_map>

#include "pybind11/pybind11.h"

#include "pynative/base.h"
#include "utils/context/ms_context.h"

namespace mindspore {
namespace pynative {

namespace py = pybind11;

py::object RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status);

py::tuple RunOp(const py::args &args);
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYNATIVE_PYNATIVE_EXECUTE_H_
