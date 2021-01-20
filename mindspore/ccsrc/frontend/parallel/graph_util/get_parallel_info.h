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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GET_GRAPH_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GET_GRAPH_INFO_H_

#include "pybind11/stl.h"
#include "pybind11/pybind11.h"
#include "ir/anf.h"

namespace py = pybind11;
namespace mindspore {
namespace parallel {
py::dict GetParameterLayout(const FuncGraphPtr &graph);
py::dict GetAllreduceFusion(const FuncGraphPtr &graph);
py::list GetParallelParameterNameList(const FuncGraphPtr &graph);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GET_GRAPH_INFO_H_
