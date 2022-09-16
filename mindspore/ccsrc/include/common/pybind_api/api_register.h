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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_PYBIND_API_API_REGISTER_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_PYBIND_API_API_REGISTER_H_

#include <map>
#include <string>
#include <memory>
#include <functional>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace mindspore {
void RegTyping(py::module *m);
void RegCNode(py::module *m);
void RegCell(py::module *m);
void RegMetaFuncGraph(py::module *m);
void RegFuncGraph(py::module *m);
void RegUpdateFuncGraphHyperParams(py::module *m);
void RegParamInfo(py::module *m);
void RegPrimitive(py::module *m);
void RegSignatureEnumRW(py::module *m);
void RegValues(py::module *m);
void RegMsContext(py::module *m);
void RegSecurity(py::module *m);

namespace initializer {
void RegRandomNormal(py::module *m);
}

namespace pynative {
void RegPynativeExecutor(py::module *m);
}

namespace tensor {
void RegMetaTensor(py::module *m);
void RegCSRTensor(py::module *m);
void RegCOOTensor(py::module *m);
void RegRowTensor(py::module *m);
void RegMapTensor(py::module *m);
}  // namespace tensor

namespace opt {
namespace python_pass {
void RegPattern(py::module *m);
void RegPyPassManager(py::module *m);
}  // namespace python_pass
}  // namespace opt

#ifndef ENABLE_SECURITY
namespace profiler {
void RegProfilerManager(py::module *m);
void RegProfiler(py::module *m);
}  // namespace profiler
#endif

namespace prim {
void RegCompositeOpsGroup(py::module *m);
}
#ifdef _MSC_VER
namespace abstract {
void RegPrimitiveFrontEval();
}
#endif
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_PYBIND_API_API_REGISTER_H_
