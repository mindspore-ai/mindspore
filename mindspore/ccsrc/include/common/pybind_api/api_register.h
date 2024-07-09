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

#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace mindspore {
void RegTyping(py::module *m);
void RegCNode(const py::module *m);
void RegCell(const py::module *m);
void RegMetaFuncGraph(const py::module *m);
void RegFuncGraph(const py::module *m);
void RegUpdateFuncGraphHyperParams(py::module *m);
void RegParamInfo(const py::module *m);
void RegPrimitive(const py::module *m);
void RegPrimitiveFunction(const py::module *m);
void RegSignatureEnumRW(const py::module *m);
void RegValues(const py::module *m);
void RegMsContext(const py::module *m);
void RegSecurity(py::module *m);
void RegForkUtils(py::module *m);
void RegRandomSeededGenerator(py::module *m);
void RegStress(py::module *m);

namespace hal {
void RegStream(py::module *m);
void RegEvent(py::module *m);
void RegMemory(py::module *m);
}  // namespace hal
namespace initializer {
void RegRandomNormal(py::module *m);
}

namespace pynative {
void RegPyNativeExecutor(const py::module *m);
void RegisterPyBoostFunction(py::module *m);
}  // namespace pynative

namespace pijit {
void RegPIJitInterface(py::module *m);
}

namespace tensor {
void RegMetaTensor(const py::module *m);
void RegCSRTensor(const py::module *m);
void RegCOOTensor(const py::module *m);
void RegRowTensor(const py::module *m);
void RegMapTensor(const py::module *m);
}  // namespace tensor

#ifndef ENABLE_SECURITY
namespace profiler {
void RegProfilerManager(const py::module *m);
void RegProfiler(const py::module *m);
}  // namespace profiler
#endif

namespace prim {
void RegCompositeOpsGroup(const py::module *m);
}
#ifdef _MSC_VER
namespace abstract {
void RegPrimitiveFrontEval();
}
#endif

namespace ops {
void RegOpEnum(py::module *m);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_PYBIND_API_API_REGISTER_H_
