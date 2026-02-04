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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_PYBIND_API_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_PYBIND_API_H_

#include "pybind11/pybind11.h"
#include "include/utils/visible.h"

namespace py = pybind11;
namespace mindspore {
namespace pijit {
FRONTEND_EXPORT void RegPIJitInterface(py::module *m);
}

namespace prim {
FRONTEND_EXPORT void RegCompositeOpsGroup(const py::module *m);
}

#ifdef _MSC_VER
namespace abstract {
FRONTEND_EXPORT void RegPrimitiveFrontEval();
}
#endif

namespace trace {
FRONTEND_EXPORT void RegTraceRecorderPy(const py::module *m);
}
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_FRONTEND_FRONTEND_API_H_
