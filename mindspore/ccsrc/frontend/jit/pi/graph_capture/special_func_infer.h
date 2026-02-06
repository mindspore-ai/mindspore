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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_INLINE_CHECK_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_INLINE_CHECK_H

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "frontend/jit/pi/graph_capture/node.h"
#include "frontend/jit/pi/graph_capture/graph_build.h"

namespace mindspore {
namespace pijit {
using InferFunc = bool (*)(CallNode *, GraphBuilder *);
InferFunc FindInferFunc(const py::object &callable);

bool JustCallAndSetRes(CallNode *call_node, GraphBuilder *g = nullptr);
bool JustCallAndSetResWithArgs(CallNode *call_node, const std::vector<py::object> &args, GraphBuilder *g = nullptr);

// check a variable is not referenced by other object
bool IsReferencedVariable(ValueNode *);

bool CheckJitConstexpr(const py::object &func);
bool CheckMSConstexpr(const py::object &func);
bool CheckBuiltinFuncOrMethod(const py::object &func);
bool IsPSJitFunction(const py::object &callable_info);

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_INLINE_CHECK_H
