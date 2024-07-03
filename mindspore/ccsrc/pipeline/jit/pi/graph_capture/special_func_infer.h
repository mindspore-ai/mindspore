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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_INLINE_CHECK_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_INLINE_CHECK_H

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "pipeline/jit/pi/graph_capture/node.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"

namespace mindspore {
namespace pijit {

using InferFunc = bool (*)(CallNode *, GraphBuilder *);
InferFunc FindInferFunc(const py::object &callable, bool trace_flag = false);

void HandleGradFuncCall(CallNode *call_node, AObject *decorated, bool sens_param,
                        const py::object &after_grad = py::object());
bool GuardConstCallNodeParam(CallNode *call_node, Graph *sub_graph, int max_guard_depth);
bool JustCallAndSetRes(CallNode *call_node, GraphBuilder *g = nullptr);
bool JustCallAndSetResWithArgs(CallNode *call_node, const std::vector<py::object> &args, GraphBuilder *g = nullptr);

bool CheckJitConstexpr(const py::object &func);
bool CheckMSConstexpr(const py::object &func);
bool CheckBuiltinFuncOrMethod(const py::object &func);

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_INLINE_CHECK_H
