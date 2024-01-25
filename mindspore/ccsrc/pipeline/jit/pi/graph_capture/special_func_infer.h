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
#include "pipeline/jit/pi/graph_capture/node.h"
#include "pipeline/jit/pi/graph_guard/trace.h"

namespace mindspore {
namespace pijit {
// check the function is special function that mindspore support and not inline,
// the return values or type can be infer
// set key for handler
bool IsFuncInWhiteList(const py::object &, std::string *key, bool bInferPrimitive);

// infer the return value of special function and generate subgraph, or clear subgraph
// return true if special function has subgraph
bool HandleFuncInWhiteList(const std::string &, CallNode *);

void HandleGradFuncCall(CallNode *call_node, AObject *decorated, bool sens_param);

bool GuardConstCallNodeParam(CallNode *call_node, Graph *sub_graph, int max_guard_depth);
bool JustCallAndSetRes(CallNode *call_node);

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_INLINE_CHECK_H
