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
#include "pipeline/jit/pi/graph_guard/trace.h"

namespace mindspore {
namespace pijit {
using CheckFunc = bool (*)(const py::object &);
using InferFunc = bool (*)(CallNode *);
struct SpecialAction {
  CheckFunc check;
  InferFunc infer;
};

const char *GetFuncName(const py::object &f);
bool CheckPrimitive(const py::object &func);
void HandleGradFuncCall(CallNode *call_node, AObject *decorated, bool sens_param);
bool GuardConstCallNodeParam(CallNode *call_node, Graph *sub_graph, int max_guard_depth);
bool JustCallAndSetRes(CallNode *call_node);
const std::unordered_map<std::string, SpecialAction> &GetFuncWhiteListMap(bool trace_flag = false);
const std::vector<std::pair<CheckFunc, std::string>> &GetFuncWhiteListFuzzyMatcher(bool trace_flag = false);
const std::string GetMindsporeNamePrimitive();

bool InferListAppend(CallNode *call_node);

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_INLINE_CHECK_H
