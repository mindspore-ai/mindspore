/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_UTILS_TRACE_BASE_H_
#define MINDSPORE_CORE_UTILS_TRACE_BASE_H_

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <stack>

#include "utils/info.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/any.h"

namespace mindspore {
namespace trace {
std::string GetDebugInfo(const DebugInfoPtr &info, SourceLineTip tip = kSourceLineTipNextLine);
std::string GetDebugInfo(const DebugInfoPtr &info, const std::string &prefix,
                         SourceLineTip tip = kSourceLineTipNextLine);
// Generate the call stack of python source code to a string
std::string DumpSourceLines(const AnfNodePtr &node);
std::string DumpSourceLines(AnfNode *node);
// Generate the call stack of python source code to a vector
std::vector<std::string> GetSourceLineList(const AnfNodePtr &node);
// Get the locations of the call stack of python source code
std::vector<LocationPtr> GetSourceLocationList(const AnfNodePtr &node);
// Generate the call stack of python source code with relevant trace info
std::string GetDebugTraceInfo(const AnfNodePtr &node, bool is_debug = false);
}  // namespace trace
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_TRACE_BASE_H_
