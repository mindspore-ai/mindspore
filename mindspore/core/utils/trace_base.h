/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "utils/info.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/any.h"

namespace mindspore {
namespace trace {
constexpr auto kSectionPrefix = " - ";

MS_CORE_API DebugInfoPtr GetSourceCodeDebugInfo(const DebugInfoPtr &info);
MS_CORE_API std::string GetDebugInfo(const DebugInfoPtr &info, SourceLineTip tip = kSourceLineTipNextLine);
MS_CORE_API std::string GetDebugInfo(const DebugInfoPtr &info, const std::string &prefix,
                                     SourceLineTip tip = kSourceLineTipNextLine);
// Generate the call stack of python source code to a string
MS_CORE_API std::string DumpSourceLines(const AnfNodePtr &node, bool has_title = true);
MS_CORE_API std::string DumpSourceLines(AnfNode *node, bool has_title = true);
// Generate the call stack of python source code to a vector
MS_CORE_API std::vector<std::string> GetSourceLineList(const AnfNodePtr &node);
MS_CORE_API std::vector<std::string> GetSourceLineList(const DebugInfoPtr &debug_info);
// Get the locations of the call stack of python source code
std::vector<LocationPtr> GetSourceLocationList(const AnfNodePtr &node);
// Generate the call stack of python source code with relevant trace info
std::string GetDebugTraceInfo(const AnfNodePtr &node, bool is_debug = false);
}  // namespace trace
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_TRACE_BASE_H_
