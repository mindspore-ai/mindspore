/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEBUGGER_PROTO_EXPORTER_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEBUGGER_PROTO_EXPORTER_H_

#include <string>

#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "tools/visible.h"
#include "utils/symbolic.h"

namespace mindspore {
enum LocDebugDumpMode { kDebugOff = 0, kDebugTopStack = 1, kDebugWholeStack = 2 };
TOOLS_EXPORT void DumpIRProtoWithSrcInfo(const FuncGraphPtr &func_graph, const std::string &suffix,
                                         const std::string &target_dir,
                                         LocDebugDumpMode dump_location = kDebugWholeStack);
TOOLS_EXPORT void DumpIRProtoWithSrcInfoDebugWholeStack(const FuncGraphPtr &func_graph, const std::string &suffix,
                                                        const std::string &target_dir);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEBUGGER_PROTO_EXPORTER_H_
