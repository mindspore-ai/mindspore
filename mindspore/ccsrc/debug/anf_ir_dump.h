/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_ANF_IR_DUMP_H_
#define MINDSPORE_CCSRC_DEBUG_ANF_IR_DUMP_H_

#include <string>
#include <vector>
#include "ir/dtype/type.h"
#include "ir/anf.h"
#include "debug/common.h"
#include "utils/hash_set.h"

namespace mindspore {
enum LocDumpMode : int { kOff = 0, kTopStack = 1, kWholeStack = 2, kInValid = 3 };
auto constexpr kDumpConfigLineLevel0 = "LINE_LEVEL0";
auto constexpr kDumpConfigLineLevel1 = "LINE_LEVEL1";
auto constexpr kDumpConfigLineLevel2 = "LINE_LEVEL2";
auto constexpr kDumpConfigDisableBackend = "DISABLE_BACKEND";
auto constexpr kDumpConfigEnablePassIR = "ENABLE_PASS_IR";
typedef struct DumpConfig {
  LocDumpMode dump_line_level = kInValid;
  bool disable_backend_dump = false;
  bool enable_dump_pass_ir = false;
} DumpConfig;

constexpr char PARALLEL_STRATEGY[] = "strategy";
void DumpIR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name = false,
            LocDumpMode dump_location = kOff, const std::string &target_file = "");
void DumpIR(std::ostringstream &graph_buffer, const FuncGraphPtr &graph, bool dump_full_name = false,
            LocDumpMode dump_location = kOff);

void GatherInputAndOutputInferType(std::ostringstream &buffer, const AnfNodePtr &node);

void DumpIRForRDR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name = false,
                  LocDumpMode dump_location = kOff);
const std::string ToShortString(const TypeId &typeId);
DumpConfig GetDumpConfig();
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_ANF_IR_DUMP_H_
