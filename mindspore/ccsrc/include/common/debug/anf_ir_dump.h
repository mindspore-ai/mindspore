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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ANF_IR_DUMP_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ANF_IR_DUMP_H_

#include <string>
#include <memory>
#include "ir/dtype/type.h"
#include "ir/anf.h"
#include "include/common/debug/common.h"
#include "utils/hash_set.h"
#include "include/common/visible.h"

namespace mindspore {
enum LocDumpMode : int { kOff = 0, kTopStack = 1, kWholeStack = 2, kInValid = 3 };
auto constexpr kDumpConfigLineLevel0 = "LINE_LEVEL0";
auto constexpr kDumpConfigLineLevel1 = "LINE_LEVEL1";
auto constexpr kDumpConfigLineLevel2 = "LINE_LEVEL2";
auto constexpr kDumpConfigDisableBackend = "DISABLE_BACKEND";
auto constexpr kDumpConfigEnablePassIR = "ENABLE_PASS_IR";
struct DumpConfig {
  LocDumpMode dump_line_level = kInValid;
  bool disable_backend_dump = false;
  bool enable_dump_pass_ir = false;
};

struct SubGraphIRInfo {
  int32_t local_var;
  std::ostringstream buffer;
  OrderedMap<AnfNodePtr, int32_t> local_var_map;
};
COMMON_EXPORT void DumpCNode(const CNodePtr &node, const FuncGraphPtr &sub_graph,
                             const OrderedMap<AnfNodePtr, int32_t> &para_map,
                             const std::shared_ptr<SubGraphIRInfo> &gsub, bool dump_full_name = false,
                             LocDumpMode dump_location = kOff);
COMMON_EXPORT int32_t DumpParams(const FuncGraphPtr &graph, std::ostringstream &buffer,
                                 OrderedMap<AnfNodePtr, int32_t> *para_map);
COMMON_EXPORT void OutputOrderList(const FuncGraphPtr &sub_graph, std::ostringstream &oss);
constexpr char PARALLEL_STRATEGY[] = "strategy";
COMMON_EXPORT void DumpIR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name = false,
                          LocDumpMode dump_location = kOff, const std::string &target_file = "");
COMMON_EXPORT void DumpIR(std::ostringstream &graph_buffer, const FuncGraphPtr &graph, bool dump_full_name = false,
                          LocDumpMode dump_location = kOff);

COMMON_EXPORT void GatherInputAndOutputInferType(std::ostringstream &buffer, const AnfNodePtr &node);

COMMON_EXPORT void DumpIRForRDR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name = false,
                                LocDumpMode dump_location = kOff);
COMMON_EXPORT DumpConfig GetDumpConfig();
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ANF_IR_DUMP_H_
