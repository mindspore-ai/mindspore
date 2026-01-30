/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_UTILS_IR_DUMP_ANF_IR_DUMP_INTERFACE_H_
#define MINDSPORE_CCSRC_UTILS_IR_DUMP_ANF_IR_DUMP_INTERFACE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/utils/common.h"
#include "include/utils/visible.h"
#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "utils/ordered_map.h"

namespace mindspore {
enum LocDumpMode : int { kOff = 0, kTopStack = 1, kWholeStack = 2, kInValid = 3 };

struct ParamPtrEqual {
  bool operator()(AnfNodePtr const &t1, AnfNodePtr const &t2) const {
    const ParameterPtr param1 = dyn_cast<Parameter>(t1);
    const ParameterPtr param2 = dyn_cast<Parameter>(t2);

    if (param1 == nullptr || param2 == nullptr) {
      return false;
    }

    return *param1 == *param2;
  }
};

struct ParamPtrHasher {
  std::size_t operator()(AnfNodePtr const &param) const {
    const ParameterPtr parameter = dyn_cast<Parameter>(param);
    if (parameter == nullptr) {
      return 0;
    }
    std::size_t hash = std::hash<std::string>()(parameter->name());
    return hash;
  }
};

using ParamIndexMap = OrderedMap<AnfNodePtr, int, ParamPtrHasher, ParamPtrEqual, true>;

struct SubGraphIRInfo {
  int32_t local_var;
  std::ostringstream buffer;
  OrderedMap<AnfNodePtr, int32_t> local_var_map;
  int32_t format_level;
  int32_t cnode_num = 0;
};

struct DumpConfig {
  LocDumpMode dump_line_level = kInValid;
  bool disable_backend_dump = false;
  bool enable_dump_pass_ir = false;
};

COMMON_EXPORT int32_t DumpParams(const FuncGraphPtr &graph, std::ostringstream &buffer,
                                 OrderedMap<AnfNodePtr, int32_t> *para_map);
COMMON_EXPORT void OutputOrderList(const FuncGraphPtr &sub_graph, std::ostringstream &oss);

COMMON_EXPORT void DumpIR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name = false,
                          LocDumpMode dump_location = kWholeStack, const std::string &target_file = "");
COMMON_EXPORT void DumpIR(std::ostringstream &graph_buffer, const FuncGraphPtr &graph, bool dump_full_name = false,
                          int dump_location = kWholeStack, bool avoid_circle = false);
COMMON_EXPORT void DumpParallelJson(const std::string &filename, const FuncGraphPtr &graph,
                                    const int64_t global_rank_id,
                                    const std::unordered_map<std::string, std::vector<uint32_t>> &group_map);
COMMON_EXPORT DumpConfig GetDumpConfig();

COMMON_EXPORT void DumpOperator(const AnfNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpOperands(const AnfNodePtr &node, const OrderedMap<AnfNodePtr, int32_t> &para_map,
                                const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpOperateAttrs(const AnfNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpCNodeAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpCNodePrimalAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpParallelInfo(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT int32_t DumpParams(const FuncGraphPtr &graph, std::ostringstream &buffer,
                                 OrderedMap<AnfNodePtr, int32_t> *para_map);

COMMON_EXPORT void DumpGlobalInfoEntry(const FuncGraphPtr &graph, std::ostringstream &buffer, size_t sub_graphs_size);

COMMON_EXPORT void DumpCommentsInCurrentScope(const DebugInfoPtr &debug_info, std::string &result);

COMMON_EXPORT void GetSubgraphAttrAsString(const FuncGraphPtr &func_graph, std::ostringstream &oss);

COMMON_EXPORT void ExportIR(const std::string &filename, const FuncGraphPtr &func_graph);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_IR_DUMP_ANF_IR_DUMP_INTERFACE_H_
