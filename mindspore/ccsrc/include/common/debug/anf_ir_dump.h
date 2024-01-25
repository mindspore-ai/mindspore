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
#include <vector>
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
  int32_t format_level;
};
COMMON_EXPORT void DumpCNode(const CNodePtr &node, const FuncGraphPtr &sub_graph,
                             const OrderedMap<AnfNodePtr, int32_t> &para_map,
                             const std::shared_ptr<SubGraphIRInfo> &gsub, bool dump_full_name = false,
                             LocDumpMode dump_location = kOff);
COMMON_EXPORT int32_t DumpParams(const FuncGraphPtr &graph, std::ostringstream &buffer,
                                 OrderedMap<AnfNodePtr, int32_t> *para_map);
COMMON_EXPORT void OutputOrderList(const FuncGraphPtr &sub_graph, std::ostringstream &oss);
constexpr char PARALLEL_STRATEGY[] = "strategy";
COMMON_EXPORT void DumpIRHead(const FuncGraphPtr &graph, std::ostringstream &buffer);
COMMON_EXPORT void DumpIR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name = false,
                          LocDumpMode dump_location = kOff, const std::string &target_file = "");
COMMON_EXPORT void DumpIR(std::ostringstream &graph_buffer, const FuncGraphPtr &graph, bool dump_full_name = false,
                          LocDumpMode dump_location = kOff);

COMMON_EXPORT void GatherInputAndOutputInferType(std::ostringstream &buffer, const AnfNodePtr &node);

COMMON_EXPORT void DumpIRForRDR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name = false,
                                LocDumpMode dump_location = kOff);
COMMON_EXPORT DumpConfig GetDumpConfig();
std::string GetValueText(const ValuePtr &value, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpOperator(const AnfNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpOperands(const AnfNodePtr &node, const OrderedMap<AnfNodePtr, int32_t> &para_map,
                                const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpOperateAttrs(const AnfNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpCNodeAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpCNodePrimalAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT void DumpParallelInfo(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub);

COMMON_EXPORT int32_t DumpParams(const FuncGraphPtr &graph, std::ostringstream &buffer,
                                 OrderedMap<AnfNodePtr, int32_t> *para_map);

COMMON_EXPORT void DumpIRInSubgraph(const std::vector<AnfNodePtr> &nodes, OrderedMap<AnfNodePtr, int32_t> *para_map,
                                    OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *const sub_graphs,
                                    int32_t total_para, bool dump_full_name = false, LocDumpMode dump_location = kOff);

COMMON_EXPORT void DumpGlobalInfoEntry(const FuncGraphPtr &graph, std::ostringstream &buffer, size_t sub_graphs_size);

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

class AnfExporter {
 public:
  explicit AnfExporter(bool export_used = true, bool check_integrity = false)
      : param_index_(1), export_used_(export_used), check_integrity_(check_integrity) {
    func_graph_set_.clear();
    exported_.clear();
  }
  virtual ~AnfExporter() {}

  void ExportFuncGraph(const std::string &filename, const FuncGraphPtr &func_graph);

 protected:
  void OuputIrStyleCNodes(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &nodes, int32_t total_para,
                          std::ostringstream &oss, OrderedMap<AnfNodePtr, int32_t> *para_map);

  virtual void ExportOneFuncGraph(const FuncGraphPtr &func_graph, const TaggedNodeMap &tagged_cnodes_map,
                                  std::ostringstream &oss, int32_t total_para = 0,
                                  OrderedMap<AnfNodePtr, int32_t> *para_map = nullptr);

  OrderedMap<FuncGraphPtr, ParamIndexMap> exported_;
  bool is_top_graph_;

 private:
  int param_index_;
  OrderedSet<FuncGraphPtr> func_graph_set_{};
  bool export_used_ = true;       // whether export function graphs used in current exporting function graph
  bool check_integrity_ = false;  // whether check integrity or not, when dumping ir for loading, must set it to true
};

COMMON_EXPORT void ExportIR(const std::string &filename, const FuncGraphPtr &func_graph);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ANF_IR_DUMP_H_
