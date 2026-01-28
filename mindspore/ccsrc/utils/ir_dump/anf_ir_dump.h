/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_UTILS_IR_DUMP_ANF_IR_DUMP_H_
#define MINDSPORE_CCSRC_UTILS_IR_DUMP_ANF_IR_DUMP_H_

#include "include/utils/ir_dump/anf_ir_dump_interface.h"

namespace mindspore {
auto constexpr kDumpConfigLineLevel0 = "LINE_LEVEL0";
auto constexpr kDumpConfigLineLevel1 = "LINE_LEVEL1";
auto constexpr kDumpConfigLineLevel2 = "LINE_LEVEL2";
auto constexpr kDumpConfigDisableBackend = "DISABLE_BACKEND";
auto constexpr kDumpConfigEnablePassIR = "ENABLE_PASS_IR";
void DumpCNode(const CNodePtr &node, const FuncGraphPtr &sub_graph, const OrderedMap<AnfNodePtr, int32_t> &para_map,
               const std::shared_ptr<SubGraphIRInfo> &gsub, bool dump_full_name = false,
               LocDumpMode dump_location = kWholeStack);
constexpr char PARALLEL_STRATEGY[] = "strategy";
void SaveIRFile(const std::string &filename, const std::string &str, const std::string &target_file);

void GetValueText(const ValuePtr &value, const std::shared_ptr<SubGraphIRInfo> &gsub);

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
  bool is_top_graph_{false};

 private:
  int param_index_;
  OrderedSet<FuncGraphPtr> func_graph_set_{};
  bool export_used_ = true;       // whether export function graphs used in current exporting function graph
  bool check_integrity_ = false;  // whether check integrity or not, when dumping ir for loading, must set it to true
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_IR_DUMP_ANF_IR_DUMP_H_
