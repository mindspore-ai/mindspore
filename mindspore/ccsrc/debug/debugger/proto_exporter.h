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
#ifndef MINDSPORE_PROTO_EXPORTER_H
#define MINDSPORE_PROTO_EXPORTER_H

#include <map>
#include <string>
#include <vector>

#include "include/common/debug/common.h"
#include "debug/data_dump/dump_json_parser.h"
#include "ir/graph_utils.h"
#include "proto/debug_graph.pb.h"
#include "utils/symbolic.h"
#include "utils/trace_base.h"

namespace mindspore {
enum LocDebugDumpMode { kDebugOff = 0, kDebugTopStack = 1, kDebugWholeStack = 2 };
class DebuggerProtoExporter {
 public:
  DebuggerProtoExporter() {}
  ~DebuggerProtoExporter() {}

  std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph,
                                      LocDebugDumpMode dump_location = kDebugWholeStack);
  debugger::ModelProto GetFuncGraphProto(const FuncGraphPtr &func_graph);

 private:
  void InitModelInfo();
  void GetOpNodeTypeAndAttrs(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                             debugger::NodeProto *node_proto) const;
  std::string GetOpNodeInputId(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                               const std::map<AnfNodePtr, size_t> &apply_map,
                               std::map<AnfNodePtr, size_t> *const_map_ptr) const;
  void SetValueToProto(const ValuePtr &attr_value, debugger::ValueProto *value_proto) const;
  void SetScalarToProto(const ScalarPtr &val, debugger::ValueProto *value_proto) const;
  void SetSequenceToProto(const ValueSequencePtr &val, debugger::ValueProto *value_proto) const;
  void SetDictionaryToProto(const ValueDictionaryPtr &val, debugger::ValueProto *value_proto) const;
  void SetNodeOutputType(const AnfNodePtr &node, debugger::TypeProto *type_proto) const;
  void ExportFuncGraph(const FuncGraphPtr &func_graph, debugger::GraphProto *const graph_proto,
                       LocDebugDumpMode dump_location = kDebugWholeStack);
  void ExportParameters(const FuncGraphPtr &func_graph, debugger::GraphProto *graph_proto) const;
  void ExportCNodes(const FuncGraphPtr &func_graph, debugger::GraphProto *const graph_proto,
                    std::map<AnfNodePtr, size_t> *const_map_ptr, LocDebugDumpMode dump_location = kDebugWholeStack);
  void ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *apply_map_ptr,
                   std::map<AnfNodePtr, size_t> *const_map_ptr, debugger::GraphProto *const graph_proto,
                   LocDebugDumpMode dump_location);
  void ExportFuncGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &ret_node,
                             const std::map<AnfNodePtr, size_t> &apply_map, std::map<AnfNodePtr, size_t> *const_map_ptr,
                             debugger::GraphProto *graph_proto) const;
  void ExportValueNodes(const std::map<AnfNodePtr, size_t> &const_map, debugger::GraphProto *graph_proto) const;

  static std::string GetConstNodeId(size_t idx) { return std::string("cst") + std::to_string(idx); }

  debugger::ModelProto model_;
};
BACKEND_EXPORT void DumpIRProtoWithSrcInfo(const FuncGraphPtr &func_graph, const std::string &suffix,
                                           const std::string &target_dir,
                                           LocDebugDumpMode dump_location = kDebugWholeStack);
void DumpConstantInfo(const KernelGraphPtr &graph, const std::string &target_dir);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_DEBUGGER_MINDSPORE_PROTO_EXPORTER_H_
