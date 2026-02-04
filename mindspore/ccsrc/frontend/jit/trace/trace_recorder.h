/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_TRACE_RECORDER_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_TRACE_RECORDER_H_

#include <memory>
#include <map>
#include <stack>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "pybind11/pybind11.h"
#include "include/frontend/operator/primitive_py.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "frontend/jit/ps/parse/resolve.h"

namespace mindspore {
namespace trace {
py::object CaptureRun(const py::args &args, const py::object &res, const py::object &prim_py);
py::object DefaultOutput();
bool Compiled();

class TraceRecorder {
 public:
  TraceRecorder() = default;
  ~TraceRecorder() = default;
  TraceRecorder(const TraceRecorder &) = delete;
  TraceRecorder(TraceRecorder &&) = delete;
  TraceRecorder &operator=(const TraceRecorder &) = delete;
  TraceRecorder &operator=(TraceRecorder &&) = delete;

  static std::shared_ptr<TraceRecorder> GetInstance() {
    static auto trace_recorder = std::make_shared<TraceRecorder>();
    return trace_recorder;
  }

  FuncGraphPtr InitTopGraph(const DebugInfoPtr &debug_info);
  void BeginGraph(const py::object &func_name, const py::object &phase, const py::list &file_names,
                  const py::list &linenos, const py::args &args);
  void EndGraph(const py::list &file_names, const py::list &linenos, const py::dict &jit_config,
                const py::args &output_args);
  void NewFuncGraphNode(const py::tuple &info, const py::args &inputs);
  void NewNode(const py::object &prim_obj, const py::tuple &op_info, const py::args &inputs);
  void ProcessNewNode(const PrimitivePtr &prim, const py::object &prim_res, const DebugInfoPtr &debug_info,
                      const py::args &inputs, bool do_signature);
  void ProcessNewResolveNode(const parse::NameSpacePtr &name_space, const parse::SymbolPtr &resolve_symbol,
                             const py::object &prim_res, const DebugInfoPtr debug_info, const py::args &inputs,
                             bool do_signature);
  std::pair<AnfNodePtrList, AbstractBasePtrList> GenerateInputs(const py::args &inputs, const DebugInfoPtr &debug_info);
  py::object RunGraph(const py::object &phase, const py::dict &jit_config, const py::tuple &args);

  void SyncTensorNode(const py::object &old_tensor_obj, const py::object &new_tensor_obj);
  bool BuildingTraceGraph() { return !graph_stack_.empty(); }
  FuncGraphPtr BuildEndGraph(const py::list &file_names, const py::list &linenos, const py::args &output_args,
                             bool nested = false);
  py::object InitTraceGraphInputs(const AbstractBasePtr &abs, const AnfNodePtr &param);
  void PassNode(const py::object &origin_obj, const py::object &new_obj);

 private:
  AnfNodePtr GetNode(const py::object &obj, const DebugInfoPtr &debug_info, bool set_abstract = false);
  AnfNodePtr GetTensorNode(const py::object &tensor_obj, const DebugInfoPtr &debug_info, bool set_abstract);
  AnfNodePtr GetTupleNode(const py::tuple &tuple_obj, const DebugInfoPtr &debug_info, bool set_abstract);
  AnfNodePtr GetListNode(const py::list &list_obj, const DebugInfoPtr &debug_info, bool set_abstract);
  AnfNodePtr GetDictNode(const py::dict &dict_obj, const DebugInfoPtr &debug_info, bool set_abstract);
  AnfNodePtr GetDictKeyNode(const py::object &dict_keys, const DebugInfoPtr &debug_info, bool set_abstract);
  AnfNodePtr GetDictValueNode(const py::object &dict_values, const DebugInfoPtr &debug_info, bool set_abstract);
  AnfNodePtr GetDictItemNode(const py::object &dict_items, const DebugInfoPtr &debug_info, bool set_abstract);
  AnfNodePtr ConvertParameterObj(const py::object &input_obj);

  void SetNode(const py::object &obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info,
               bool set_abstract = false);
  void SetTupleNode(const py::tuple &tuple_obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info,
                    bool set_abstract);
  void SetListNode(const py::list &list_obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info, bool set_abstract);

  void SetDictNode(const py::dict &dict_obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info, bool set_abstract);

  void Clear();

  std::string phase_;
  py::args args_;
  OrderedSet<AnfNodePtr> side_effect_nodes_;
  std::stack<FuncGraphPtr> graph_stack_;
  std::unordered_map<std::string, AnfNodePtr> py_obj_node_map_;  // The map from py::object id() to AnfNode.
};
}  // namespace trace
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_TRACE_RECORDER_H_
