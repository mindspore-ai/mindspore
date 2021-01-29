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
#ifndef MINDSPORE_CCSRC_DEBUG_RDR_GRAPH_RECORDER_H_
#define MINDSPORE_CCSRC_DEBUG_RDR_GRAPH_RECORDER_H_

#include <vector>
#include <string>
#include <memory>

#include "debug/anf_ir_utils.h"
#include "debug/rdr/base_recorder.h"
namespace mindspore {
class FuncGraph;
using FuncGraphPtr = std::shared_ptr<FuncGraph>;
class GraphRecorder : public BaseRecorder {
 public:
  GraphRecorder() : BaseRecorder(), func_graph_(nullptr), graph_type_("") {}
  GraphRecorder(const std::string &module, const std::string &tag, const FuncGraphPtr &graph,
                const std::string &file_type, const int graph_id)
      : BaseRecorder(module, tag), func_graph_(graph), graph_type_(file_type), id_(graph_id) {}
  ~GraphRecorder() {}
  void SetModule(const std::string &module) { module_ = module; }
  void SetGraphType(const std::string &file_type) { graph_type_ = file_type; }
  void SetFuncGraph(const FuncGraphPtr &func_graph) { func_graph_ = func_graph; }
  void SetDumpFlag(bool full_name) { full_name_ = full_name; }
  void SetNodeId(int id) { id_ = id; }
  virtual void Export();

 private:
  FuncGraphPtr func_graph_;
  std::string graph_type_;
  bool full_name_{false};
  int id_{0};
};
using GraphRecorderPtr = std::shared_ptr<GraphRecorder>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_GRAPH_RECORDER_H_
