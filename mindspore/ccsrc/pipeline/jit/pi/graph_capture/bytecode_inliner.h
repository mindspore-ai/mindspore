/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PI_GRAPH_CAPTURE_BYTECODE_INLINER_H
#define MINDSPORE_CCSRC_PIPELINE_JIT_PI_GRAPH_CAPTURE_BYTECODE_INLINER_H

#include <vector>
#include <map>
#include <memory>
#include "pipeline/jit/pi/graph_capture/cfg.h"
#include "pipeline/jit/pi/graph_capture/code_generator.h"

namespace mindspore {
namespace jit {
namespace graph {

class CallNode;

/**
 * used by kFeatureBreakAtInlinedFunction
 */
class BytecodeInliner {
 public:
  BytecodeInliner(Graph *graph, const py::dict &global) : graph_(graph), extra_globals_(global), new_break_bci_(-1) {}

  void Run();

 private:
  void Rebuild();
  void Rebuild(CodeGenerator *cg);
  void ResetCFG(CodeGenerator *cg);
  void ResetGraphStat();

  void ProcessGraph(Graph *, int local_off = 0);
  void EraseDeadLocal(const std::vector<ValueNode *> &alive_nodes);
  void Reconstruct(ValueNode *node, int local_off);
  void InitCFG();

  void FixInstr(Graph *, int local_off, std::vector<std::unique_ptr<Instr>> *);
  void CollectTracedNodes(Graph *);

  Graph *const graph_;
  std::vector<ValueNode *> traced_nodes_;
  py::dict extra_globals_;

  std::unique_ptr<CFG> cfg_;
  std::unique_ptr<FrameStates> last_frame_;
  std::map<int, std::unique_ptr<FrameStates>> new_frames_;
  int new_break_bci_;
  bool inline_partial_;
};

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
#endif
