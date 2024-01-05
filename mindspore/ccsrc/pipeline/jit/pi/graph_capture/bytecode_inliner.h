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
 * collect trace nodes for each sub-graph, rebuild bytecode by nodes.
 * if allowed inline the break graph, inline the second half bytecode.
 * guard the global variable if the globals of inlined function is different
 * from top function, eliminate the sideeffect or do not inline a function
 * with sideeffect
 */
class BytecodeInliner {
 public:
  BytecodeInliner(Graph *graph, const py::dict &global)
      : graph_(graph),
        traced_nodes_(),
        extra_globals_(global),
        cfg_(),
        last_frame_(),
        new_frames_(),
        reconstructed_value_(nullptr),
        new_break_bci_(-1),
        inline_partial_(false) {}

  void Run();

 private:
  // prepare and call rebuild bytecodes by nodes
  void Rebuild();

  // rebuild bytecodes and frame statue
  void Rebuild(CodeGenerator *cg);

  void EraseDeadLocal(const std::vector<ValueNode *> &alive_nodes);

  void ResetCFG(CodeGenerator *cg);

  void ResetGraphStat();

  // collect traced nodes, collect bytecodes after break
  void ProcessGraph(Graph *, int local_off = 0);

  // reconstruct node by bytecode
  void Reconstruct(ValueNode *node, int local_off);

  // initialize cfg by instruction list
  void InitCFG();

  // reset instruction oparg, guard globals which merge to top func. eliminate sideeffect of inline
  void FixInstr(Graph *, int local_off, std::vector<std::unique_ptr<Instr>> *);

  void CollectTracedNodes(Graph *);

  // top graph
  Graph *const graph_;

  // all traced nodes
  std::vector<ValueNode *> traced_nodes_;

  // used globals of function and inlined function
  py::dict extra_globals_;

  // new cfg
  std::unique_ptr<CFG> cfg_;

  // new last frame
  std::unique_ptr<FrameStates> last_frame_;

  std::map<int, std::unique_ptr<FrameStates>> new_frames_;

  ValueNode *reconstructed_value_;

  int new_break_bci_;

  bool inline_partial_;
};

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
#endif
