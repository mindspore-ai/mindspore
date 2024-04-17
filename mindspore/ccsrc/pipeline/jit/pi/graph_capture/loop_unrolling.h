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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_LOOP_UNROLLING_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_LOOP_UNROLLING_H

#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include "pipeline/jit/pi/pydef.h"
#include "pipeline/jit/pi/graph_capture/cfg.h"
#include "pipeline/jit/pi/graph_capture/loop.h"
#include "pipeline/jit/pi/utils/allocator.h"
#include "pipeline/jit/pi/utils/utils.h"

namespace mindspore {
namespace pijit {
class Graph;
class LoopUnrolling {
 public:
  static bool IsloopUnorlling(LoopUnrollingReason res) { return res == kCanForItemUnroll || res == kCanWhileUnroll; }

  explicit LoopUnrolling(Graph &graph) : graph_(graph) {}
  virtual ~LoopUnrolling() = default;

  LoopUnrollingReason ExecuteLoopUnroll(Block *header);
  bool IsCFGChanged() const { return is_cfg_changed_; }
  std::string DumpLoopUnrolling();

 private:
  void Run();
  LoopUnrollingReason AnalyzeForItem();
  bool AddLoopGurad(ValueNode *value);
  LoopUnrollingReason CheckLoopUnrollingSideeffect();
  std::map<int, Block *> CopyBB();
  void CopyAndInsertBB();
  void RemoveBackedge();
  void AddLoopUnrollingInstr(Block *bb, int count);
  void FixupInstr();

  Graph &graph_;
  LoopInfo *loop_ = nullptr;
  LoopUnrollingReason res_ = kCanNotUnroll;
  int unrolling_count_ = 0;
  int loop_op_ = -1;
  int loop_arg_ = -1;
  Instr *iter_instr_ = nullptr;
  ValueNode *loop_value_ = nullptr;
  bool is_cfg_changed_ = false;
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_LOOP_UNROLLING_H
