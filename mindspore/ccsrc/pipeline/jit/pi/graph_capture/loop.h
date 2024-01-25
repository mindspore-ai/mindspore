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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_LOOP_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_LOOP_H

#include <memory>
#include <queue>
#include <set>
#include <string>
#include "pipeline/jit/pi/pydef.h"
#include "pipeline/jit/pi/utils/allocator.h"

namespace mindspore {
namespace pijit {
class Block;
class LoopInfo {
 public:
  LoopInfo() = default;
  ~LoopInfo() = default;

  LoopInfo *prev() { return prev_; }
  void set_prev(LoopInfo *loop) { prev_ = loop; }
  LoopInfo *next() { return next_; }
  void set_next(LoopInfo *loop) { next_ = loop; }
  Block *header() const { return header_; }
  void set_header(Block *bb) { header_ = bb; }
  const std::set<Block *> &loop_members() const { return loop_members_; }
  const std::set<Block *> &backedges() const { return backedges_; }
  const std::set<Block *> &exits() const { return exits_; }
  void InsertLoopMembers(Block *bb) { loop_members_.insert(bb); }
  void InsertBackedge(Block *bb) { backedges_.insert(bb); }
  void InsertExit(Block *bb) { exits_.insert(bb); }
  std::string Dump() const;

 protected:
  LoopInfo *prev_ = nullptr;
  LoopInfo *next_ = nullptr;

 private:
  Block *header_ = nullptr;
  std::set<Block *> otherLoopEntries_;
  std::set<Block *> loop_members_;
  std::set<Block *> backedges_;
  std::set<Block *> exits_;
  std::set<Block *> innerLoops_;
  LoopInfo *outerLoop_ = nullptr;
};

class Graph;
class LoopFinder {
 public:
  explicit LoopFinder(Graph *graph);
  ~LoopFinder() = default;

  void FormSimpleLoopInfo();
  void UpdateLoop2Graph();

 private:
  Graph &graph_;
  Allocator &alloc_;
  LoopInfo *loops_ = nullptr;
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_LOOP_H
