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
#include "pipeline/jit/graph_jit/graph_capture/loop.h"
#include <fstream>
#include <string>
#include "pipeline/jit/graph_jit/graph_capture/cfg.h"
#include "pipeline/jit/graph_jit/graph_capture/graph.h"

namespace mindspore {
namespace jit {
namespace graph {
std::string LoopInfo::Dump() const {
  std::stringstream os;
  os << "header: " << header_->id();
  os << " backedge: ";
  for (auto *bb : backedges_) {
    os << bb->id() << " ";
  }
  os << "\n members: ";
  for (auto *bb : loop_members_) {
    os << bb->id() << " ";
  }
  os << "\n exits: ";
  for (auto *bb : exits_) {
    os << bb->id() << " ";
  }
  os << "\n";
  return os.str();
}

LoopFinder::LoopFinder(Graph *graph) : graph_(*graph), alloc_(graph->allocator()) {}

void LoopFinder::UpdateLoop2Graph() {
  for (LoopInfo *loop = loops_; loop != nullptr; loop = loop->next()) {
    graph_.AddLoop(loop);
  }
}

void LoopFinder::FormSimpleLoopInfo() {
  const std::unique_ptr<CFG> &cfg = graph_.GetCFG();
  if (cfg == nullptr) {
    return;
  }
  LoopInfo *tail_loop = nullptr;
  for (Block *bb : *cfg) {
    if (!bb->is_loop_head() || bb->GetJumpBB() == nullptr || bb->GetJumpBB()->id() == bb->id()) {
      continue;
    }
    LoopInfo *loop = alloc_.NewLoopInfo<LoopInfo>();
    loop->set_header(bb);
    loop->InsertExit(bb->GetJumpBB());
    for (Block *pred : bb->pred_bbs()) {
      if (pred->instrs().front().bci() > bb->instrs().front().bci()) {
        loop->InsertBackedge(pred);
      }
    }
    if (loops_ == nullptr) {
      loops_ = loop;
    } else {
      tail_loop->set_next(loop);
      loop->set_prev(tail_loop);
    }
    for (uint32_t i = bb->id(); i < bb->GetJumpBB()->id(); ++i) {
      loop->InsertLoopMembers(cfg->bb_pool()[i].get());
    }
    tail_loop = loop;
  }
  UpdateLoop2Graph();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
