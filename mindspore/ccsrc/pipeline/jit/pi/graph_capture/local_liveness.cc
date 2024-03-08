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
#include "pipeline/jit/pi/graph_capture/local_liveness.h"
#include <algorithm>
#include <memory>
#include <utility>
#include "pipeline/jit/pi/graph_capture/graph.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace pijit {

BitMap Liveness::CollectAlive(int start_bci) const {
  Block *cur = cfg_->GetBlockByBci(start_bci);
  if (start_bci == cur->begin_ci()) {
    return alive_[cur->id()];
  }
  BitMap read(cfg_->GetLocalCount());
  BitMap write(cfg_->GetLocalCount());
  for (int bci = start_bci; bci < cur->end_ci(); ++bci) {
    Liveness::BuildRW(*cfg_->instr_pool()[bci], &read, &write);
  }
  for (const auto &i : cur->succ_bbs()) {
    read.Or(alive_[i->id()]);
  }
  read.Diff(write);
  return read;
}

void Liveness::BuildRW(const Instr &instr, BitMap *read, BitMap *write) {
  if (instr.op() == STORE_FAST || instr.op() == DELETE_FAST) {
    if (!read->Get(instr.arg())) {
      write->Set(instr.arg());
    }
    return;
  }
  if (instr.op() == LOAD_FAST) {
    if (!write->Get(instr.arg())) {
      read->Set(instr.arg());
    }
    return;
  }
}

void Liveness::Init() {
  const auto &bb = cfg_->bb_pool();
  int block_count = bb.size();
  read_.resize(block_count, BitMap(cfg_->GetLocalCount()));
  write_.resize(block_count, BitMap(cfg_->GetLocalCount()));
  alive_.resize(block_count, BitMap(cfg_->GetLocalCount()));
  alive_effect_.resize(block_count, BitMap(cfg_->GetLocalCount()));

  // generate read write for each block
  for (const auto &block : cfg_->bb_pool()) {
    int id = block->id();
    for (int bci = block->begin_ci(); bci != block->end_ci(); ++bci) {
      Liveness::BuildRW(*cfg_->instr_pool()[bci], &read_[id], &write_[id]);
    }
  }

  // reverse traversal each block, generate alive effect of previous block, and propagate alive to previous
  std::vector<Block *> list;
  std::transform(bb.begin(), bb.end(), std::back_inserter(list), [](const auto &i) { return i.get(); });
  while (!list.empty()) {
    Block *cur = list.back();
    list.pop_back();
    Propagate(cur, &list);
  }
}

/**
 * liveness propagate for each block.
 * merge alive to each previous block, it is previous block end alive effect.
 * merge alive effect to alive, difference block write(kill), merge block read(generate),
 * it is current block start alive.
 */
void Liveness::Propagate(Block *cur, std::vector<Block *> *list) {
  int index = cur->id();
  alive_[index].Or(alive_effect_[index]);
  alive_[index].Diff(write_[index]);
  alive_[index].Or(read_[index]);

  for (auto i : cur->pred_bbs()) {
    int next = i->id();
    if (alive_effect_[next].OrWithChange(alive_[index])) {
      list->push_back(i);
    }
  }
}

}  // namespace pijit
}  // namespace mindspore
