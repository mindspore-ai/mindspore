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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_ALLOCATOR_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_ALLOCATOR_H

#include <utility>
#include <vector>
#include "frontend/jit/pi/graph_capture/node.h"

namespace mindspore {
namespace pijit {

class Instr;
class LoopInfo;
class Allocator {
 public:
  Allocator() = default;
  ~Allocator();

  const std::vector<Instr *> &instr_pool() const { return instr_pool_; }
  const std::vector<AbstractNode *> &node_pool() const { return node_pool_; }

  InstrNode *NewInstrNode(int op, int arg);
  ValueNode *NewValueNode(AObject *a, int b, int c, const std::vector<ValueNode *> &d);

  template <class T, typename... Args>
  T *NewNode(Args &&...args) {
    T *v = new T(std::forward<Args>(args)...);
    AddNodePool(v);
    return v;
  }

  template <class T, typename... Args>
  T *NewInstr(Args &&...args) {
    T *v = new T(std::forward<Args>(args)...);
    AddInstrPool(v);
    return v;
  }

 private:
  void AddInstrPool(Instr *v) { instr_pool_.push_back(v); }
  void AddNodePool(AbstractNode *v) { node_pool_.push_back(v); }

  std::vector<Instr *> instr_pool_;
  std::vector<AbstractNode *> node_pool_;
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_ALLOCATOR_H
