/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_PARTIAL_ELIMINATE_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_PARTIAL_ELIMINATE_H_

#include <vector>
#include <algorithm>
#include <memory>

#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {{prim::kPrimPartial, X, Xs}, Ys} -> {X, Xs, Ys}
class PartialEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }

    Xs_.clear();
    auto &inputs = node->cast<CNodePtr>()->inputs();
    Visit(inputs[0]);

    if (Xs_.size() == 0) {
      return nullptr;
    }

    // {X, Xs, Ys}
    std::vector<AnfNodePtr> args{};
    (void)std::copy(Xs_.begin(), Xs_.end(), std::back_inserter(args));
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args));
    TraceManager::DebugTrace(std::make_shared<TracePartialTransform>(node->debug_info()));
    auto new_node = node->func_graph()->NewCNode(args);
    TraceManager::EndTrace();
    return new_node;
  }

  void Visit(const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPartial)) {
      return;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    // {prim::kPrimPartial, X, Xs}
    if (inputs.size() < 2) {
      return;
    }

    // fill Xs
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(Xs_));
  }

 private:
  std::vector<AnfNodePtr> Xs_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_PARTIAL_ELIMINATE_H_
