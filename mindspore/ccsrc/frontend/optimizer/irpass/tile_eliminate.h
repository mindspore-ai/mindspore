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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_TILE_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_TILE_ELIMINATE_H_

#include <vector>
#include <algorithm>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// check if node is value tuple and all one. e.g. (1, 1, 1)
// {PrimTile, X, MultiOne} -> X
// {PrimTile, X, Empty} -> X
class TileEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTile, {IsNode, IsVNode})(node);

    // check pattern match
    if (tuple_ == nullptr) {
      return nullptr;
    }

    auto value = GetValueNode(tuple_);
    auto elements = GetValue<std::vector<int64_t>>(value);
    if (elements.empty()) {
      return x_;
    }

    auto cmp = std::all_of(elements.cbegin(), elements.cend(), [](int64_t i) { return i == 1; });
    if (cmp) {
      return x_;
    }

    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (x_ == nullptr) {
      x_ = node;
    } else {
      tuple_ = node;
    }
  }

  void Reset() {
    x_ = nullptr;
    tuple_ = nullptr;
  }

 private:
  AnfNodePtr x_{nullptr}, tuple_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_TILE_ELIMINATE_H_
