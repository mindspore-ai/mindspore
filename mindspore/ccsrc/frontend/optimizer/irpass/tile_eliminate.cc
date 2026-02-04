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
#include "frontend/optimizer/irpass/tile_eliminate.h"
#include <vector>
#include <algorithm>

#include "frontend/optimizer/irpass.h"
#include "primitive/array_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
using ShapePtr = mindspore::abstract::ShapePtr;
// check if node is value tuple and all one. e.g. (1, 1, 1)
// {PrimTile, X, MultiOne} and x.dim >= len(tuple) -> X
// {PrimTile, X, Empty} -> X
AnfNodePtr TileEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
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

  auto fn = [this]() -> size_t {
    auto x_shape_base = x_->Shape();
    uint64_t x_size = 0;
    ShapePtr x_shape;
    if (x_shape_base && (x_shape = x_shape_base->cast<ShapePtr>())) {
      x_size = x_shape->shape().size();
    }
    return x_size;
  };

  // Return x_ directly when x.dim >= len(tuple) and all elements of tuple are 1.
  // if len(tuple) > x.dim need expand x.dim
  auto cmp = std::all_of(elements.cbegin(), elements.cend(), [](int64_t i) { return i == 1; });
  if (cmp && fn() >= elements.size()) {
    return x_;
  }

  return nullptr;
}

void TileEliminater::Visit(const AnfNodePtr &node) {
  if (x_ == nullptr) {
    x_ = node;
  } else {
    tuple_ = node;
  }
}

void TileEliminater::Reset() {
  x_ = nullptr;
  tuple_ = nullptr;
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
