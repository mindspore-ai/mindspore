/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/transpose_eliminate.h"

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
// check if node is value tuple and ascends one by one from zero. e.g., (0, 1, 2, 3)
// {PrimTranspose, X, AscendingNums}
AnfNodePtr TransposeSameIOEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimTranspose, {IsNode, IsVNode})(node);

  // check pattern match
  if (tuple_ == nullptr) {
    return nullptr;
  }

  auto value = GetValueNode(tuple_);
  if (value == nullptr || !value->isa<ValueSequence>()) {
    return nullptr;
  }
  auto elements = GetValue<std::vector<int64_t>>(value);
  if (elements.empty()) {
    return nullptr;
  }

  int64_t j = 0;
  bool cmp = std::all_of(elements.cbegin(), elements.cend(), [&j](int64_t i) { return i == j++; });
  // same IO settings, eliminate this transpose
  if (cmp) {
    return x_;
  }

  return nullptr;
}

void TransposeSameIOEliminater::Visit(const AnfNodePtr &node) {
  if (x_ == nullptr) {
    x_ = node;
  } else {
    tuple_ = node;
  }
}

void TransposeSameIOEliminater::Reset() {
  x_ = nullptr;
  tuple_ = nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
