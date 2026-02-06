/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/accumulaten_eliminate.h"
#include <vector>
#include <algorithm>
#include <memory>

#include "frontend/optimizer/irpass.h"
#include "primitive/sequence_ops.h"
#include "primitive/array_ops.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "primitive/auto_generate/gen_ops_primitive_z.h"

namespace mindspore {
namespace opt {
namespace irpass {

AnfNodePtr AccumulateNV2Eliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimAccumulateNV2, {IsCNode})(node);

  if (inputs_.empty() || node->func_graph() == nullptr) {
    return nullptr;
  }

  // If only two filtered inputs nodes, as {make_tuple, x}, return x.
  if (inputs_.size() == 2) {
    return inputs_[1];
  }

  // If only one filtered node, all inputs nodes are zerolike, return one of the input.
  if (inputs_.size() == 1 && args_.size() > 0) {
    return args_[0];
  }

  if (!has_zero_like_) {
    return nullptr;
  }

  auto cnode = node->cast<CNodePtr>();
  auto accumulaten = NewValueNode(GetValueNode(cnode->input(0)));
  auto fg = node->func_graph();
  auto make_tuple = fg->NewCNode(inputs_);
  return fg->NewCNode({accumulaten, make_tuple});
}

void AccumulateNV2Eliminater::Visit(const CNodePtr &cnode) {
  if (!IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
    return;
  }

  auto &inputs = cnode->inputs();
  (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));

  // {kPrimMakeTuple, X1, X2, ...}
  inputs_.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (auto &x : args_) {
    if (!IsPrimitiveCNode(x, prim::kPrimZerosLike)) {
      inputs_.push_back(x);
    } else {
      has_zero_like_ = true;
    }
  }
}

void AccumulateNV2Eliminater::Reset() {
  args_.clear();
  inputs_.clear();
  has_zero_like_ = false;
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
