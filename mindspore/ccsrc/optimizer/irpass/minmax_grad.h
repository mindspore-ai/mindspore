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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_MINMAX_GRAD_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_MINMAX_GRAD_H_

#include <vector>
#include <memory>

#include "optimizer/optimizer.h"
#include "optimizer/irpass.h"
#include "ir/visitor.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
// check if node is MinimumGrad() or MaximumGrad()
bool IsOriginMaxMinGrad(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimMaximumGrad) && !IsPrimitiveCNode(node, prim::kPrimMinimumGrad)) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto x_v = prim->GetAttr("grad_x");
  auto y_v = prim->GetAttr("grad_y");
  if (x_v == nullptr || y_v == nullptr || !x_v->isa<BoolImm>() || !y_v->isa<BoolImm>()) {
    return false;
  }

  bool x = GetValue<bool>(x_v);
  bool y = GetValue<bool>(y_v);
  return x && y;
}
}  // namespace internal

// {prim::kPrimTupleGetItem, {target_grad, Xs}, C}
class MinMaximumGrad : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {internal::IsOriginMaxMinGrad, IsValueNode<Int32Imm>})(node);
    if (grad_ == nullptr || idx_ < 0 || idx_ > 1 || node->func_graph() == nullptr) {
      return nullptr;
    }

    // check single use
    auto mng = optimizer->resource()->manager();
    auto &users = mng->node_users();
    if (users.find(grad_) == users.end() || users[grad_].size() != 1) {
      return nullptr;
    }

    // {target_grad, Xs}
    auto &inputs = grad_->inputs();
    auto prim = GetValueNode<PrimitivePtr>(inputs[0]);

    auto new_prim = std::make_shared<Primitive>(prim->name());
    new_prim->set_attr("grad_x", MakeValue(true));
    new_prim->set_attr("grad_y", MakeValue(true));

    if (idx_ == 0) {
      new_prim->set_attr("grad_y", MakeValue(false));
    }
    if (idx_ == 1) {
      new_prim->set_attr("grad_x", MakeValue(false));
    }

    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(new_prim));
    (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());

    auto fg = node->func_graph();
    auto tuple = fg->NewCNode(args);

    return fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), tuple, NewValueNode(MakeValue(idx_))});
  }

  void Visit(const CNodePtr &cnode) override { grad_ = cnode; }

  void Visit(const ValueNodePtr &vnode) override { idx_ = GetValue<int>(vnode->value()); }

  void Reset() {
    idx_ = -1;
    grad_ = nullptr;
  }

 private:
  int idx_{-1};
  CNodePtr grad_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_MINMAX_GRAD_H_
