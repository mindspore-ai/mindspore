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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MINMAX_GRAD_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MINMAX_GRAD_H_

#include <vector>
#include <memory>

#include "frontend/optimizer/optimizer.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimTupleGetItem, {target_grad, Xs}, C}
class MinMaximumGrad : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {MinMaximumGrad::IsOriginMaxMinGrad, IsValueNode<Int64Imm>})(node);
    if (grad_ == nullptr || idx_ < 0 || idx_ > 1 || node->func_graph() == nullptr) {
      return nullptr;
    }

    // check single use
    auto mng = optimizer->manager();
    MS_EXCEPTION_IF_NULL(mng);
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
    (void)args.insert(args.cend(), inputs.cbegin() + 1, inputs.cend());

    auto fg = node->func_graph();
    auto new_code = fg->NewCNode(args);
    if (AnfUtils::GetDumpFlag(grad_)) {
      AnfUtils::SetDumpFlag(new_code);
    }

    return fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), new_code, NewValueNode(MakeValue(idx_))});
  }

  void Visit(const CNodePtr &cnode) override { grad_ = cnode; }

  void Visit(const ValueNodePtr &vnode) override { idx_ = GetValue<int64_t>(vnode->value()); }

  void Reset() {
    idx_ = -1;
    grad_ = nullptr;
  }

  // Check if node is MinimumGrad() or MaximumGrad()
  static bool IsOriginMaxMinGrad(const AnfNodePtr &node) {
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

 private:
  int64_t idx_{-1};
  CNodePtr grad_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MINMAX_GRAD_H_
