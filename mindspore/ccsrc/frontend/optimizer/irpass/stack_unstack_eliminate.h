/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_STACK_UNSTACK_ELIMINATE_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_STACK_UNSTACK_ELIMINATE_H

#include <algorithm>
#include <memory>
#include <vector>
#include <string>

#include "frontend/optimizer/optimizer_caller.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimStack, {prim::kPrimUnstack, X}} => X
// prim::kPrimUnstack and prim::kPrimStack should have same attribute value of kAttrNum and kAttrAxis.
class StackUnstackEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();

    if (!IsPrimitiveCNode(node, prim::kPrimUnstack)) {
      return nullptr;
    }

    if (!FetchUnstackAttrs(node)) {
      return nullptr;
    }
    AnfVisitor::Match(prim::kPrimUnstack, {IsCNode})(node);

    if (is_match_) {
      return stack_->input(1);
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimStack)) {
      auto prim = GetCNodePrimitive(cnode);
      auto num_val = prim->GetAttr(kAttrNum);
      // Stack may not be inferred and do not have attribute axis.
      if (num_val == nullptr) {
        return;
      }
      auto axis_val = prim->GetAttr(kAttrAxis);
      MS_EXCEPTION_IF_NULL(axis_val);
      auto num = dyn_cast<Int64Imm>(num_val)->value();
      auto axis = dyn_cast<Int64Imm>(axis_val)->value();
      if (num == num_ && axis == axis_) {
        is_match_ = true;
        stack_ = cnode;
      }
    }
  }

  bool FetchUnstackAttrs(const AnfNodePtr &node) {
    auto prim = GetCNodePrimitive(node);
    auto num_val = prim->GetAttr(kAttrNum);
    // UnStack may not be inferred and do not have attribute axis.
    if (num_val == nullptr || num_val->isa<None>()) {
      return false;
    }
    auto axis_val = prim->GetAttr(kAttrAxis);
    MS_EXCEPTION_IF_NULL(axis_val);
    num_ = dyn_cast<Int64Imm>(num_val)->value();
    axis_ = dyn_cast<Int64Imm>(axis_val)->value();
    return true;
  }

  void Reset() {
    is_match_ = false;
    num_ = 0;
    axis_ = 0;
    stack_ = nullptr;
  }

 private:
  bool is_match_{false};
  int64_t num_{0};
  int64_t axis_{0};
  CNodePtr stack_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_STACK_UNSTACK_ELIMINATE_H
