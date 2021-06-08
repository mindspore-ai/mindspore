/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SWITCH_LAYER_DEFER_INLINE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SWITCH_LAYER_DEFER_INLINE_H_

#include <vector>
#include <algorithm>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimSwitch, cond, true_branch, false_branch}
class SwitchDeferInline : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto cnode = node->cast<CNodePtr>();
    auto true_abstract = dyn_cast<abstract::FuncGraphAbstractClosure>(cnode->input(2)->abstract());
    if (true_abstract != nullptr) {
      *(true_abstract->func_graph()->switch_input()) = true;
    }
    auto false_abstract = dyn_cast<abstract::FuncGraphAbstractClosure>(cnode->input(3)->abstract());
    if (false_abstract != nullptr) {
      *(false_abstract->func_graph()->switch_input()) = true;
    }
    return nullptr;
  }
};

// {prim::kPrimSwitchLayer, Index, layers}
class SwitchLayerDeferInline : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto cnode = node->cast<CNodePtr>();
    auto tuple = dyn_cast<abstract::AbstractTuple>(cnode->input(2)->abstract());
    if (tuple == nullptr) {
      return nullptr;
    }
    for (auto elem : tuple->elements()) {
      auto abstract = dyn_cast<abstract::FuncGraphAbstractClosure>(elem);
      if (abstract != nullptr) {
        *(abstract->func_graph()->switch_layer_input()) = true;
      }
    }
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SWITCH_LAYER_DEFER_INLINE_H_
