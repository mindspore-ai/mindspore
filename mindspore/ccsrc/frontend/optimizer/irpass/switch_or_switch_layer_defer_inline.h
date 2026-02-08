/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include "primitive/framework_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "utils/compile_config.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimPartial, func_graph, ...}
class PartialDeferInline : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    static const bool enable_pre_lift = (common::GetCompileConfig("PRE_LIFT") == "1");
    if (!enable_pre_lift) {
      return nullptr;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr auto func_index = 1;
    auto inp1 = cnode->input(func_index);
    MS_EXCEPTION_IF_NULL(inp1);
    auto func_abs = inp1->abstract();
    MS_EXCEPTION_IF_NULL(func_abs);
    auto real_func = dyn_cast<abstract::FuncGraphAbstractClosure>(func_abs);
    if (real_func != nullptr) {
      *(real_func->func_graph()->indirect()) = true;
    }
    return nullptr;
  }
};

// {prim::kPrimSwitch, cond, true_branch, false_branch}
class SwitchDeferInline : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr size_t true_index = 2;
    auto true_abstract = dyn_cast<abstract::FuncGraphAbstractClosure>(cnode->input(true_index)->abstract());
    if (true_abstract != nullptr) {
      *(true_abstract->func_graph()->indirect()) = true;
    }
    constexpr size_t false_index = 3;
    auto false_abstract = dyn_cast<abstract::FuncGraphAbstractClosure>(cnode->input(false_index)->abstract());
    if (false_abstract != nullptr) {
      *(false_abstract->func_graph()->indirect()) = true;
    }
    return nullptr;
  }
};

// {prim::kPrimSwitchLayer, Index, layers}
class SwitchLayerDeferInline : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr size_t layers_index = 2;
    auto tuple = dyn_cast<abstract::AbstractTuple>(cnode->input(layers_index)->abstract());
    if (tuple == nullptr) {
      return nullptr;
    }
    for (auto elem : tuple->elements()) {
      auto abstract = dyn_cast<abstract::FuncGraphAbstractClosure>(elem);
      if (abstract != nullptr) {
        *(abstract->func_graph()->indirect()) = true;
      }
    }
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SWITCH_LAYER_DEFER_INLINE_H_
