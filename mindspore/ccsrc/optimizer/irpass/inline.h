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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_INLINE_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_INLINE_H_

#include <vector>
#include <utility>
#include <algorithm>

#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
class ReplaceApplicator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsValueNode<FuncGraph>(node)) {
      return nullptr;
    }

    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE)) {
      return nullptr;
    }

    auto out = fg->output();
    MS_EXCEPTION_IF_NULL(out);
    if (!out->isa<CNode>()) {
      return nullptr;
    }

    auto &inputs = out->cast<CNodePtr>()->inputs();
    auto params = fg->parameters();

    // Exclude first elements of inputs which is fn.
    auto input_size = inputs.size();
    auto param_size = params.size();
    if ((input_size == 1 && param_size == 0) || (input_size > 1 && (input_size - 1) == param_size &&
                                                 std::equal(inputs.begin() + 1, inputs.end(), params.begin()))) {
      auto inner = inputs[0];
      if (IsValueNode<Primitive>(inner) ||
          (IsValueNode<FuncGraph>(inner) && GetValueNode<FuncGraphPtr>(inner)->parent() == nullptr)) {
        return inner;
      }
    }

    return nullptr;
  }
};

using CriterionFuncType = std::function<bool(FuncGraphPtr, AnfNodePtr)>;

bool IsTrivial(const FuncGraphPtr &fg, AnfNodePtr) {
  auto n_cnode = fg->nodes().size() - fg->parameters().size();
  // There is at least one CNode(return, other_node).
  return n_cnode <= 2;
}

bool IsUniqueUse(const FuncGraphPtr &fg, AnfNodePtr) {
  auto &cnodes = fg->func_graph_cnodes_index();
  int n_use =
    std::accumulate(cnodes.begin(), cnodes.end(), 0,
                    [](int sum, const std::pair<const CNodeIndexPairPtr, int> &item) { return sum + item.second; });
  return n_use == 1;
}

bool IsInside(FuncGraphPtr, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node->func_graph());
  return node->func_graph()->has_flag("inline_inside");
}

bool IsCore(const FuncGraphPtr &fg, AnfNodePtr) { return fg->has_flag("core"); }

bool NoCriterion(FuncGraphPtr, AnfNodePtr) { return true; }

// {G, Xs}
class InlinerBase : public AnfVisitor {
 public:
  explicit InlinerBase(std::vector<std::pair<CriterionFuncType, bool>> criterions) : criterions_(criterions) {}
  ~InlinerBase() override = default;
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>()) {
      return nullptr;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    if (inputs.size() < 1 || !IsValueNode<FuncGraph>(inputs[0])) {
      return nullptr;
    }

    // G
    auto fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    if (fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE)) {
      return nullptr;
    }
    // Do not inline GraphKernel to Cell.
    if (fg->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL) && !node->func_graph()->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      // If the GraphKernel only contains a return node, we make it inlined.
      if (fg->nodes().size() - fg->parameters().size() > 1) {
        return nullptr;
      }
    }

    Reset();
    bool is_match = false;
    for (auto &criterion : criterions_) {
      if (!criterion.first(fg, node)) {
        continue;
      }

      if (criterion.second && IsRecursive(fg)) {
        continue;
      }

      is_match = true;
      break;
    }

    if (!is_match) {
      return nullptr;
    }

    std::vector<AnfNodePtr> params;
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(params));

    if (IsUniqueUse(fg, nullptr)) {
      auto mng = fg->manager();
      MS_EXCEPTION_IF_NULL(mng);
      ReplaceParams(mng, params, fg);
      auto out_node = fg->output();
      mng->MoveAllCNodeDropGraph(fg, node->func_graph(), inputs[0]->scope());
      return out_node;
    }

    return InlineClone(fg, node->func_graph(), params, inputs[0]->scope());
  }

  void ReplaceParams(const FuncGraphManagerPtr &mng, const std::vector<AnfNodePtr> &new_params,
                     const FuncGraphPtr &fg) {
    auto params = fg->parameters();
    auto old_size = params.size();
    if (old_size != new_params.size()) {
      MS_LOG(EXCEPTION) << "Parameter size not match." << old_size << " new " << new_params.size()
                        << fg->output()->DebugString(10);
    }
    for (size_t i = 0; i < old_size; i++) {
      (void)mng->Replace(params[i], new_params[i]);
    }
  }

  bool IsRecursive(const FuncGraphPtr &fg) {
    if (!is_checked_) {
      is_checked_ = true;
      is_recursive_ = fg->recursive();
    }
    return is_recursive_;
  }

  void Reset() {
    is_checked_ = false;
    is_recursive_ = false;
  }

 private:
  bool is_checked_{false}, is_recursive_{false};
  std::vector<std::pair<CriterionFuncType, bool>> criterions_;
};

class Inliner : public InlinerBase {
 public:
  Inliner()
      : InlinerBase({
          {IsUniqueUse, true},
          {IsTrivial, false},
          {IsInside, false},
          {IsCore, false},
          {NoCriterion, true},
        }) {}
  ~Inliner() override = default;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_INLINE_H_
