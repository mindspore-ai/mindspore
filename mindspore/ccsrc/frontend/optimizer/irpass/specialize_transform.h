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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPECIALIZE_TRANSFORM_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPECIALIZE_TRANSFORM_H_

#include <map>
#include <vector>
#include <memory>
#include <utility>
#include <tuple>

#include "utils/hash_map.h"
#include "frontend/optimizer/irpass.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/operator/ops.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
class SpecializeTransform {
 public:
  SpecializeTransform() : cache_() {}
  ~SpecializeTransform() = default;

  FuncGraphPtr operator()(const FuncGraphPtr &func_graph, const std::vector<ValuePtr> &need_eliminate_args) {
    if (cache_.count(func_graph) == 0) {
      cache_[func_graph] = {};
    }
    auto &cache = cache_[func_graph];
    const auto &key = need_eliminate_args;
    if (cache.count(key) == 0) {
      auto mng = func_graph->manager();
      MS_EXCEPTION_IF_NULL(mng);
      FuncGraphPtr new_fg = TransformableClone(func_graph, MakeTraceInfo<TraceTransform>("sp"));
      mng->AddFuncGraph(new_fg);
      std::vector<AnfNodePtr> params = new_fg->parameters();
      std::vector<AnfNodePtr> new_params;
      for (size_t i = 0; i < need_eliminate_args.size(); i++) {
        // keep the parameter
        if (need_eliminate_args[i] == nullptr) {
          new_params.push_back(params[i]);
          continue;
        }
        // replace the parameter with arg in new_fg without changing origin func_graph.
        (void)mng->Replace(params[i], NewReplaceValueNode(need_eliminate_args[i], func_graph->return_node()));
      }
      mng->SetParameters(new_fg, new_params);
      cache[key] = new_fg;
    }
    return cache[key];
  }

 private:
  mindspore::HashMap<FuncGraphPtr, std::map<std::vector<ValuePtr>, FuncGraphPtr>> cache_;
  static ValueNodePtr NewReplaceValueNode(const ValuePtr &value, const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<FuncGraph>() || value->isa<Primitive>() || value->isa<parse::NameSpace>()) {
      return NewValueNode(value);
    }
    if (value->isa<tensor::Tensor>()) {
      auto &const_tensor = *(value->cast<tensor::TensorPtr>());
      auto const_tensor_ptr = std::make_shared<tensor::Tensor>(const_tensor);
      return NewValueNode(const_tensor_ptr);
    }
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node) << "Unexpected value:" << value->ToString();
  }
};
}  // namespace internal

// {G, Xs}
class SpecializeOnGraphArguments : public AnfVisitor {
 public:
  SpecializeOnGraphArguments() : specialize_transform_() {}
  ~SpecializeOnGraphArguments() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    if (!IsValueNode<FuncGraph>(inputs[0])) {
      return nullptr;
    }

    auto inp0_fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    if (inp0_fg == nullptr || inp0_fg->has_flag(FUNC_GRAPH_FLAG_NO_INLINE) || IsSetRecomputed(inp0_fg) ||
        inp0_fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) || inp0_fg->recursive()) {
      return nullptr;
    }
    std::vector<ValuePtr> need_eliminated_args;
    std::vector<AnfNodePtr> new_xs;
    bool hasVNode = false;
    for (size_t i = 1; i < inputs.size(); i++) {
      if (IsValueNode<FuncGraph>(inputs[i]) || IsValueNode<Primitive>(inputs[i]) ||
          IsValueNode<tensor::Tensor>(inputs[i]) || IsValueNode<parse::NameSpace>(inputs[i])) {
        need_eliminated_args.push_back(GetValueNode(inputs[i]));
        hasVNode = true;
      } else {
        (void)need_eliminated_args.emplace_back(nullptr);
        new_xs.push_back(inputs[i]);
      }
    }
    if (!hasVNode) {
      return nullptr;
    }
    auto new_fg = specialize_transform_(inp0_fg, need_eliminated_args);
    (void)new_xs.insert(new_xs.cbegin(), NewValueNode(new_fg));

    return node->func_graph()->NewCNode(new_xs);
  }

 private:
  static bool IsSetRecomputed(const FuncGraphPtr &fg) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
    return fg->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE) ||
           (cell_reuse &&
            (fg->has_flag(FUNC_GRAPH_NOT_RECOMPUTE_K_GRAPH) || fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)));
  }

  internal::SpecializeTransform specialize_transform_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPECIALIZE_TRANSFORM_H_
