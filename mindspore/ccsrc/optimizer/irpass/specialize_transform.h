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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_SPECIALIZE_TRANSFORM_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_SPECIALIZE_TRANSFORM_H_

#include <map>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>

#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"
#include "ir/manager.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
class SpecializeTransform {
 public:
  SpecializeTransform() : cache_() {}
  ~SpecializeTransform() = default;

  FuncGraphPtr operator()(const FuncGraphPtr &func_graph, std::vector<FuncGraphPtr> graph_args,
                          std::vector<PrimitivePtr> prim_args) {
    if (cache_.count(func_graph) == 0) {
      cache_[func_graph] = {};
    }

    auto &cache = cache_[func_graph];
    auto key = std::make_pair(graph_args, prim_args);
    if (cache.count(key) == 0) {
      auto mng = func_graph->manager();
      MS_EXCEPTION_IF_NULL(mng);

      FuncGraphPtr new_fg = TransformableClone(func_graph, std::make_shared<TraceTransform>("sp"));
      mng->AddFuncGraph(new_fg);

      std::vector<AnfNodePtr> params = new_fg->parameters();
      std::vector<AnfNodePtr> new_params;
      size_t n = graph_args.size();
      for (size_t i = 0; i < n; i++) {
        if (graph_args[i] != nullptr) {
          auto arg = NewValueNode(graph_args[i]);
          (void)mng->Replace(params[i], arg);
          continue;
        }
        if (prim_args[i] != nullptr) {
          auto arg = NewValueNode(prim_args[i]);
          (void)mng->Replace(params[i], arg);
          continue;
        }
        new_params.push_back(params[i]);
      }

      mng->SetParameters(new_fg, new_params);
      cache[key] = new_fg;
    }
    return cache[key];
  }

 private:
  std::unordered_map<FuncGraphPtr,
                     std::map<std::pair<std::vector<FuncGraphPtr>, std::vector<PrimitivePtr>>, FuncGraphPtr>>
    cache_;
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
    if (inp0_fg->recursive()) {
      return nullptr;
    }

    std::vector<FuncGraphPtr> graph_args;
    std::vector<PrimitivePtr> prim_args;
    std::vector<AnfNodePtr> new_xs;
    bool hasVNode = false;
    for (size_t i = 1; i < inputs.size(); i++) {
      if (IsValueNode<FuncGraph>(inputs[i])) {
        auto fg_vnode = GetValueNode<FuncGraphPtr>(inputs[i]);
        graph_args.push_back(fg_vnode);
        prim_args.emplace_back(nullptr);
        hasVNode = true;
      } else if (IsValueNode<Primitive>(inputs[i])) {
        auto p_vnode = GetValueNode<PrimitivePtr>(inputs[i]);
        graph_args.emplace_back(nullptr);
        prim_args.push_back(p_vnode);
        hasVNode = true;
      } else {
        graph_args.emplace_back(nullptr);
        prim_args.emplace_back(nullptr);
        new_xs.push_back(inputs[i]);
      }
    }

    if (!hasVNode) {
      return nullptr;
    }

    auto new_fg = specialize_transform_(inp0_fg, graph_args, prim_args);
    (void)new_xs.insert(new_xs.begin(), NewValueNode(new_fg));

    return node->func_graph()->NewCNode(new_xs);
  }

 private:
  internal::SpecializeTransform specialize_transform_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_SPECIALIZE_TRANSFORM_H_
