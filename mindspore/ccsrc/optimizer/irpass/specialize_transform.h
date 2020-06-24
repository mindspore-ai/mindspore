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
#include <unordered_set>

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
                          std::vector<PrimitivePtr> prim_args, std::vector<tensor::TensorPtr> value_args) {
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
        if (value_args[i] != nullptr) {
          auto &const_tensor = *value_args[i];
          auto const_tensor_ptr = std::make_shared<tensor::Tensor>(const_tensor);
          AnfNodePtr arg = NewValueNode(const_tensor_ptr);
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
    std::vector<tensor::TensorPtr> value_node_args;
    std::vector<AnfNodePtr> new_xs;
    bool hasVNode = false;
    for (size_t i = 1; i < inputs.size(); i++) {
      if (IsValueNode<FuncGraph>(inputs[i])) {
        auto fg_vnode = GetValueNode<FuncGraphPtr>(inputs[i]);
        graph_args.push_back(fg_vnode);
        prim_args.emplace_back(nullptr);
        value_node_args.emplace_back(nullptr);
        hasVNode = true;
      } else if (IsValueNode<Primitive>(inputs[i])) {
        auto p_vnode = GetValueNode<PrimitivePtr>(inputs[i]);
        graph_args.emplace_back(nullptr);
        prim_args.push_back(p_vnode);
        value_node_args.emplace_back(nullptr);
        hasVNode = true;
      } else if (IsValueNode<tensor::Tensor>(inputs[i])) {
        tensor::TensorPtr t_vnode = GetValueNode<tensor::TensorPtr>(inputs[i]);
        graph_args.emplace_back(nullptr);
        prim_args.emplace_back(nullptr);
        value_node_args.emplace_back(t_vnode);
        hasVNode = true;
      } else {
        graph_args.emplace_back(nullptr);
        prim_args.emplace_back(nullptr);
        value_node_args.emplace_back(nullptr);
        new_xs.push_back(inputs[i]);
      }
    }

    if (!hasVNode) {
      return nullptr;
    }

    auto new_fg = specialize_transform_(inp0_fg, graph_args, prim_args, value_node_args);
    (void)new_xs.insert(new_xs.begin(), NewValueNode(new_fg));

    return node->func_graph()->NewCNode(new_xs);
  }

 private:
  internal::SpecializeTransform specialize_transform_;
};

// Eliminate unused parameters.
// {G, Xs}
class UnusedParasEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    auto fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    MS_EXCEPTION_IF_NULL(fg);

    std::vector<AnfNodePtr> parameters = fg->parameters();
    size_t size = parameters.size();
    if (size != inputs.size() - 1) {
      return nullptr;
    }

    std::vector<AnfNodePtr> new_xs;
    std::vector<bool> keep_parameters;
    auto mng = fg->manager();
    MS_EXCEPTION_IF_NULL(mng);
    auto &node_users = mng->node_users();
    bool has_unused_para = false;
    for (size_t i = 0; i < size; ++i) {
      auto iter = node_users.find(parameters[i]);
      if (iter != node_users.end() && !iter->second.empty()) {
        keep_parameters.push_back(true);
        new_xs.push_back(inputs[i + 1]);
        continue;
      }
      keep_parameters.push_back(false);
      has_unused_para = true;
    }

    if (!has_unused_para) {
      return nullptr;
    }
    FuncGraphPtr new_fg = TransformableClone(fg, std::make_shared<TraceTransform>("sp"));
    mng->AddFuncGraph(new_fg);

    std::vector<AnfNodePtr> new_fg_parameters = new_fg->parameters();
    std::vector<AnfNodePtr> new_parameters;
    for (size_t i = 0; i < size; i++) {
      if (keep_parameters[i]) {
        if (parameters[i]->abstract() != nullptr) {
          new_fg_parameters[i]->set_abstract(parameters[i]->abstract());
        }
        new_parameters.push_back(new_fg_parameters[i]);
      }
    }
    mng->SetParameters(new_fg, new_parameters);

    (void)new_xs.insert(new_xs.begin(), NewValueNode(new_fg));
    return node->func_graph()->NewCNode(new_xs);
  }
};

// Eliminate unused outputs.
// {G, Xs}
class UnusedOutputEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    auto fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    MS_EXCEPTION_IF_NULL(fg);
    auto mng = fg->manager();
    MS_EXCEPTION_IF_NULL(mng);
    if (fg->recursive()) {
      return nullptr;
    }

    auto new_fg = TransformableClone(fg, std::make_shared<TraceTransform>("fg"));
    mng->AddFuncGraph(new_fg);
    auto new_fg_output = new_fg->output();
    if (!IsPrimitiveCNode(new_fg_output, prim::kPrimMakeTuple)) {
      return nullptr;
    }

    auto output_cnode = new_fg_output->cast<CNodePtr>();
    auto &node_users = mng->node_users();
    if (node_users.count(node) == 0 || node_users[node].empty()) {
      return nullptr;
    }
    std::unordered_set<int> used_output_idx;
    std::vector<std::pair<AnfNodePtr, int>> all_users;
    for (auto &node_user : node_users[node]) {
      if (!IsPrimitiveCNode(node_user.first, prim::kPrimTupleGetItem)) {
        return nullptr;
      }
      auto user_cnode = node_user.first->cast<CNodePtr>();
      size_t used_idx = GetValue<int>(user_cnode->input(2)->cast<ValueNodePtr>()->value());
      used_output_idx.insert(used_idx);
      all_users.push_back(std::make_pair(node_user.first, used_idx));
    }

    if (used_output_idx.size() >= output_cnode->inputs().size() - 1) {
      // all output has users.
      return nullptr;
    }

    if (used_output_idx.empty()) {
      // we do not process this case.
      return nullptr;
    } else if (used_output_idx.size() == 1) {
      // after eliminate, only one output left.
      new_fg->set_output(output_cnode->input(*used_output_idx.begin() + 1));
      // update users.
      for (auto &ret_user : all_users) {
        (void)mng->Replace(ret_user.first, node);
      }
    } else {
      // after eliminate, create new multi output.
      std::vector<AnfNodePtr> new_output_inputs{output_cnode->input(0)};
      std::unordered_map<int, int> new_idx_map;
      for (auto idx : used_output_idx) {
        new_idx_map[idx] = SizeToInt(new_output_inputs.size() - 1);
        new_output_inputs.push_back(output_cnode->input(idx + 1));
      }
      new_fg->set_output(new_fg->NewCNode(new_output_inputs));
      // update users.
      for (auto &ret_user : all_users) {
        auto ret_user_cnode = ret_user.first->cast<CNodePtr>();
        ret_user_cnode->set_input(2, NewValueNode(new_idx_map[ret_user.second]));
      }
    }

    auto new_sx = inputs;
    new_sx[0] = NewValueNode(new_fg);
    return node->func_graph()->NewCNode(new_sx);
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_SPECIALIZE_TRANSFORM_H_
