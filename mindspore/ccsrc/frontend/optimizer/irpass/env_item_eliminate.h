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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ENV_ITEM_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ENV_ITEM_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/inline.h"
#include "frontend/optimizer/optimizer.h"
#include "utils/symbolic.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
class EnvGetitemTransform {
 public:
  EnvGetitemTransform() : cache_() {}
  ~EnvGetitemTransform() = default;

  FuncGraphPtr operator()(const FuncGraphPtr &fg, const SymbolicKeyInstancePtr &key, const AnfNodePtr &default_node) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    auto hash_key = std::make_pair(key, default_node);
    if (cache.find(hash_key) == cache.end()) {
      std::ostringstream ss("env", std::ostringstream::app);
      if (key->node() != nullptr) {
        ss << key->node()->ToString();
      }

      auto new_fg = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
      auto env = new_fg->output();
      while (IsPrimitiveCNode(env, prim::kPrimEnvSetItem)) {
        // {prim::kPrimEnvSetItem, env, symbolickey, value}
        auto &inputs = env->cast<CNodePtr>()->inputs();
        if (inputs.size() != 4) {
          MS_LOG(WARNING) << "Input size should be 4";
          return nullptr;
        }
        if (!IsValueNode<SymbolicKeyInstance>(inputs[2])) {
          MS_LOG(DEBUG) << "Input 2 is not a SymbolicKeyInstance?";
          return nullptr;
        }

        env = inputs[1];
        auto value = inputs[3];
        auto key2 = GetValueNode<SymbolicKeyInstancePtr>(inputs[2]);
        if (*key2 == *key) {
          new_fg->set_output(value);
          cache[hash_key] = new_fg;
          cache_[fg] = cache;
          return new_fg;
        }
      }
      new_fg->set_output(new_fg->NewCNode({NewValueNode(prim::kPrimEnvGetItem), env, NewValueNode(key), default_node}));
      cache[hash_key] = new_fg;
    }

    return cache[hash_key];
  }

 private:
  std::unordered_map<FuncGraphPtr,
                     std::unordered_map<std::pair<SymbolicKeyInstancePtr, AnfNodePtr>, FuncGraphPtr, PairHasher>>
    cache_;
};

class EnvGetitemTransformACrossGraph {
 public:
  EnvGetitemTransformACrossGraph() : cache_() {}
  ~EnvGetitemTransformACrossGraph() = default;

  FuncGraphPtr operator()(const FuncGraphPtr &fg, const SymbolicKeyInstancePtr &key, const AnfNodePtr &default_node) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    auto hash_key = std::make_pair(key, default_node);
    if (cache.find(hash_key) == cache.end()) {
      std::ostringstream ss("env", std::ostringstream::app);
      if (key->node() != nullptr) {
        ss << key->node()->ToString();
      }

      auto new_fg_outer = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
      auto output_outer = new_fg_outer->output();
      if (!IsValueNode<FuncGraph>(output_outer)) {
        MS_LOG(WARNING) << "Output of outer graph should be a func_graph";
        return nullptr;
      }
      auto fg_inner = GetValueNode<FuncGraphPtr>(output_outer);
      auto new_fg = TransformableClone(fg_inner, std::make_shared<TraceTransform>(ss.str()));
      new_fg_outer->set_output(NewValueNode(new_fg));

      auto env = new_fg->output();
      while (IsPrimitiveCNode(env, prim::kPrimEnvSetItem)) {
        // {prim::kPrimEnvSetItem, env, symbolickey, value}
        auto &inputs = env->cast<CNodePtr>()->inputs();
        if (inputs.size() != 4) {
          MS_LOG(WARNING) << "Input size should be 4";
          return nullptr;
        }
        if (!IsValueNode<SymbolicKeyInstance>(inputs[2])) {
          MS_LOG(DEBUG) << "Input 2 is not a SymbolicKeyInstance?";
          return nullptr;
        }

        env = inputs[1];
        auto value = inputs[3];
        auto key2 = GetValueNode<SymbolicKeyInstancePtr>(inputs[2]);
        if (*key2 == *key) {
          new_fg->set_output(value);
          cache[hash_key] = new_fg_outer;
          return new_fg_outer;
        }
      }
      new_fg->set_output(new_fg->NewCNode({NewValueNode(prim::kPrimEnvGetItem), env, NewValueNode(key), default_node}));
      cache[hash_key] = new_fg_outer;
    }

    return cache[hash_key];
  }

 private:
  std::unordered_map<FuncGraphPtr,
                     std::unordered_map<std::pair<SymbolicKeyInstancePtr, AnfNodePtr>, FuncGraphPtr, PairHasher>>
    cache_;
};
}  // namespace internal

// {prim::kPrimEnvGetItem, C1, C2, Y} -> Y
class NewEnvGetItem : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode c1, c2, y;
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimEnvGetItem, c1, c2, y), y,
                     (IsValueNode<EnvInstance>(c1.GetNode(node)) && IsVNode(c2.GetNode(node)) &&
                      (GetValueNode<EnvInstancePtr>(c1.GetNode(node)))->Len() == 0));
    return nullptr;
  }
};

// {prim::kPrimEnvGetItem, {prim::kPrimEnvAdd, X, Y}, C, Z} ->
// {prim::GetPythonOps("hyper_add"), {prim::kPrimEnvGetItem, X, C, Z}, {prim::kPrimEnvGetItem, Y, C, Z}}
class AddEnvGetItem : public AnfVisitor {
 public:
  AddEnvGetItem() : PrimHyperAdd_(prim::GetPythonOps("hyper_add")) {}
  ~AddEnvGetItem() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    auto IsAddCNode = [](const AnfNodePtr &node) -> bool {
      return IsPrimitiveCNode(node, prim::kPrimEnvAdd) && node->cast<CNodePtr>()->size() == 3;
    };
    AnfVisitor::Match(prim::kPrimEnvGetItem, {IsAddCNode, IsVNode, IsNode})(node);

    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {prim::kPrimEnvGetItem, {...}, C, Z}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto c = cnode->input(2);
    auto z = cnode->input(3);

    // {prim::kPrimEnvAdd, X, Y}
    auto x = inp1->input(1);
    auto y = inp1->input(2);

    auto fg = node->func_graph();
    auto xcz = fg->NewCNode({NewValueNode(prim::kPrimEnvGetItem), x, c, z});
    auto ycz = fg->NewCNode({NewValueNode(prim::kPrimEnvGetItem), y, c, z});

    return fg->NewCNode({NewValueNode(PrimHyperAdd_), xcz, ycz});
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
  ValuePtr PrimHyperAdd_;
};

// {prim::kPrimEnvGetItem, {prim::kPrimEnvSetItem, X, C1, Y}, C2, Z}
class EnvGetSetItem : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    auto IsSetCNode = [](const AnfNodePtr &node) -> bool {
      if (!IsPrimitiveCNode(node, prim::kPrimEnvSetItem)) {
        return false;
      }

      // {prim::kPrimEnvSetItem, X, C1, Y}
      auto &inputs = node->cast<CNodePtr>()->inputs();
      if (inputs.size() != 4) {
        return false;
      }

      return IsValueNode<SymbolicKeyInstance>(inputs[2]);
    };
    AnfVisitor::Match(prim::kPrimEnvGetItem, {IsSetCNode, IsValueNode<SymbolicKeyInstance>, IsNode})(node);

    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {prim::kPrimEnvGetItem, {...}, C2, Z}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto key2 = cnode->input(2);
    auto c2 = GetValueNode<SymbolicKeyInstancePtr>(key2);
    auto default_v = cnode->input(3);

    // {prim::kPrimEnvSetItem, X, C1, Y}
    auto env = inp1->input(1);
    auto c1 = GetValueNode<SymbolicKeyInstancePtr>(inp1->input(2));
    auto last_set = inp1->input(3);

    if (*c1 == *c2) {
      return last_set;
    }

    while (IsPrimitiveCNode(env, prim::kPrimEnvSetItem)) {
      // {prim::kPrimEnvSetItem, env, symbolickey, value}
      auto &inputs = env->cast<CNodePtr>()->inputs();
      if (inputs.size() != 4) {
        MS_LOG(WARNING) << "Input size should be 4";
        return nullptr;
      }
      if (!IsValueNode<SymbolicKeyInstance>(inputs[2])) {
        MS_LOG(DEBUG) << "Input 2 is not a SymbolicKeyInstance?";
        return nullptr;
      }

      env = inputs[1];
      last_set = inputs[3];
      auto symbolic_c1 = GetValueNode<SymbolicKeyInstancePtr>(inputs[2]);
      if (*symbolic_c1 == *c2) {
        return last_set;
      }
    }

    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimEnvGetItem), env, key2, default_v});
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
};

// {prim::kPrimEnvGetitem, {prim::kPrimDepend, X1, X2}, item, dflt} ->
// {prim::kPrimDepend, {prim::kPrimEnvGetitem, X1, item, dflt}, X2}
class SwapEnvGetItemDepend : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }
    ScopePtr scope = node->cast<CNodePtr>()->scope();
    ScopeGuard scope_guard(scope);

    PatternNode x1, x2, item, dflt;
    MATCH_REPLACE(node, PPrimitive(prim::kPrimEnvGetItem, PPrimitive(prim::kPrimDepend, x1, x2), item, dflt),
                  PPrimitive(prim::kPrimDepend, PPrimitive(prim::kPrimEnvGetItem, x1, item, dflt), x2));
    return nullptr;
  }
};

class EnvGetItemEliminater : public OptimizerCaller {
 public:
  EnvGetItemEliminater()
      : new_env_get_item_(std::make_shared<NewEnvGetItem>()),
        add_env_get_item_(std::make_shared<AddEnvGetItem>()),
        env_get_set_item_(std::make_shared<EnvGetSetItem>()),
        swap_env_get_item_depend_(std::make_shared<SwapEnvGetItemDepend>()) {
    eliminaters_.emplace_back(new_env_get_item_);
    eliminaters_.emplace_back(add_env_get_item_);
    eliminaters_.emplace_back(env_get_set_item_);
    eliminaters_.emplace_back(swap_env_get_item_depend_);
  }
  ~EnvGetItemEliminater() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    AnfNodePtr new_node;
    for (auto &eliminater : eliminaters_) {
      new_node = (*eliminater)(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  OptimizerCallerPtr new_env_get_item_, add_env_get_item_, env_get_set_item_, swap_env_get_item_depend_;
  std::vector<OptimizerCallerPtr> eliminaters_{};
};

// {prim::kPrimEnvGetItem, {G, Xs}, C, Y}
class IncorporateEnvGetitem : public AnfVisitor {
 public:
  explicit IncorporateEnvGetitem(bool bypass_recursive = false)
      : env_get_item_transform_(), bypass_recursive_(bypass_recursive) {}
  ~IncorporateEnvGetitem() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    auto IsGCNode = [](const AnfNodePtr &node) -> bool {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr || cnode->size() < 1) {
        return false;
      }
      return IsValueNode<FuncGraph>(cnode->input(0));
    };
    AnfVisitor::Match(prim::kPrimEnvGetItem, {IsGCNode, IsValueNode<SymbolicKeyInstance>, IsNode})(node);

    if (!is_match_) {
      return nullptr;
    }

    // {prim::kPrimEnvGetItem, {...}, C, Y}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto key = GetValueNode<SymbolicKeyInstancePtr>(cnode->input(2));
    auto default_v = cnode->input(3);

    // {G, Xs}
    auto inputs = inp1->inputs();
    auto fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    auto new_fg = env_get_item_transform_(fg, key, default_v);
    if (fg->recursive() && bypass_recursive_) {
      MS_LOG(DEBUG) << "Bypass env_get_item transform for recursive fg=" << fg->ToString();
      return nullptr;
    }
    if (new_fg == nullptr) {
      return nullptr;
    }
    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(new_fg));
    (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());

    return node->func_graph()->NewCNode(args);
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
  internal::EnvGetitemTransform env_get_item_transform_;
  bool bypass_recursive_;
};

// {prim::kPrimEnvGetItem, {{prim::kPrimSwitch, X, G1, G2}, Xs}, C, Y}
class IncorporateEnvGetitemSwitch : public AnfVisitor {
 public:
  IncorporateEnvGetitemSwitch() : env_get_item_transform_() {}
  ~IncorporateEnvGetitemSwitch() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    auto IsSwNode = [](const AnfNodePtr &node) -> bool {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr || cnode->size() < 1) {
        return false;
      }

      return IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch);
    };
    AnfVisitor::Match(prim::kPrimEnvGetItem, {IsSwNode, IsValueNode<SymbolicKeyInstance>, IsNode})(node);
    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {prim::kPrimEnvGetItem, {...}, C, Y}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto key = GetValueNode<SymbolicKeyInstancePtr>(cnode->input(2));
    auto default_v = cnode->input(3);

    // {{prim::kPrimSwitch, X, G1, G2}, Xs}
    auto inputs = inp1->inputs();
    is_match_ = false;
    AnfVisitor::Match(prim::kPrimSwitch, {IsNode, IsValueNode<FuncGraph>, IsValueNode<FuncGraph>})(inputs[0]);
    if (!is_match_) {
      return nullptr;
    }

    // {prim::kPrimSwitch, X, G1, G2}
    auto sw = inputs[0]->cast<CNodePtr>();
    auto x = sw->input(1);
    auto g1 = GetValueNode<FuncGraphPtr>(sw->input(2));
    auto g2 = GetValueNode<FuncGraphPtr>(sw->input(3));
    auto new_g1 = env_get_item_transform_(g1, key, default_v);
    auto new_g2 = env_get_item_transform_(g2, key, default_v);
    if (new_g1 == nullptr || new_g2 == nullptr) {
      return nullptr;
    }
    auto fg = node->func_graph();
    auto new_sw = fg->NewCNode({NewValueNode(prim::kPrimSwitch), x, NewValueNode(new_g1), NewValueNode(new_g2)});

    std::vector<AnfNodePtr> args{new_sw};
    (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());

    return fg->NewCNode(args);
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
  internal::EnvGetitemTransform env_get_item_transform_;
};

// {prim::kPrimEnvGetItem, {{{prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, G1, G2...}}, Xs}, Ys}, C, Y}
class IncorporateEnvGetitemSwitchLayer : public AnfVisitor {
 public:
  IncorporateEnvGetitemSwitchLayer() : env_get_item_transform_() {}
  ~IncorporateEnvGetitemSwitchLayer() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    AnfVisitor::Match(prim::kPrimEnvGetItem, {IsCNode, IsValueNode<SymbolicKeyInstance>, IsNode})(node);
    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }
    // {prim::kPrimEnvGetItem, {...}, C, Y}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    auto key = GetValueNode<SymbolicKeyInstancePtr>(cnode->input(2));
    auto default_v = cnode->input(3);

    // {{prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, G1, G2...}}, Xs}, Ys}
    auto &inputs_outer = inp1->inputs();
    if (!inputs_outer[0]->isa<CNode>()) {
      return nullptr;
    }
    std::vector<AnfNodePtr> args_outer;
    args_outer.insert(args_outer.end(), inputs_outer.begin() + 1, inputs_outer.end());
    auto &input_switch_layer = inputs_outer[0]->cast<CNodePtr>()->inputs();

    is_match_ = false;
    AnfVisitor::Match(prim::kPrimSwitchLayer, {IsNode, IsCNode})(input_switch_layer[0]);
    if (!is_match_) {
      return nullptr;
    }
    std::vector<AnfNodePtr> args;
    (void)args.insert(args.end(), input_switch_layer.begin() + 1, input_switch_layer.end());

    // {prim::kPrimSwitchLayers, X, {prim::kPrimMakeTuple, G1, G2...}}
    auto sw = input_switch_layer[0]->cast<CNodePtr>();
    std::vector<FuncGraphPtr> graphs{};
    auto graphs_cnode = sw->input(2)->cast<CNodePtr>();
    auto &graphs_inputs = graphs_cnode->inputs();
    if (IsPrimitiveCNode(graphs_cnode, prim::kPrimMakeTuple) && graphs_inputs.size() >= 2 &&
        IsValueNode<FuncGraph>(graphs_inputs[1])) {
      (void)std::transform(graphs_inputs.begin() + 1, graphs_inputs.end(), std::back_inserter(graphs),
                           [](const AnfNodePtr &vnode) { return GetValueNode<FuncGraphPtr>(vnode); });
    }
    if (graphs.empty()) {
      return nullptr;
    }

    auto fg = node->func_graph();
    std::vector<AnfNodePtr> layers;
    for (auto &graph : graphs) {
      auto fg_transform = env_get_item_transform_(graph, key, default_v);
      if (fg_transform == nullptr) {
        return nullptr;
      }
      layers.push_back(NewValueNode(fg_transform));
    }
    auto layers_node = fg->NewCNode(prim::kPrimMakeTuple, layers);
    auto new_sw = fg->NewCNode({NewValueNode(prim::kPrimSwitchLayer), sw->input(1), layers_node});
    args.insert(args.begin(), new_sw);
    auto inner_call = fg->NewCNode(args);
    args_outer.insert(args_outer.begin(), inner_call);
    return fg->NewCNode(args_outer);
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
  internal::EnvGetitemTransformACrossGraph env_get_item_transform_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ENV_ITEM_ELIMINATE_H_
