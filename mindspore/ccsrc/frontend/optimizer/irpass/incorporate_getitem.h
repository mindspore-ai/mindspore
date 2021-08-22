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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>
#include <utility>

#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
class GetitemTransform {
 public:
  GetitemTransform() : cache_() {}
  ~GetitemTransform() = default;

  FuncGraphPtr operator()(const AnfNodePtr &node, const FuncGraphPtr &fg, int64_t idx) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    if (cache.find(idx) == cache.end()) {
      std::ostringstream ss("tp", std::ostringstream::app);
      ss << idx;

      auto new_fg = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
      auto output = new_fg->output();
      if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
        auto cnode = output->cast<CNodePtr>();
        auto ids = LongToSize(idx + 1);
        // Inputs should be [make_tuple, item1, item2, ...], so have to offset idx in tuple_getitem by 1.
        if (ids >= cnode->size()) {
          MS_LOG(EXCEPTION) << "index " << ids << " is out of inputs length " << cnode->size();
        }
        new_fg->set_output(cnode->input(ids));
      } else {
        auto idx_node = NewValueNode(idx);
        idx_node->set_abstract(std::make_shared<abstract::AbstractScalar>(idx));
        auto output_node = new_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), output, idx_node});
        output_node->set_abstract(node->abstract());
        new_fg->set_output(output_node);
      }

      cache[idx] = new_fg;
    }
    return cache[idx];
  }

 private:
  std::unordered_map<FuncGraphPtr, std::unordered_map<int64_t, FuncGraphPtr>> cache_;
};

class GetItemTransformACrossGraph {
 public:
  GetItemTransformACrossGraph() : cache_() {}
  ~GetItemTransformACrossGraph() = default;

  FuncGraphPtr operator()(const AnfNodePtr &node, const FuncGraphPtr &fg, int64_t idx) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    if (cache.find(idx) == cache.end()) {
      std::ostringstream ss("tp", std::ostringstream::app);
      ss << idx;

      auto new_fg_outer = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
      MS_EXCEPTION_IF_NULL(fg->manager());
      fg->manager()->AddFuncGraph(new_fg_outer);
      auto output_outer = new_fg_outer->output();
      if (!IsValueNode<FuncGraph>(output_outer)) {
        MS_LOG(WARNING) << "Output of outer graph should be a func_graph";
        return nullptr;
      }
      auto fg_inner = GetValueNode<FuncGraphPtr>(output_outer);
      auto new_fg = TransformableClone(fg_inner, std::make_shared<TraceTransform>(ss.str()));
      new_fg_outer->set_output(NewValueNode(new_fg));
      auto output = new_fg->output();
      if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
        auto cnode = output->cast<CNodePtr>();
        auto ids = LongToSize(idx + 1);
        // Inputs should be [make_tuple, item1, item2, ...], so have to offset idx in tuple_getitem by 1.
        if (ids >= cnode->size()) {
          MS_LOG(EXCEPTION) << "index " << ids << " is out of inputs length " << cnode->size();
        }
        new_fg->set_output(cnode->input(ids));
      } else {
        auto idx_node = NewValueNode(idx);
        idx_node->set_abstract(std::make_shared<abstract::AbstractScalar>(idx));
        auto output_node = new_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), output, idx_node});
        output_node->set_abstract(node->abstract());
        new_fg->set_output(output_node);
      }

      cache[idx] = new_fg_outer;
    }
    return cache[idx];
  }

 private:
  std::unordered_map<FuncGraphPtr, std::unordered_map<int64_t, FuncGraphPtr>> cache_;
};
}  // namespace internal

// {prim::kPrimTupleGetItem, {G, Xs}, C}
class IncorporateGetitem : public AnfVisitor {
 public:
  IncorporateGetitem() : getitem_transform_() {}
  ~IncorporateGetitem() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    if (node->func_graph() == nullptr || idx_ == -1 || fg_ == nullptr || fg_->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) ||
        fg_->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      return nullptr;
    }

    auto new_fg = getitem_transform_(node, fg_, idx_);
    (void)args_.insert(args_.begin(), NewValueNode(new_fg));
    auto new_node = node->func_graph()->NewCNode(args_);
    // Check if the another only usage of {G, Xs} is UpdateState{s, {G, Xs}}, if yes, replace
    // UpdateState{s, {G, Xs}} with UpdateState{s, new_node};
    const auto &manager = fg_->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &node_users_map = manager->node_users();
    auto it = node_users_map.find(fg_cnode_);
    if (it != node_users_map.end()) {
      AnfNodePtr update_state_node = nullptr;
      auto &node_users = it->second;
      if (node_users.size() == 2) {
        for (auto &node_user : node_users) {
          if (IsPrimitiveCNode(node_user.first, prim::kPrimUpdateState)) {
            update_state_node = node_user.first;
          }
        }
      }
      if (update_state_node != nullptr) {
        auto update_state_cnode = update_state_node->cast<CNodePtr>();
        // double check;
        if (update_state_cnode->input(2) == fg_cnode_) {
          MS_LOG(DEBUG) << "Replace UpdateState node: " << update_state_cnode->DebugString(2)
                        << ", input 2 with: " << new_node->DebugString();
          manager->SetEdge(update_state_cnode, 2, new_node);
        }
      }
    }
    new_node->set_abstract(node->abstract());
    return new_node;
  }

  void Visit(const CNodePtr &cnode) override {
    if (cnode->size() == 0 || !IsValueNode<FuncGraph>(cnode->input(0))) {
      return;
    }

    fg_cnode_ = cnode;
    auto &inputs = cnode->inputs();
    fg_ = GetValueNode<FuncGraphPtr>(inputs[0]);
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));
  }

  void Visit(const ValueNodePtr &vnode) override { idx_ = GetValue<int64_t>(vnode->value()); }

  void Reset() {
    idx_ = -1;
    fg_ = nullptr;
    fg_cnode_ = nullptr;
    args_.clear();
  }

 private:
  int64_t idx_{-1};
  FuncGraphPtr fg_{nullptr};
  AnfNodePtr fg_cnode_{nullptr};
  std::vector<AnfNodePtr> args_{};
  internal::GetitemTransform getitem_transform_;
};

// A special case, cannot wait for TupleListGetitemDependReorder pass.
// {prim::kPrimTupleGetItem, {prim::kPrimDepend, {G, Xs}, {prim::kPrimUpdateState, Y, {G, Xs}}}, C} ->
// {prim::kPrimDepend, {tp_idx_G, Xs}, {prim::kPrimUpdateState, Y, {tp_idx_G, Xs}}} ->
class IncorporateGetitemDepend : public AnfVisitor {
 public:
  IncorporateGetitemDepend() : getitem_transform_() {}
  ~IncorporateGetitemDepend() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    if (node->func_graph() == nullptr || idx_ == -1 || fg_ == nullptr || fg_->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) ||
        fg_->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      return nullptr;
    }

    auto new_fg = getitem_transform_(node, fg_, idx_);
    (void)args_.insert(args_.begin(), NewValueNode(new_fg));
    auto new_fg_cnode = node->func_graph()->NewCNode(args_);
    AnfNodePtr new_depend_cnode;
    if (used_in_update_) {
      auto update_cnode = depend_2nd_input_->cast<CNodePtr>();
      AnfNodePtrList new_update_inputs;
      (void)std::copy(update_cnode->inputs().begin(), update_cnode->inputs().end() - 1,
                      std::back_inserter(new_update_inputs));
      new_update_inputs.push_back(new_fg_cnode);
      auto new_update_cnode = node->func_graph()->NewCNode(new_update_inputs);
      new_depend_cnode =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimDepend), new_fg_cnode, new_update_cnode});
    } else {
      new_depend_cnode =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimDepend), new_fg_cnode, depend_2nd_input_});
    }
    new_depend_cnode->set_abstract(node->abstract());
    return new_depend_cnode;
  }

  void Visit(const CNodePtr &cnode) override {
    // cnode : {kPrimDepend, {G, Xs}, {kPrimUpdatestate, Y, {G, Xs}}}
    if (!IsPrimitiveCNode(cnode, prim::kPrimDepend)) {
      return;
    }
    if (cnode->size() != 3 || !IsCNode(cnode->input(1))) {
      return;
    }
    depend_2nd_input_ = cnode->input(2);

    auto fg_cnode = cnode->input(1)->cast<CNodePtr>();
    // fg_cnode : {G, Xs}
    if (!IsValueNode<FuncGraph>(fg_cnode->input(0))) {
      return;
    }

    auto &inputs = fg_cnode->inputs();
    fg_ = GetValueNode<FuncGraphPtr>(inputs[0]);
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));

    if (!IsPrimitiveCNode(depend_2nd_input_, prim::kPrimUpdateState)) {
      return;
    }
    auto update_cnode = depend_2nd_input_->cast<CNodePtr>();
    if (update_cnode->size() != 3) {
      return;
    }
    // match {kPrimUpdateState, Y, {G, Xs}}
    if (update_cnode->input(2) == fg_cnode) {
      used_in_update_ = true;
    }
  }

  void Visit(const ValueNodePtr &vnode) override { idx_ = GetValue<int64_t>(vnode->value()); }

  void Reset() {
    idx_ = -1;
    fg_ = nullptr;
    args_.clear();
  }

 private:
  bool used_in_update_{false};
  int64_t idx_{-1};
  FuncGraphPtr fg_{nullptr};
  AnfNodePtr depend_2nd_input_{nullptr};
  std::vector<AnfNodePtr> args_{};
  internal::GetitemTransform getitem_transform_;
};

// {prim::kPrimTupleGetItem, {{prim::kPrimSwitch, X, G1, G2}, Xs}, C}
class IncorporateGetitemSwitch : public AnfVisitor {
 public:
  IncorporateGetitemSwitch() : getitem_transform_() {}
  ~IncorporateGetitemSwitch() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    is_in_get_ = true;
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    is_in_get_ = false;

    auto fg = node->func_graph();
    if (idx_ == -1 || switch_ == nullptr || fg == nullptr) {
      return nullptr;
    }

    is_in_switch_ = true;
    AnfVisitor::Match(prim::kPrimSwitch, {IsNode, IsValueNode<FuncGraph>, IsValueNode<FuncGraph>})(switch_);
    is_in_switch_ = false;

    if (g2_ == nullptr) {
      return nullptr;
    }
    auto tuple_getitem = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    bool has_env_type = false;
    if (tuple_getitem->input(1)->abstract() && tuple_getitem->input(1)->abstract()->isa<abstract::AbstractTuple>()) {
      const auto &abs_tuple = *(tuple_getitem->input(1)->abstract()->cast<abstract::AbstractTuplePtr>());
      // eliminate (envinstance, value1, value2, ...) built by bprop func_graph()
      if (abs_tuple.size() >= 1) {
        // Value maybe kAnyValue, so check the type track;
        if (abs_tuple[0]->isa<abstract::AbstractScalar>() && abs_tuple[0]->GetTypeTrack()->isa<EnvType>()) {
          has_env_type = true;
        }
      }
      // eliminate (value, bprop_func) built by fprop func_graph
      if (abs_tuple.size() >= 2) {
        if (abs_tuple[1]->isa<abstract::AbstractFunction>()) {
          has_env_type = true;
        }
      }
    }
    // If exist env_getitem/env_setitem in this funcgraph or
    // if g1_/g2_ is fprop func_graph and the corresponding bprop funcgraph has any env_getitem or env_setitem;
    if (MultipleUseOfSwitch(tuple_getitem->input(1), fg) && !ExistEnvNode(fg) && !ExistEnvNodeInTupleItem(g1_) &&
        !ExistEnvNodeInTupleItem(g2_) && !has_env_type) {
      return nullptr;
    }
    auto new_g1 = getitem_transform_(node, g1_, idx_);
    auto new_g2 = getitem_transform_(node, g2_, idx_);
    auto sw_node = fg->NewCNode({NewValueNode(prim::kPrimSwitch), x_, NewValueNode(new_g1), NewValueNode(new_g2)});
    (void)args_.insert(args_.begin(), sw_node);

    auto new_node = fg->NewCNode(args_);
    new_node->set_abstract(node->abstract());
    return new_node;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_in_switch_ && x_ == nullptr) {
      x_ = node;
      return;
    }
    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    if (is_in_get_ && cnode->size() != 0) {
      auto &inputs = cnode->inputs();
      switch_ = inputs[0];
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (is_in_get_) {
      idx_ = GetValue<int64_t>(vnode->value());
    }

    if (is_in_switch_) {
      auto g = GetValueNode<FuncGraphPtr>(vnode);
      if (g1_ == nullptr) {
        g1_ = g;
      } else {
        g2_ = g;
      }
    }
  }

  void Reset() {
    x_ = nullptr;
    g1_ = nullptr;
    g2_ = nullptr;
    switch_ = nullptr;
    args_.clear();
    is_in_get_ = false;
    is_in_switch_ = false;
  }

 private:
  bool MultipleUseOfSwitch(const AnfNodePtr &switch_call, const FuncGraphPtr &fg) const {
    auto switch_call_cnode = switch_call->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_call_cnode);
    auto manager = fg->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &node_users_map = manager->node_users();
    auto it = node_users_map.find(switch_call);
    if (it == node_users_map.end()) {
      return false;
    }
    auto &node_users = it->second;
    // If switch was used by more than 1 tuple_getitem nodes, this pass shouldn't be execute.s
    auto tuple_getitem_num = std::count_if(node_users.begin(), node_users.end(), [](std::pair<AnfNodePtr, int> &user) {
      return IsPrimitiveCNode(user.first, prim::kPrimTupleGetItem);
    });
    return tuple_getitem_num > 1;
  }

  static bool inline ExistEnvNode(const FuncGraphPtr &fg) {
    MS_EXCEPTION_IF_NULL(fg);
    auto &nodes = fg->value_nodes();
    return std::any_of(nodes.begin(), nodes.end(), [](const auto &node) {
      return IsPrimitive(node.first, prim::kPrimEnvSetItem) || IsPrimitive(node.first, prim::kPrimEnvGetItem);
    });
  }

  static bool inline ExistEnvNodeInTupleItem(const FuncGraphPtr &fg) {
    MS_EXCEPTION_IF_NULL(fg);
    const auto &output = fg->output();
    if (!IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
      return false;
    }
    const auto &cnode = output->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    return std::any_of(inputs.cbegin() + 1, inputs.cend(), [](const auto &input) {
      auto sub_fg = GetValueNode<FuncGraphPtr>(input);
      if (sub_fg != nullptr && ExistEnvNode(sub_fg)) {
        return true;
      }
      return false;
    });
  }

  int64_t idx_{-1};
  AnfNodePtr switch_{nullptr}, x_{nullptr};
  FuncGraphPtr g1_{nullptr}, g2_{nullptr};
  bool is_in_get_{false}, is_in_switch_{false};
  std::vector<AnfNodePtr> args_{};
  internal::GetitemTransform getitem_transform_;
};

// {prim::kPrimTupleGetItem, {{prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, G1, G2...}}, Xs}, C}
class IncorporateGetitemSwitchLayerA : public AnfVisitor {
 public:
  IncorporateGetitemSwitchLayerA() : getitem_transform_() {}
  ~IncorporateGetitemSwitchLayerA() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    is_in_get_ = true;
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    is_in_get_ = false;

    auto fg = node->func_graph();
    if (idx_ == -1 || switch_layer_ == nullptr || fg == nullptr) {
      return nullptr;
    }

    is_in_switch_ = true;
    AnfVisitor::Match(prim::kPrimSwitchLayer, {IsNode, IsCNode})(switch_layer_);
    is_in_switch_ = false;

    if (graphs_.empty()) {
      return nullptr;
    }

    std::vector<AnfNodePtr> layers;
    for (auto &graph : graphs_) {
      auto fg_transform = getitem_transform_(node, graph, idx_);
      if (fg_transform == nullptr) {
        return nullptr;
      }
      layers.push_back(NewValueNode(fg_transform));
    }
    auto layers_node = fg->NewCNode(prim::kPrimMakeTuple, layers);
    std::vector<AnfNodePtr> sw_args{NewValueNode(prim::kPrimSwitchLayer), x_, layers_node};
    auto sw_node = fg->NewCNode(sw_args);
    (void)args_.insert(args_.begin(), sw_node);

    return fg->NewCNode(args_);
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_in_switch_ && x_ == nullptr) {
      x_ = node;
      return;
    }
    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    if (is_in_get_ && cnode->size() != 0) {
      auto &inputs = cnode->inputs();
      switch_layer_ = inputs[0];
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));
    }
    if (is_in_switch_ && cnode->size() >= 2) {
      auto &inputs = cnode->inputs();
      if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) && IsValueNode<FuncGraph>(inputs[1])) {
        (void)std::transform(inputs.begin() + 1, inputs.end(), std::back_inserter(graphs_),
                             [](const AnfNodePtr &vnode) { return GetValueNode<FuncGraphPtr>(vnode); });
      }
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (is_in_get_) {
      idx_ = GetValue<int64_t>(vnode->value());
    }
  }

  void Reset() {
    x_ = nullptr;
    graphs_.clear();
    switch_layer_ = nullptr;
    args_.clear();
    is_in_get_ = false;
    is_in_switch_ = false;
  }

 private:
  int64_t idx_{-1};
  AnfNodePtr switch_layer_{nullptr}, x_{nullptr};
  std::vector<FuncGraphPtr> graphs_{};
  bool is_in_get_{false}, is_in_switch_{false};
  std::vector<AnfNodePtr> args_{};
  internal::GetitemTransform getitem_transform_;
};

// {prim::kPrimTupleGetItem, {{{prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, G1, G2...}}, Xs}, Ys}, C}
class IncorporateGetitemSwitchLayerB : public AnfVisitor {
 public:
  IncorporateGetitemSwitchLayerB() : getitem_transform_() {}
  ~IncorporateGetitemSwitchLayerB() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    is_in_get_ = true;
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    is_in_get_ = false;

    auto fg = node->func_graph();
    if (idx_ == -1 || switch_layer_call_ == nullptr || !switch_layer_call_->isa<CNode>() || fg == nullptr) {
      return nullptr;
    }

    auto &switch_layer_call_inputs = switch_layer_call_->cast<CNodePtr>()->inputs();
    (void)std::copy(switch_layer_call_inputs.begin() + 1, switch_layer_call_inputs.end(), std::back_inserter(args_));

    is_in_switch_ = true;
    AnfVisitor::Match(prim::kPrimSwitchLayer, {IsNode, IsCNode})(switch_layer_call_inputs[0]);
    is_in_switch_ = false;

    if (graphs_.empty()) {
      return nullptr;
    }

    std::vector<AnfNodePtr> layers;
    for (auto &graph : graphs_) {
      auto fg_transform = getitem_transform_(node, graph, idx_);
      if (fg_transform == nullptr) {
        return nullptr;
      }
      layers.push_back(NewValueNode(fg_transform));
    }
    auto layers_node = fg->NewCNode(prim::kPrimMakeTuple, layers);
    std::vector<AnfNodePtr> sw_args{NewValueNode(prim::kPrimSwitchLayer), x_, layers_node};
    auto sw_node = fg->NewCNode(sw_args);
    (void)args_.insert(args_.begin(), sw_node);
    auto call_switch_layer = fg->NewCNode(args_);
    (void)outer_call_args_.insert(outer_call_args_.begin(), call_switch_layer);
    return fg->NewCNode(outer_call_args_);
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_in_switch_ && x_ == nullptr) {
      x_ = node;
      return;
    }
    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    if (is_in_get_ && cnode->size() != 0) {
      auto &inputs = cnode->inputs();
      switch_layer_call_ = inputs[0];
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(outer_call_args_));
    }
    if (is_in_switch_ && cnode->size() >= 2) {
      auto &inputs = cnode->inputs();
      if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) && IsValueNode<FuncGraph>(inputs[1])) {
        (void)std::transform(inputs.begin() + 1, inputs.end(), std::back_inserter(graphs_),
                             [](const AnfNodePtr &vnode) { return GetValueNode<FuncGraphPtr>(vnode); });
      }
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (is_in_get_) {
      idx_ = GetValue<int64_t>(vnode->value());
    }
  }

  void Reset() {
    x_ = nullptr;
    graphs_.clear();
    switch_layer_call_ = nullptr;
    args_.clear();
    outer_call_args_.clear();
    is_in_get_ = false;
    is_in_switch_ = false;
  }

 private:
  int64_t idx_{-1};
  AnfNodePtr switch_layer_call_{nullptr}, x_{nullptr};
  std::vector<FuncGraphPtr> graphs_{};
  bool is_in_get_{false}, is_in_switch_{false};
  std::vector<AnfNodePtr> args_{};
  std::vector<AnfNodePtr> outer_call_args_{};
  internal::GetItemTransformACrossGraph getitem_transform_;
};

class IncorporateGetitemSet : public OptimizerCaller {
 public:
  IncorporateGetitemSet()
      : incorporate_getitem_(std::make_shared<IncorporateGetitem>()),
        incorporate_getitem_depend_(std::make_shared<IncorporateGetitemDepend>()),
        incorporate_getitem_switch_(std::make_shared<IncorporateGetitemSwitch>()),
        incorporate_getitem_switch_layer_a_(std::make_shared<IncorporateGetitemSwitchLayerA>()),
        incorporate_getitem_switch_layer_b_(std::make_shared<IncorporateGetitemSwitchLayerB>()) {
    eliminaters_.emplace_back(incorporate_getitem_);
    eliminaters_.emplace_back(incorporate_getitem_depend_);
    eliminaters_.emplace_back(incorporate_getitem_switch_);
    eliminaters_.emplace_back(incorporate_getitem_switch_layer_a_);
    eliminaters_.emplace_back(incorporate_getitem_switch_layer_b_);
  }
  ~IncorporateGetitemSet() = default;

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
  OptimizerCallerPtr incorporate_getitem_, incorporate_getitem_depend_, incorporate_getitem_switch_,
    incorporate_getitem_switch_layer_a_, incorporate_getitem_switch_layer_b_;
  std::vector<OptimizerCallerPtr> eliminaters_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_
