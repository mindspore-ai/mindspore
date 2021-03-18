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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
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

  FuncGraphPtr operator()(const FuncGraphPtr &fg, int64_t idx) {
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
        new_fg->set_output(new_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), output, NewValueNode(idx)}));
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

  FuncGraphPtr operator()(const FuncGraphPtr &fg, int64_t idx) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    if (cache.find(idx) == cache.end()) {
      std::ostringstream ss("tp", std::ostringstream::app);
      ss << idx;

      auto new_fg_outer = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
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
        new_fg->set_output(new_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), output, NewValueNode(idx)}));
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
    if (node->func_graph() == nullptr || idx_ == -1 || fg_ == nullptr || fg_->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE)) {
      return nullptr;
    }

    if (fg_->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      // If graph kernel has muti output, do not split.
      // some graph kernel output has EnvInstance node or DeadCode node should split.
      auto output = fg_->output();
      if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
        auto output_cnode = output->cast<CNodePtr>();
        auto outputs = output_cnode->inputs();
        int64_t real_output_cnt = 0;
        for (size_t i = 1; i < outputs.size(); ++i) {
          if (IsCNode(outputs[i]) || IsValueNode<tensor::Tensor>(outputs[i]) || IsParam(outputs[i])) {
            real_output_cnt++;
            if (real_output_cnt > 1) {
              return nullptr;
            }
          }
        }
      }
    }

    auto new_fg = getitem_transform_(fg_, idx_);
    (void)args_.insert(args_.begin(), NewValueNode(new_fg));
    return node->func_graph()->NewCNode(args_);
  }

  void Visit(const CNodePtr &cnode) override {
    if (cnode->size() == 0 || !IsValueNode<FuncGraph>(cnode->input(0))) {
      return;
    }

    auto &inputs = cnode->inputs();
    fg_ = GetValueNode<FuncGraphPtr>(inputs[0]);
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));
  }

  void Visit(const ValueNodePtr &vnode) override { idx_ = GetValue<int64_t>(vnode->value()); }

  void Reset() {
    idx_ = -1;
    fg_ = nullptr;
    args_.clear();
  }

 private:
  int64_t idx_{-1};
  FuncGraphPtr fg_{nullptr};
  std::vector<AnfNodePtr> args_{};
  internal::GetitemTransform getitem_transform_;
};

class IncorporateGetitemFromParam : public AnfVisitor {
 public:
  void Process(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const AnfNodePtr &param, size_t input_idx) {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    auto &node_users = mng->node_users();
    if (node_users.find(param) == node_users.end() || node_users[param].empty()) {
      args_.push_back(cnode->input(input_idx + 1));
      return;
    }

    for (auto &user : node_users[param]) {
      if (!IsPrimitiveCNode(user.first, prim::kPrimTupleGetItem)) {
        // we do not process this case.
        args_.push_back(cnode->input(input_idx + 1));
        return;
      }
    }

    // update new args.
    if (IsPrimitiveCNode(cnode->input(input_idx + 1), prim::kPrimMakeTuple)) {
      // case 1
      replace_parameters_[input_idx] = true;
      need_update_ = true;
      auto make_tuple_cnode = cnode->input(input_idx + 1)->cast<CNodePtr>();
      auto &make_tuple_cnode_inputs = make_tuple_cnode->inputs();
      inputs_num_[input_idx] = make_tuple_cnode_inputs.size() - 1;
      args_.insert(args_.end(), make_tuple_cnode_inputs.begin() + 1, make_tuple_cnode_inputs.end());
    } else {
      // case 2
      auto prev_cnode = cnode->input(input_idx + 1)->cast<CNodePtr>();
      auto prev_fg = GetValueNode<FuncGraphPtr>(prev_cnode->input(0));
      auto fg_output = prev_fg->output();
      if (!IsPrimitiveCNode(fg_output, prim::kPrimMakeTuple)) {
        MS_LOG(ERROR) << "The return of: " << prev_fg->ToString()
                      << " should be a make tuple, but got: " << fg_output->DebugString();
        return;
      }
      replace_parameters_[input_idx] = true;
      need_update_ = true;
      auto make_tuple_cnode = fg_output->cast<CNodePtr>();
      inputs_num_[input_idx] = make_tuple_cnode->inputs().size() - 1;
      for (size_t output_i = 0; output_i < inputs_num_[input_idx]; ++output_i) {
        auto new_getitem =
          func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), prev_cnode, NewValueNode(SizeToLong(output_i))});
        auto aptr = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(SizeToLong(output_i)));
        new_getitem->input(2)->set_abstract(aptr);
        new_getitem->set_abstract(make_tuple_cnode->input(output_i + 1)->abstract());
        args_.push_back(new_getitem);
      }
    }
  }

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (node->func_graph() == nullptr) {
      return nullptr;
    }

    Reset();

    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      return nullptr;
    }
    auto &inputs = cnode->inputs();
    auto fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    if (fg == nullptr) {
      return nullptr;
    }
    auto mng = fg->manager();
    MS_EXCEPTION_IF_NULL(mng);
    auto parameters = fg->parameters();
    if (parameters.size() != inputs.size() - 1) {
      return nullptr;
    }
    replace_parameters_ = std::vector<bool>(parameters.size(), false);
    inputs_num_ = std::vector<size_t>(parameters.size(), 1);
    auto node_fg = node->func_graph();

    for (size_t i = 1; i < inputs.size(); ++i) {
      if (IsPrimitiveCNode(inputs[i], prim::kPrimMakeTuple) || IsCNodeGraphKernel(inputs[i])) {
        Process(node_fg, cnode, parameters[i - 1], i - 1);
      } else {
        args_.push_back(inputs[i]);
      }
    }

    if (!need_update_) {
      return nullptr;
    }

    FuncGraphPtr new_fg = TransformableClone(fg, std::make_shared<TraceTransform>("sp"));
    mng->AddFuncGraph(new_fg);

    auto node_users = mng->node_users();
    std::vector<AnfNodePtr> new_fg_parameters = new_fg->parameters();
    std::vector<AnfNodePtr> new_parameters;
    size_t curr_input_idx{0};
    for (size_t param_i = 0; param_i < new_fg_parameters.size(); ++param_i) {
      if (!replace_parameters_[param_i]) {
        if (parameters[param_i]->abstract() != nullptr) {
          new_fg_parameters[param_i]->set_abstract(parameters[param_i]->abstract());
        }
        new_parameters.push_back(new_fg_parameters[param_i]);
        curr_input_idx++;
        continue;
      }

      // make a new parameter.
      for (size_t input_i = 0; input_i < inputs_num_[param_i]; ++input_i) {
        auto new_param = std::make_shared<Parameter>(new_fg);
        new_param->set_abstract(args_.at(curr_input_idx)->abstract());

        // update users of new parameter.
        for (auto &user : node_users[new_fg_parameters[param_i]]) {
          idx_ = -1;
          AnfVisitor::Match(prim::kPrimTupleGetItem, {IsParam, IsValueNode<Int64Imm>})(user.first);
          if (idx_ == -1) {
            MS_LOG(ERROR) << "User of: " << new_fg_parameters[param_i]->DebugString()
                          << " must be tuple getitem here, but got: " << user.first->DebugString();
            return nullptr;
          }

          if (input_i == LongToSize(idx_)) {
            for (auto &sub_user : node_users[user.first]) {
              auto sub_user_cnode = sub_user.first->cast<CNodePtr>();
              MS_EXCEPTION_IF_NULL(sub_user_cnode);
              sub_user_cnode->set_input(sub_user.second, new_param);
              (void)mng->Replace(sub_user.first, sub_user_cnode);
            }
          }
        }

        new_parameters.push_back(new_param);
        curr_input_idx++;
      }
    }

    mng->SetParameters(new_fg, new_parameters);
    (void)args_.insert(args_.begin(), NewValueNode(new_fg));
    auto new_call = node_fg->NewCNode(args_);
    new_call->set_abstract(node->abstract());
    return new_call;
  }

  void Visit(const ValueNodePtr &vnode) override { idx_ = GetValue<int64_t>(vnode->value()); }

  void Visit(const CNodePtr &cnode) override {}

  void Reset() {
    replace_parameters_.clear();
    args_.clear();
    inputs_num_.clear();
    need_update_ = false;
    idx_ = -1;
  }

 private:
  std::vector<bool> replace_parameters_{};
  std::vector<AnfNodePtr> args_{};
  std::vector<size_t> inputs_num_{};
  bool need_update_{false};
  int64_t idx_{-1};
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
    if (idx_ == -1 || switch_ == nullptr || fg == nullptr ||
        (fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) && !ExistEnvNode(fg))) {
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
    if (MultipleUseOfSwitch(tuple_getitem->input(1), fg) && !ExistEnvNode(fg)) {
      return nullptr;
    }
    auto new_g1 = getitem_transform_(g1_, idx_);
    auto new_g2 = getitem_transform_(g2_, idx_);
    auto sw_node = fg->NewCNode({NewValueNode(prim::kPrimSwitch), x_, NewValueNode(new_g1), NewValueNode(new_g2)});
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
      auto fg_transform = getitem_transform_(graph, idx_);
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
      auto fg_transform = getitem_transform_(graph, idx_);
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
        incorporate_getitem_switch_(std::make_shared<IncorporateGetitemSwitch>()),
        incorporate_getitem_switch_layer_a_(std::make_shared<IncorporateGetitemSwitchLayerA>()),
        incorporate_getitem_switch_layer_b_(std::make_shared<IncorporateGetitemSwitchLayerB>()) {
    eliminaters_.emplace_back(incorporate_getitem_);
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
  OptimizerCallerPtr incorporate_getitem_, incorporate_getitem_switch_, incorporate_getitem_switch_layer_a_,
    incorporate_getitem_switch_layer_b_;
  std::vector<OptimizerCallerPtr> eliminaters_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_
